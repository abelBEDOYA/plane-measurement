import cv2
import numpy as np
from pyposition import Position
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
import pickle
import argparse
import plotly.graph_objects as go
import random


# Función para desdistorsionar todos los puntos de una imagen
def undistort_image_points(imgshape, camera_matrix, dist_coeffs):
    height, width =imgshape
    grid_x, grid_y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    points_original = np.vstack((grid_x.ravel(), grid_y.ravel())).T
    points_original = points_original.reshape(-1, 1, 2).astype(np.float32)
    inv_camera_matrix = np.linalg.inv(camera_matrix)
    # Desdistorsionar puntos
    points_undistorted = cv2.undistortPoints(points_original, camera_matrix, dist_coeffs)

    # Convertir puntos desdistorsionados a formato de matriz (ancho x alto x 2)
    points_undistorted = points_undistorted.reshape(height, width, 2)
    ones_column = np.ones_like(points_undistorted[:, :, :1])
    combined_array = np.concatenate((points_undistorted, ones_column), axis=-1)

    # print(point2d_x)
    fx = camera_matrix[0,0]
    fy = camera_matrix[1,1]
    cx = camera_matrix[0,-1]
    cy = camera_matrix[1,-1]
    combined_array[:,:,0] = combined_array[:,:,0] * fx + cx
    combined_array[:,:,1] = combined_array[:,:,1] * fy + cy
    tans  = np.dot(inv_camera_matrix, combined_array.reshape(-1, 3).T)
    fovx = tans[0,:]
    fovy = tans[1,:]
    fx0 = np.arctan(fovx)
    fy0 = np.arctan(fovy)
    fovx = fx0*180/np.pi
    fovy = fy0*180/np.pi
    fovx, fovy = fovx.reshape((height, width)), fovy.reshape((height, width))
    return fovx, fovy


def _____distX(altura, beta0, fovX, fovY):
    """Distancia a camara sobre el agua en el eje X
    pxy: pixel en el eje Y de la foto
    altura: float = altura de la camara sobre el agua
    beta: float = angulo de incidencia de la camara sobre el plano de algua
    gamma: float = angulo dividido por dos del fov de la camara en el eje Y de la foto
    """
    imgshape=fovY.shape
    fovY_rad = fovY*np.pi/180 
    fovX_rad = fovX*np.pi/180  
    
    ptex = np.tan(fovX_rad)
    ptez = np.tan(fovY_rad)
    x = -altura*ptex/(np.sin(beta0)+np.cos(beta0)*ptez)
    # M = np.array([[1,0,0],
    #               [0, np.cos(beta0),  -np.sin(beta0)],
    #               [0, np.sin(beta0), np.cos(beta0)]])
    # # M = np.array([[np.cos(beta0), 0, np.sin(beta0)],
    # #               [0, 1, 0],
    # #               [-np.sin(beta0), 0, np.cos(beta0)]])
    # # print(xyz.shape)
    # resultado = np.matmul(M, xyz)
    
    # x,y,z = resultado[0], resultado[1], resultado[2]
    
    # x,y,z = xyz[0], xyz[1], xyz[2]
    # # difx = xyz[0]-x
    # # plt.hist(difx)
    # # plt.show()
    # # plt.imshow(difx.reshape((1080, 1920)),cmap='viridis', norm=LogNorm())
    # # plt.show()
    # # dify = xyz[1]-y
    # # plt.imshow(dify.reshape((3000,4000)))
    # # plt.show()
    # # difz = xyz[2]-z
    # # plt.imshow(difz.reshape((3000,4000)))
    # # plt.show()
    # # print(np.where(z > 1))
    # # print(np.where(z < -1))
    # # print(z[np.where(z > 1)])
    # # print(z[np.where(z < -1)])
    # theta = np.arccos(z/1)
    # phi = np.arctan2(y,x)

    # # plt.imshow(theta.reshape(imgshape))
    # # plt.show()
    # # plt.imshow(phi.reshape(imgshape))
    # # plt.show()
    
    # # print(resultado.shape)
    # distx = -altura*np.tan(theta)*np.cos(phi)
    
    # denominador = np.tan(fovY_rad+beta0)
    # distx = altura/denominador
    # distx = -altura*np.tan(np.pi/2 + fovY_rad+beta0)*np.cos(fovX_rad)
    # distx = -altura*np.tan(np.pi/2 + new_fov_Y_rad+beta0)*np.cos(new_fov_X_rad)
    return x # distx.reshape(imgshape)


def dist(altura, beta0, fovX, fovY, gamma =0.0):
    """Distancia a camara sobre el agua en el eje Y
    pxx: pixel en el eje X de la foto
    altura: float = altura de la camara sobre el agua
    beta: float = angulo de incidencia de la camara sobre el plano de algua
    gamma: float = angulo dividido por dos del fov de la camara en el eje X de la foto
    """
    # fovX, fovY = fovx.astype(np.float64), fovY.astype(np.float64)
    fovX_rad = fovX*np.pi/180 
    fovY_rad = fovY*np.pi/180  
    # gamma=-0.08
    ptex = np.tan(fovX_rad)
    ptez = np.tan(fovY_rad)
    # y = -altura*(-np.cos(beta0)+np.sin(beta0)*ptez)/(np.sin(beta0)+np.cos(beta0)*ptez)
    # x = -altura*ptex/(np.sin(beta0)+np.cos(beta0)*ptez)
    
    x = -altura*(np.cos(gamma)*ptex+ptez*np.sin(gamma))/(np.sin(beta0)+np.cos(beta0)*(-np.sin(gamma)*ptex+np.cos(gamma)*ptez))
    y = -altura*(np.cos(beta0)+np.sin(beta0)*(+np.sin(gamma)*ptex-np.cos(gamma)*ptez))/(np.sin(beta0)+np.cos(beta0)*(-np.sin(gamma)*ptex+np.cos(gamma)*ptez))
    return x, y 


def get_mapa_areas(altura, beta0_deg, fovx, fovy, gamma=0.0):
    beta0 = beta0_deg*np.pi/180
    gamma = gamma*np.pi/180
    distxx, distyy = dist(altura, beta0, fovx, fovy, gamma=gamma)
    
    # Calcular la matriz de diferencias con la columna anterior
    diferencias_x = np.diff(distxx, axis=1)
    # Agregar la última columna de la matriz original a la matriz de diferencias
    ultima_columna_x = diferencias_x[:, -1].reshape((-1, 1))
    gradiente_x = np.hstack((diferencias_x, ultima_columna_x))
    
    
    diferencias_y = np.diff(distyy, axis=0)
    # Agregar la última columna de la matriz original a la matriz de diferencias
    ultima_columna_y = diferencias_y[-1, :].reshape((1, -1))
    gradiente_y = np.vstack((diferencias_y, ultima_columna_y))
    return abs(gradiente_y*gradiente_x)
    # mapa_areas= np.array([])
    # for i in tqdm(range(len(distxx))):
    #     if i == len(distxx)-1:
    #         mapa_areas = np.vstack((mapa_areas, columna))
    #     else:
    #         columna = np.array([])
    #         for j in range(len(distxx[0])):
    #             if j == len(distyy[0])-1:
    #                 columna = np.append(columna, area)
    #             else:
    #                 deltaX = distxx[i,j] -distxx[i+1,j]
    #                 deltaY = distyy[i,j] -distyy[i,j+1]
    #                 area = np.abs(deltaX * deltaY)
    #                 columna = np.append(columna, area)
    #         # print(deltaX, deltaY, area)
    #         if len(mapa_areas)==0:
    #                 mapa_areas = columna
    #         else:
    #             mapa_areas = np.vstack((mapa_areas, columna))

    # return mapa_areas

def subplots_fov(xx, yy, show=False, save=False):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    # Definir niveles de contorno para X e Y
    contour_levels_x = np.linspace(np.min(xx), np.max(xx), 40)
    contour_levels_y = np.linspace(np.min(yy), np.max(yy), 40)

    # Subgráfico 1
    axes[0, 0].contour(xx, levels=contour_levels_x, cmap='viridis')
    axes[0, 0].set_title('Isoclinas del FoV en X')
    axes[0, 0].axis('equal')
    axes[0, 0].invert_yaxis()
    # Subgráfico 2
    im = axes[0, 1].imshow(xx, cmap='viridis')
    axes[0, 1].set_title('Mapa del FoV X')
    axes[0, 1].axis('equal')

    # Subgráfico 3
    x = np.arange(0, yy.shape[1])
    y = np.arange(0, yy.shape[0])
    X, Y = np.meshgrid(x, y)
    axes[1, 0].contour(X, Y, yy, levels=contour_levels_y, cmap='viridis')
    axes[1, 0].set_title('Isoclinas del FoV en Y')
    axes[1, 0].axis('equal')
    axes[1, 0].invert_yaxis()

    # Subgráfico 4
    im = axes[1, 1].imshow(yy, cmap='viridis')
    axes[1, 1].set_title('Mapa del FoV Y')
    axes[1, 1].axis('equal')

    # Añadir etiquetas globales en ambos ejes
    for ax in axes.flat:
        ax.set(xlabel='Pixeles', ylabel='Pixeles')

    # Añadir leyenda con escala de color
    fig.colorbar(im, ax=axes, orientation='vertical', label='Ángulo')

    # Título global
    fig.suptitle('FoV intrínseco según distorsión de lente', fontsize=16)

    if save:
        fig.savefig(f'{PATH}fovs_intrinsecos.png', dpi=500)
    if show:
        plt.show()
    plt.close()

def subplots_XY(xx, yy, parameters=str(), show=False, save=False):
    
    
    # Crear una figura con 2 filas y 2 columnas de subgráficos
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    # Definir niveles de contorno para X e Y
    x_mean = np.mean(xx)
    y_mean = np.mean(yy)
    x_parts = int((np.max(xx)-np.min(xx))/((x_mean-np.min(xx))/4))
    y_parts = int((np.max(yy)-np.min(yy))/((y_mean-np.min(yy))/4))
    if np.max(xx)>1000:
        contour_levels_x = np.linspace(0, 800, 90)
        contour_levels_y = np.linspace(0, 800, 90)
    else:
        contour_levels_x = np.linspace(np.min(xx), np.max(yy), 40) # x_parts)
        contour_levels_y = np.linspace(np.min(yy), np.max(yy), 40)

    # Subgráfico 1
    axes[0, 0].contour(xx, levels=contour_levels_x, cmap='viridis')
    axes[0, 0].set_title('Grid X')
    axes[0, 0].axis('equal')
    axes[0, 0].invert_yaxis()

    # Subgráfico 2
    im = axes[0, 1].imshow(xx, cmap='viridis', norm=LogNorm())
    axes[0, 1].set_title('Mapa de coordenadas X')
    axes[0, 1].axis('equal')

    # Subgráfico 3
    x = np.arange(0, yy.shape[1])
    y = np.arange(0, yy.shape[0])
    X, Y = np.meshgrid(x, y)
    axes[1, 0].contour(X, Y, yy, levels=contour_levels_y, cmap='viridis')
    axes[1, 0].set_title('Grid Y')
    axes[1, 0].axis('equal')
    axes[1, 0].invert_yaxis()

    # Subgráfico 4
    im = axes[1, 1].imshow(yy, cmap='viridis', norm=LogNorm())
    axes[1, 1].set_title('Mapa de coordenadas Y')
    axes[1, 1].axis('equal')

    # Añadir etiquetas globales en ambos ejes
    for ax in axes.flat:
        ax.set(xlabel='Pixeles', ylabel='Pixeles')

    # Añadir leyenda con escala de color
    fig.colorbar(im, ax=axes, orientation='vertical', label='Valor coordenada')

    # Título global
    fig.suptitle(f'Grid de coords. absolutas XY en el plano. {parameters}', fontsize=16)

    if save:
        fig.savefig(f'{PATH}mapas_xy.png', dpi=500)
    if show:
        plt.show()

def plot_mapa_area(mapa_areas, title=str(), show=False, save=False):
    log_norm = LogNorm()
    max_value = np.max(mapa_areas) if np.max(mapa_areas)<10 else 10
    plt.imshow(np.where(mapa_areas > max_value, max_value, mapa_areas), cmap='viridis', norm=log_norm)
    plt.xlabel('pixel')
    plt.ylabel('pixel')
    cbar = plt.colorbar()
    cbar.set_label('area/px^2')
    plt.title(title, fontsize =10)
    # Mostrar la figura
    
    if save:
        plt.savefig(f'{PATH}mapa_area.png', dpi=500)
    if show:
        plt.show()

    plt.close()

def plot_3d(xx,yy, h):
    stacked_array = np.stack((xx, yy), axis=-1)

    tamano_celda_x = stacked_array.shape[0]//10
    num_celdas_x = (stacked_array.shape[0])//tamano_celda_x  # Número de celdas en el eje x
    tamano_celda_y = stacked_array.shape[0]//10
    num_celdas_y = (stacked_array.shape[1])//tamano_celda_y  # Número de celdas en el eje y

    # Crear una cuadrícula usando meshgrid
    x = np.arange(0, tamano_celda_x * num_celdas_x, tamano_celda_x)
    y = np.arange(0, tamano_celda_y * num_celdas_y, tamano_celda_y)
    # print('x', x)
    # print('y', y)
    xx, yy = np.meshgrid(x, y)
    # plt.imshow(xx)
    # plt.show()
    # Obtener la lista de puntos como pares (x, y)
    puntos_cuadricula = np.column_stack((xx.ravel(), yy.ravel()))
    # Punto fijo (0, 0, 43)
    fixed_point = np.array([0, 0, h])

    # Crear la figura 3D
    fig = go.Figure()

    # Graficar el punto fijo
    fig.add_trace(go.Scatter3d(
        x=[fixed_point[0]],
        y=[fixed_point[1]],
        z=[fixed_point[2]],
        mode='markers',
        marker=dict(size=5, color='red'),
        name=f'Punto Fijo (0, 0, {fixed_point[2]})'
    ))

    # print(muestra)
    # Graficar las rectas desde (0, 0, 43) hasta (x, y, 0)
    for xy in puntos_cuadricula[1:]:
        x, y = stacked_array[xy[0], xy[1]]
        # print(x,y)
        end_point = np.array([x, y, 0])
        fig.add_trace(go.Scatter3d(
            x=[fixed_point[0], end_point[0]],
            y=[fixed_point[1], end_point[1]],
            z=[fixed_point[2], end_point[2]],
            mode='lines+markers',
            marker=dict(size=1, color='blue'),
            name=f'({x}, {y}, 0)'
        ))

    # Configurar el diseño del gráfico
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Eje X'),
            yaxis=dict(title='Eje Y'),
            zaxis=dict(title='Eje Z'),
            aspectmode='data',
            aspectratio=dict(x=1, y=1, z=1)
        )
    )

    # Mostrar el gráfico
    fig.show()

if __name__=='__main__':
    
    SAVE_PLOTS = True
    SHOW = True
    PATH = './'
    parser = argparse.ArgumentParser(description='Calcular mapa de areas para una posicion dada de una camara respecto a un plano')
    parser.add_argument('--cfg_path', type=str, default='./cfg/', 
                        help='Ruta del archivo de configuración')
    parser.add_argument('--height', type=float, default=9, 
                        help='Altura en unidades que luego quieras')
    parser.add_argument('--angle', type=float, default=17, 
                        help='angulo de la camara respecto al plano')
    parser.add_argument('--roll', type=float, default=0, 
                        help='angulo de roll de la camara en torno a su optico')
    parser.add_argument('--imgshape', nargs=2, type=int, default=[1080, 1920], 
                        help='Dimensiones de la imagen en formato alto ancho')
    
    args = parser.parse_args()
    # Accede a los valores de los argumentos
    cfg_path = args.cfg_path
    altura = args.height
    beta0 = args.angle
    gamma = args.roll
    imgshape = tuple(args.imgshape)
    
    
    beta0_rad = beta0*np.pi/180
    gamma_rad=gamma*np.pi/180
    print('Abriendo cfgs calibracion de la camara')
    position = Position.from_cfg(cfg_path)
    position.imgshape = np.array(list(imgshape))
    print(position.camera_parameters)
    # position.camera_matrix[0,2]=np.float64(2000.2)
    # position.camera_matrix[1,2]=np.float64(1500.2)
    # # position.camera_matrix[0,0]*=1.05
    # # position.camera_matrix[1,1]*=1.05
    camera_matrix = position.camera_matrix
    dist_coeffs = position.dist_coeffs  # *0.8

    print(camera_matrix)
    print(f'\n Configuracion tomada:') 
    print(f'\t -Intrinseca: \n \t  \t  {cfg_path}')
    print(camera_matrix, '\n')
    print(f'\t -Extrinseca:  \n \t  \t altura: {altura} \n \t  \t angulo: {beta0} \n')

    print('1. Calculando FoVs de distorsion de lente')
    fovx, fovy = undistort_image_points(imgshape, camera_matrix, dist_coeffs)
    subplots_fov(fovx, fovy,
                save=SAVE_PLOTS,
                show=SHOW)

    print('2. Calculando mapas de distancias:')
    
    
    print('\t 2.1 Calculando mapa X e Y')
    XX, YY = dist(altura, beta0_rad, fovx, fovy, gamma=gamma_rad)
    pickle.dump(XX, open(PATH+'xx'+'.pkl', "wb" ))
    pickle.dump(YY, open(PATH+'yy'+'.pkl', "wb" ))
    plot_3d(XX,YY, altura)
    max_value = 500
    subplots_XY(np.where(np.abs(XX) > max_value, max_value, np.abs(XX)), 
                np.where(np.abs(YY) > max_value, max_value, np.abs(YY)),
                parameters= f'altura: {altura}, angulo: {beta0}, roll: {gamma}',
                save=SAVE_PLOTS,
                show=SHOW)

    print('3. Generando mapa de area a partir de mapas X e Y')
    mapa_area = get_mapa_areas(altura, beta0, fovx, fovy, gamma=gamma)
    title=f'Mapa area: Altura: {altura} m, incidencia: {beta0}, roll: {gamma} deg'
    plot_mapa_area(mapa_area, title=title,
                save=SAVE_PLOTS,
                show=SHOW)

    print(f'4. Guardado pesos en: {PATH}')
    name = f'mapa_areas_altura{altura}_angulo{beta0}_roll{gamma}'.replace('.', '-')
    pickle.dump(mapa_area, open(PATH+name+'.pkl', "wb" ))
    
    print(f'\n Mapa de areas guardado correctamente: \n {PATH+name}')
    