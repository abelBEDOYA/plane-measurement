import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from PIL import Image
class Area:
    def __init__(self, file_name='/cfg/mapa_pesos.pkl'):
        try:
            with open(file_name, 'rb') as archivo:
                self.mapa_pesos = pickle.load(archivo)
            self.img_shape = self.mapa_pesos.shape
            self.area = 0
        except:
            print(f'No se ha encontrado el mapa de pesos en: {file_name}')

    def get_mask(self, contour):
        mask = np.zeros(self.img_shape, dtype=np.uint8)
        contour_points = np.int32(contour)
        cv2.fillPoly(mask, [contour_points], color=1)
        return mask

    def get_area(self, segmentation, verbose=False):
        ancho = self.img_shape[1]
        alto = self.img_shape[0]
        mask = self.get_mask(segmentation)
        # cv2.imwrite('/files/mascara.jpg', np.copy(mask)*255)
        region = mask*self.mapa_pesos
        # cv2.imwrite('/files/region.jpg', np.copy(region)*999)
        area = np.sum(region)
        self.area = area
        if verbose:
            print(f'\n El area es: {area}.Del alga {puntos_deteccion} \n')
        return area




path_img =  'folio_validacion/images/003_.jpg' # 'movil_abel/21cm0grados/IMG_20231214_113331.jpg'  #  # './movil_abel/img/folio72.jpeg'  # './examples/00-00-03.jpg'  # './00-00-03.jpg'
# path_mask = './mask.png'  #'movil_abel/21cm0grados/mask21.png'  #  './examples/mask.png'
path_mask = 'folio_validacion/images/mask003.png'
path_pesos= 'folio_validacion/mapa_areas_altura0-768_angulo45-0_roll-3-0.pkl'
# Cargar la imagen binaria
imagen_binaria = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
imagen_bgr = cv2.imread(path_img)
# Encontrar contornos en la imagen
contornos, jerarquia = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



# area = Area(file_name='./mapa_areas_altura9-0_angulo19-0.pkl')
area = Area(file_name=path_pesos)

# area = Area(file_name=  './anav_con_distorsion/norte/mapa_areas_altura9_angulo17.pkl')## './anav_con_distorsion/nortemapa_areas_altura9_angulo17.pkl')
# area = Area(file_name='./mapa_areas_altura9_angulo-5-0.pkl') 
# area = Area(file_name=  './mapa_areas_altura0-755_angulo90-0.pkl') # mapa_areas_altura9-0_angulo20-0.pkl
# Crear una imagen en blanco para la máscara

norm = LogNorm(vmin=area.mapa_pesos.min(), vmax=area.mapa_pesos.max())
scalar_map_normalized = norm(area.mapa_pesos)

# Seleccionar un mapa de colores (puedes cambiar 'viridis' por otro de tu elección)
cmap = cm.get_cmap('viridis')

# Aplicar el mapa de colores al mapa escalar normalizado
colored_scalar_map = cmap(scalar_map_normalized)

# Combinar la imagen y el mapa escalar
mapa_areas_colored = (colored_scalar_map[:, :, :3] * 255).astype(np.uint8)

weight_image1 = 0.4
weight_image2 = 1 - weight_image1

# Combinar las dos imágenes con los pesos especificados
imagen_bgr = cv2.addWeighted(imagen_bgr, weight_image1, mapa_areas_colored, weight_image2, 0)

mascara = np.zeros_like(imagen_binaria)

# Iterar a través de los contornos y crear máscaras binarias para cada uno
for i, contorno in enumerate(contornos):
    cv2.drawContours(imagen_bgr, [contorno], 0, (255, 255, 255), thickness=cv2.FILLED)
    a = area.get_area(contorno)
    M = cv2.moments(contorno)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    # Agregar texto al lado del contorno
    texto = f'Area: {round(a,5)}'
    size=4
    
    cv2.putText(imagen_bgr, texto, (cx-60, cy-40), cv2.FONT_HERSHEY_SIMPLEX, size, (0,0,0), int(size*2.8+2))
    cv2.putText(imagen_bgr, texto, (cx-60, cy-40), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), size)


# Mostrar la imagen original y las máscaras
cv2.namedWindow(path_pesos, cv2.WINDOW_NORMAL)

# Establecer el tamaño de la ventana a 720x720 píxeles
cv2.resizeWindow(path_pesos, 1500, 900)

# Mostrar la imagen en la ventana
cv2.imshow(path_pesos, imagen_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
