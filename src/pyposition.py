import cv2
import numpy as np
import pickle
from scipy.spatial.transform import Rotation

class Position():
    def __init__(self, camera_matrix = None, dist_coeffs = None):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.imgshape =None

    @classmethod
    def from_cfg(cls, path):
        """
        Instanciarlo a partir de la ruta a los archivos pickle 
        que tienen la matriz de calibración y los coeficientes de
        distorson de lente
        """
        with open(path + "cameraMatrix.pkl", 'rb') as archivo:
            camera_matrix = pickle.load(archivo)
        with open(path + "dist.pkl", 'rb') as archivo:
            dist_coeffs = pickle.load(archivo)
        
        return cls(camera_matrix, dist_coeffs)

    @property
    def camera_parameters(self):
        if self.imgshape is None:
            mensaje_error = "Valor de self.imgshape no definido. Es necesario poner en el atributo imgshape"\
                            "las dimensiones de la imagen"
            raise ValueError(mensaje_error)
        inv_camera_matrix = np.linalg.inv(self.camera_matrix)
        x0, x1, x2 = 0, self.imgshape[1]//2, self.imgshape[1]
        y0, y1, y2 = 0, self.imgshape[0]//2, self.imgshape[0]
        point2d_x = np.array([[x0, y1], [x2, y1]], dtype=np.float64)
        point2d_y = np.array([[x1, y0], [x1, y2]], dtype=np.float64)
        point2d_x = cv2.undistortPoints(point2d_x, self.camera_matrix, self.dist_coeffs)[:,0,:]
        point2d_y = cv2.undistortPoints(point2d_y, self.camera_matrix, self.dist_coeffs)[:,0,:]
        
        point2d_x = np.hstack((point2d_x, np.ones((point2d_x.shape[0], 1))))
        point2d_y = np.hstack((point2d_y, np.ones((point2d_y.shape[0], 1))))
        fx = self.camera_matrix[0,0]
        fy = self.camera_matrix[1,1]
        cx = self.camera_matrix[0,-1]
        cy = self.camera_matrix[1,-1]
        point2d_x[:,0] = point2d_x[:,0] * fx + cx
        point2d_x[:,1] = point2d_x[:,1] * fy + cy
        point2d_y[:,0] = point2d_y[:,0] * fx + cx
        point2d_y[:,1] = point2d_y[:,1] * fy + cy

        fx0, fx2 =np.dot(inv_camera_matrix, point2d_x.T).T[:,0]
        fy0, fy2 =np.dot(inv_camera_matrix, point2d_y.T).T[:,1]
        camera_parameters = {
            'imgshape': self.imgshape,
            'cameramatrix': self.camera_matrix,
            'inv_cameramatrix': inv_camera_matrix,
            'distcoefs': self.dist_coeffs, 
            'fovx': (np.arctan(fx2)-np.arctan(fx0))*180/np.pi, 
            'fovy': (np.arctan(fy2)-np.arctan(fy0))*180/np.pi
        }
        return camera_parameters

    def get_3d_to_2d(self, points_3d, img_shape=(1,1)):
        """
        Transforma putos de coordenadas 3D (X,Y,Z) a 2D (x,y) en pixeles.
        """
        points_2d = []
        for point_3d in points_3d:
            point_3d = np.array(point_3d, dtype=np.float32)
            # Proyectar el punto 3D en el plano de la imagen
            point_2d, _ = cv2.projectPoints(point_3d.reshape(1, 3), np.zeros(3), np.zeros(3), self.camera_matrix, self.dist_coeffs)
            # Coordenadas en el plano de la imagen (x, y)
            x_image, y_image = point_2d[0][0]
            points_2d.append((x_image, y_image))
        return points_2d

    def get_2d_to_3d(self, Zs, points_2d, img_shape=(1,1)):
        """
        Transforma putos de coordenadas 2D (x,y) en pixeles a 3D (X,Y,Z) en distancia real. 
        Requiere el Z en distancia real.
        """
        inv_camera_matrix = np.linalg.inv(self.camera_matrix)
        points_3d = []
        for i, point_2d in enumerate(points_2d):
            x_px, y_px = point_2d
            image_coords_homogeneous = np.array([x_px, y_px, 1])

            # Corrección de distorsión usando cv2.undistortPoints
            distorted_point = np.array([[x_px, y_px]])
            punto_desdistorsionado = cv2.undistortPoints(distorted_point, self.camera_matrix, self.dist_coeffs)[0][0]
            fx = self.camera_matrix[0,0]
            fy = self.camera_matrix[1,1]
            cx = self.camera_matrix[0,-1]
            cy = self.camera_matrix[1,-1]
            undistorted_pixel_x = punto_desdistorsionado[0] * fx + cx
            undistorted_pixel_y = punto_desdistorsionado[1] * fy + cy

            # Vector de coordenadas homogéneas (x, y, 1) del punto corregido en la imagen
            image_coords_homogeneous = np.array([undistorted_pixel_x, undistorted_pixel_y, 1])
            world_coords_homogeneous = np.dot(inv_camera_matrix, image_coords_homogeneous) * Zs[i]
            # Coordenadas 3D (X, Y, Z) del punto en el mundo real en centímetros
            X, Y, Z = world_coords_homogeneous[:3]
            points_3d.append((X, Y, Z))
        return points_3d

    def get_plot(self, img, points_2d, points_3d, text = True, color = (255,0,0)):
        """
        Dibuja sobre la img los puntos 2d y escribe en texto el 3D asociado, y la devuelve
        """
        img_copy = np.copy(img)
        for point_2d, point_3d in zip(points_2d, points_3d):
            px = tuple(map(int, point_2d))
            px_text = tuple(map(int, np.array(point_2d) + 20))
            cv2.circle(img_copy, px, 7, color, -1)
            text2write = f"({point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f})"
            if text:
                cv2.putText(img_copy, text2write, px_text, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return img_copy

    def get_6d(self, det_2Dkeypoints, real_3Dkeypoints):
        """
        Calcula los 6 parametros que definen posicion y orientación del objeto, el 6D del objeto.
        Args:
            det_2Dkeypoints: list = detecciones de los kepoints en la imagen 2d ordenados
            real_3Dkeypoints: list = puntos tridimensionales reales en el mismo orden que las detecciones
        
        """
        ret, rvecs, tvecs = cv2.solvePnP(real_3Dkeypoints, det_2Dkeypoints, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        return rvecs, tvecs
    
    def get_6d_plot(self,frame, rvecs, tvecs):
        """
        Calcula los 6 parametros que definen posicion y orientación del objeto, el 6D del objeto.
        Args:
            det_2Dkeypoints: list = detecciones de los kepoints en la imagen 2d ordenados
            real_3Dkeypoints: list = puntos tridimensionales reales en el mismo orden que las detecciones
        """
        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = self.__draw_axis(frame, rvecs, tvecs)
        frame = self.__euler_text(frame, rvecs, tvecs)
        # Colocar el texto en la imagen
        return frame
    
    def __draw_axis(self, frame, rvecs, tvecs,
                    scale_factor=2.0
                    ):
        # Encontrar las esquinas del tablero
        pto_centro = (int(self.camera_matrix[0][2]), int(self.camera_matrix[1][2]))
        cv2.circle(frame, pto_centro, 6, (0, 0 ,255), -1)
        # Dibujar ejes en la imagen
        axis = np.float32([[0, 0, 0], [scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, scale_factor]]).reshape(-1, 3)
        imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, self.camera_matrix, self.dist_coeffs)
        origen_obj = (int(imgpts[0][0][0]), int(imgpts[0][0][1]))
        cv2.circle(frame, origen_obj, 6, (0, 0, 255), -1)
        cv2.line(frame, pto_centro, origen_obj, (0 ,0, 255), 2)
        frame = cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvecs, tvecs, 300)
        return frame

    def __euler_text(
            self, frame, rvecs, tvecs,
            text_position_r=(30, 50),
            text_position_t=(30, 80),
            text_color = (255, 255, 255),
            rectangle_position= (30,30),
            text_box_width=600,
            text_box_height=60,
            text_scale = 0.8,
            font = cv2.FONT_HERSHEY_SIMPLEX,
            text_thickness = 1            
            ):
        # Crear un texto con los ángulos de Euler
        rotation_matrix, _ = cv2.Rodrigues(rvecs)
        r = Rotation.from_matrix(rotation_matrix)
        # Obtener los ángulos de roll, pitch y yaw
        roll, pitch, yaw = r.as_euler('ZYX', degrees=True)
        text_r = f"(ZXY)roll: {float(roll):+.1f} pitch: {float(pitch):+.1f} yaw: {float(yaw):+.1f} (grad)"
        text_t = f"X: {float(tvecs[0]):+.1f} Y: {float(tvecs[1]):+.1f} Z: {float(tvecs[2]):+.1f} (mm)"
        cv2.rectangle(frame, rectangle_position, (rectangle_position[0] + text_box_width, rectangle_position[1] + text_box_height), (0,0,0), -1)
        cv2.putText(frame, text_r, text_position_r, font, text_scale, text_color, text_thickness, lineType=cv2.LINE_AA)
        cv2.putText(frame, text_t, text_position_t, font, text_scale, text_color, text_thickness, lineType=cv2.LINE_AA)
        return frame
