# from model import Model
import requests
import cv2
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import LogNorm

THRESH=0.2
def load_model():
    url = 'http://0.0.0.0:3000/models/load_model?model_name=./yolov8l.pt'

    # Encabezados de la solicitud
    headers = {
        'accept': 'application/json',
    }
    # Realizar la solicitud POST sin datos adjuntos
    response = requests.post(url, headers=headers)
    print(response)

def get_detecctions(image, rotation='0'):
    global THRESH
    params ={'thresh': THRESH, 'rotation': rotation}
    # URL de la solicitud POST
    url = F'http://0.0.0.0:3000/models/predict_image_detections'
    # Convertir el objeto de numpy a bytes
    retval, image_bytes = cv2.imencode('.png', image)

    # Verificar si la codificación fue exitosa
    if retval:
        image_bytes = image_bytes.tobytes()
    # Crear la estructura de datos para enviar la imagen en la solicitud POST
    files = {'image': ('far.png', image_bytes, 'image/png')}
    # Encabezados de la solicitud
    headers = {
        'accept': 'application/json',
    }
    # Realizar la solicitud POST
    response = requests.post(url, files=files, headers=headers, params=params)
    data = None
    # Obtener la respuesta
    if response.status_code == 200:
        try:
            # Intentar analizar la respuesta como JSON
            data = response.json()
            # print(data)  # Hacer algo con la respuesta recibida
            return data
        except json.JSONDecodeError:
            print("Error: La respuesta no es un JSON válido")
    else:
        print(f"Error: Código de estado de respuesta {response.status_code}")

class Coordenar_camara_plano():
    def __init__(self, file_name='/cfg'):
        try:
            file_name_xx = file_name+'/xx.pkl'
            file_name_yy = file_name+'/yy.pkl'
            with open(file_name_xx, 'rb') as archivo:
                self.xx = -pickle.load(archivo)
            with open(file_name_yy, 'rb') as archivo:
                self.yy = -pickle.load(archivo)
            plt.imshow(self.yy, norm=LogNorm())
            plt.show()
        except:
            print(f'No se ha encontrado el mapa de pesos en: {file_name}')
    def get_positions(self, data):
        h, w,_ = img.shape
        new_data = {'data':[]}
        for dat in data['data']:
            x, y = int(dat['detections'][0]*w), int(dat['detections'][1]*h)
            if dat['object']=='person':
                dat['coor']=(self.xx[y,x], self.yy[y,x])
                new_data['data'].append(dat)
            # print(self.xx[y,x], self.yy[y,x])
        print(new_data)
        return new_data

def procesar(img, data,color=[255, 255, 255], 
                  fuente=cv2.FONT_HERSHEY_SIMPLEX, 
                  escala=0.7, 
                  grosor=2):
    h, w,_ = img.shape
    print(data)
    for dat in data['data']:
        x, y = int(dat['detections'][0]*w), int(dat['detections'][1]*h)
        texto = f'x: {round(dat["coor"][0],2)}, y: {round(dat["coor"][1],2)} m'
        posicion = (x, y)
        cv2.putText(img, texto, posicion, fuente, escala, (0,0,0), grosor+2, cv2.LINE_AA)
        cv2.putText(img, texto, posicion, fuente, escala, color, grosor, cv2.LINE_AA)
    return img

if __name__=='__main__':

    load_model()

    # Mostrar inferencia:
    SHOW = False
    VIDEO=True
    IMGS_PATH = ''
    VIDEO_PATH = '/home/faraujo/siali_github/supervision/examples/video_indio_recortado.mp4'
    RESOLUCION = (4000, 2000)
    PATH_PKLS = '/home/faraujo/siali_github/camera-plano'
    coordenar = Coordenar_camara_plano(PATH_PKLS)
    if VIDEO is False:
        img_names = sorted(os.listdir(IMGS_PATH))
        for img_name in img_names:
            
            print('1. Analizando la imagen: {img_name}')
            img = cv2.imread(IMGS_PATH + '/'+img_name)
            data = get_detecctions(img)
            data = coordenar.get_positions(data)
            print('2. Pintando rois de barandilla y escalera')
            img = procesar(img, data)
            if SHOW:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img_rgb)
                plt.show()

    ### Para un video
    else:
        SAVE = True
        captura = cv2.VideoCapture(VIDEO_PATH)
        gametos_frame_anterior = []
        cont = 0
        while (captura.isOpened()):
            ret, img = captura.read() 
            if ret == True:
                cont+=1
                # if cont<1700:
                #     continue
                # Redimensiona la imagen
                img = cv2.resize(img, RESOLUCION)
                data = get_detecctions(img)
                
                data = coordenar.get_positions(data)
                img = cv2.resize(img,(1440, 720))
                imagen_inf = procesar(img, data)
                if SAVE is True:  
                    name = 1000
                    cv2.imwrite('./imagenes/{}.jpg'.format(name+cont), imagen_inf)
                if SHOW is True:
                    # Crear una ventana con tamaño personalizado
                    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Video', 1000, 1000)

                    # Mostrar la imagen en la ventana
                    cv2.imshow('Video', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    if cv2.waitKey(30) == ord('s'):
                        break
            else: break   
        captura.release()
        cv2.destroyAllWindows()