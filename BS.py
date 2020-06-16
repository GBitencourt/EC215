import cv2
import argparse
import numpy as np

capt = cv2.VideoCapture('test.mp4')

frame_width = int(capt.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height =int(capt.get(cv2.CAP_PROP_FRAME_HEIGHT))

#Parte do código para o background subtraction
parser = argparse.ArgumentParser(description='Movement Detector.')
parser.add_argument('--input', type=str, help='Caminho para o vídeo.', default='test.mp4')
parser.add_argument('--algo', type=str, help='Método da subtração de fundo.', default='MOG2')
args = parser.parse_args()

#Verifica o método do background subtraction
if args.algo == 'MOG2':
    bS = cv2.createBackgroundSubtractorMOG2() #guassiana de segmentação
else:
    bS = cv2.createBackgroundSubtractorKNN() #vizinho proximo

capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))
ret, quadro1 = capt.read()
print(quadro1.shape)

while capt.isOpened():

    ret, quadro = capture.read()
    if quadro is None:
        break
    bw = bS.apply(quadro)

    cv2.rectangle(quadro, (10, 2), (100, 20), (255, 255, 255), -1)
    cv2.putText(quadro, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    diff = cv2.absdiff(quadro1, quadro)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Cria o contorno do retângulo quando detecta o movimento
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 900:
            continue
        cv2.rectangle(quadro1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Movement Detection", quadro1)
    cv2.imshow('Background', bw)
    quadro1 = quadro
    ret, quadro = capt.read()

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
