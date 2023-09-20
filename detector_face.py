import cv2

cap = cv2.VideoCapture(0)

## leemos el modelo en coffemodel
net = cv2.dnn.readNetFromCaffe('opencv_face_detector.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

## parametros del modelo
ancho = 300
alto = 300
## valores medios de los canales
mean = [104, 117, 123]
umbral = 0.7

while True:
    ## leemos la imagen
    ret, frame = cap.read()

    ## si encuentra un error 
    if not ret:
        break
    ## cambio de forma 
    frame = cv2.flip(frame, 1)

    ## infor de los frames 
    altoframe = frame.shape[0]
    anchoframe = frame.shape[1]

    ## Preprocesamos la imagen
    ## Images - Factor de escala - tamaño - media de color - Formato de color(BGR-RGB) - Recorte
    blob = cv2.dnn.blobFromImage(frame, 1.0, (ancho, alto), mean, swapRB=False, crop=False)

    ## se carga el modelo
    net.setInput(blob)
    deteccion = net.forward()

    ## iteramos sobre las detecciones
    for i in range(deteccion.shape[2]):
        ## obtenemos la probabilidad
        prob = deteccion[0, 0, i, 2]
        ## 70% de probabilidad de que sea un rostro
        if prob > umbral:
            ## obtenemos las coordenadas
            xmin = int(deteccion[0, 0, i, 3] * anchoframe)
            ymin = int(deteccion[0, 0, i, 4] * altoframe)
            xmax = int(deteccion[0, 0, i, 5] * anchoframe)
            ymax = int(deteccion[0, 0, i, 6] * altoframe)
            ## dibujamos un rectangulo
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            ## texto con la probabilidad 
            texto = '{:.2f}% rico'.format(prob * 100)
            ## tamaño del texto
            tamanio, base_letra = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ## colocamos fondo al texto
            cv2.rectangle(frame, (xmin, ymin - tamanio[1]), (xmin + tamanio[0], ymin + base_letra), (0, 0, 0), cv2.FILLED)
            ## colocamos el texto
            cv2.putText(frame, texto, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('frame', frame)

    tecla = cv2.waitKey(1)
    if tecla == 27:
        break

cap.release()
cv2.destroyAllWindows()