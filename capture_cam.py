import cv2

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    resultado = cv2.Canny(frame, 150, 250) # umbral inferior, umbral superior

    cam = cv2.flip(resultado, 1)

    cv2.imshow('frame', cam)
    
    tecla = cv2.waitKey(1)
    if tecla == 27:
        break

cap.release()
cv2.destroyAllWindows()