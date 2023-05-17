import cv2
from pymob import Camera

# Crear una instancia de la cámara del dispositivo móvil
camera = Camera()

# Iniciar la cámara
camera.start()

# Capturar la imagen de la cámara
image = camera.get_image()

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Cargar el clasificador de detección de caras pre-entrenado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Detectar caras en la imagen
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Dibujar rectángulos alrededor de las caras detectadas
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Mostrar la imagen con las caras detectadas
cv2.imshow("Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Detener la cámara
camera.stop()
