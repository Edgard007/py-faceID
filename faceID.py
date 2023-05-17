import cv2

# Cargar la imagen
image = cv2.imread("imagen.jpg")

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
cv2.namedWindow("faces", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
im = cv2.imread("imagen.jpg")                    # Read image
imS = cv2.resize(im, (960, 540))                # Resize image
cv2.imshow("faces", image)                       # Show image
cv2.waitKey(0)                                  # Display the image infinitely until any keypress