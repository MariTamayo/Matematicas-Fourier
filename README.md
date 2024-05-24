import cv2
import numpy as np

def aplicar_filtro(frame, tipo_filtro):
    # Convertir el frame a escala de grises
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Realizar DFT (Transformada Discreta de Fourier)
    dft = cv2.dft(np.float32(gris), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Crear una máscara basada en el tipo de filtro
    filas, columnas = frame.shape[:2]
    mascara = np.zeros((filas, columnas, 2), np.uint8)
    centro_filas, centro_columnas = int(filas / 2), int(columnas / 2)
    frecuencia_corte = 30  # Ajustar este valor para cambiar la frecuencia de corte

    if tipo_filtro == 'pasa_bajas':
        mascara[centro_filas - frecuencia_corte:centro_filas + frecuencia_corte,
                centro_columnas - frecuencia_corte:centro_columnas + frecuencia_corte] = 1
    elif tipo_filtro == 'pasa_altas':
        mascara[centro_filas - frecuencia_corte:centro_filas + frecuencia_corte,
                centro_columnas - frecuencia_corte:centro_columnas + frecuencia_corte] = 0
        mascara[0:filas, 0:columnas] = 1

    # Aplicar la máscara a la DFT
    dft_filtrada = dft_shift * mascara
    dft_filtrada_shift = np.fft.ifftshift(dft_filtrada)

    # Realizar la DFT inversa
    frame_filtrado = cv2.idft(dft_filtrada_shift)
    frame_filtrado = cv2.magnitude(frame_filtrado[:, :, 0], frame_filtrado[:, :, 1])

    # Normalizar el frame filtrado
    frame_filtrado = cv2.normalize(frame_filtrado, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return frame_filtrado

# Abrir la captura de video
cap = cv2.VideoCapture(0)

while True:
    # Leer frame de la captura de video
    ret, frame = cap.read()

    if not ret:
        break

    # Aplicar filtro pasa bajas al frame
    frame_filtrado = aplicar_filtro(frame, 'pasa_bajas')

    # Mostrar el frame filtrado
    cv2.imshow('Video Filtrado', frame)

    # Romper el bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
