import os
import time
import cv2
import numpy as np
import pytesseract
import pygame
import tempfile
import pyautogui
import mediapipe as mp
import speech_recognition as sr
from gtts import gTTS

# Configurar la ruta a Tesseract en Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\maria\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Inicializar pygame para reproducir audio
pygame.mixer.init()

# Función para convertir texto a voz
def text_to_speech(text, lang='es'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_filename = temp_file.name
            tts.save(temp_filename)
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)
        os.remove(temp_filename)
    except Exception as e:
        print(f"Error en text_to_speech: {e}")

def process_frame_and_extract_text(frame):
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        red_only = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(red_only, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.equalizeHist(gray)
        binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        inverted_binary_image = cv2.bitwise_not(binary_image)
        custom_config = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
        text = pytesseract.image_to_string(inverted_binary_image, lang='eng', config=custom_config)
        return text.strip()
    except Exception as e:
        print(f"Error al procesar el frame: {e}")
        return ""

def main1():
    droidcam_ip = '10.50.94.155'
    droidcam_port = '4747'
    stream_url = f'http://{droidcam_ip}:{droidcam_port}/video'
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        text_to_speech("Error: No se pudo abrir el stream de video.")
        return
    text_to_speech("Aplicación uno iniciada. Presiona Ctrl+C para salir.")
    last_capture_time = time.time()
    capture_interval = 5
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                text_to_speech("Error: No se pudo capturar la imagen.")
                break
            current_time = time.time()
            if current_time - last_capture_time >= capture_interval:
                text = process_frame_and_extract_text(frame)
                if text:
                    text_to_speech(f"Texto extraído: {text}")
                last_capture_time = current_time
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        text_to_speech("Interrupción del usuario. Saliendo del programa...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main2():
    camara = cv2.VideoCapture(0)
    lector_puntos_faciales = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    ancho_pantalla, alto_pantalla = pyautogui.size()
    text_to_speech("Inicio del seguimiento facial.")
    while True:
        _, frame = camara.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        fotograma_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = lector_puntos_faciales.process(fotograma_rgb)
        puntos_faciales = resultado.multi_face_landmarks
        alto_fotograma, ancho_fotograma, _ = frame.shape
        if puntos_faciales:
            puntos = puntos_faciales[0].landmark
            for id, punto in enumerate(puntos[474:478]):
                x = int(punto.x * ancho_fotograma)
                y = int(punto.y * alto_fotograma)
                cv2.circle(frame, (x, y), 3, (0, 255, 0))
                if id == 1:
                    x_pantalla = ancho_pantalla / ancho_fotograma * x
                    y_pantalla = alto_pantalla / alto_fotograma * y
                    pyautogui.moveTo(x_pantalla, y_pantalla)
            ojo_izquierdo = [puntos[145], puntos[159]]
            for punto in ojo_izquierdo:
                x = int(punto.x * ancho_fotograma)
                y = int(punto.y * alto_fotograma)
                cv2.circle(frame, (x, y), 3, (0, 255, 255))
            if (ojo_izquierdo[0].y - ojo_izquierdo[1].y) < 0.008:
                pyautogui.click()
                pyautogui.sleep(1)
        if cv2.waitKey(1) == ord('q'):
            break
    camara.release()
    cv2.destroyAllWindows()
    text_to_speech("Fin del seguimiento facial.")

def recognize_speech_and_execute():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        text_to_speech("Escuchando el comando...")
        print("Escuchando el comando...")
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio, language='es-ES')
            text_to_speech(f"Comando reconocido: {command}")
            print(f"Comando reconocido: {command}")
            if "aplicación uno" in command.lower():
                main1()
            elif "aplicación dos" in command.lower():
                main2()
            else:
                text_to_speech("Comando no reconocido")
        except sr.UnknownValueError:
            text_to_speech("No se pudo entender el audio")
        except sr.RequestError as e:
            text_to_speech(f"Error del servicio de reconocimiento de voz; {e}")

if _name_ == "_main_":
    text_to_speech("Aplicación iniciada. Di 'aplicación uno' o 'aplicación dos' para abrir una aplicación.")
    recognize_speech_and_execute()
