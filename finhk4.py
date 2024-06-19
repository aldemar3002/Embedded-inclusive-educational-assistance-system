import os
import time
import cv2
import numpy as np
import pygame
import tempfile
import pyautogui
import mediapipe as mp
import speech_recognition as sr
from gtts import gTTS
from google.cloud import vision, language_v1
import pyttsx3

# Configurar la ruta a Tesseract y las credenciales de Google en Windows
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\maria\Downloads\psychic-iridium-426821-k3-250cf6f5b095.json"

# Inicializar pygame para reproducir audio
pygame.mixer.init()

# Clientes de Google Cloud
client_vision = vision.ImageAnnotatorClient()
client_language = language_v1.LanguageServiceClient()

# Configuración de pyttsx3 para convertir texto a voz
engine = pyttsx3.init()

def speak_text(text):
    """Convierte texto a voz y lo reproduce."""
    engine.say(text)
    engine.runAndWait()

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
        pygame.mixer.music.unload()  # Asegurarse de que el archivo no esté siendo utilizado
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
        
        # Guardar imagen en un archivo temporal para enviarla a Google Cloud Vision
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_filename = temp_file.name
            cv2.imwrite(temp_filename, gray)
        
        # Leer la imagen y procesarla con Google Cloud Vision OCR
        with open(temp_filename, 'rb') as image_file:
            content = image_file.read()
            image = vision.Image(content=content)
            response = client_vision.text_detection(image=image)
            text = response.text_annotations[0].description if response.text_annotations else ""
        
        # Eliminar el archivo temporal
        os.remove(temp_filename)
        
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
    capture_interval = 1
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

def main3(ip):
    def get_image_description(frame):
        """Obtiene descripciones de un frame usando Google Cloud Vision y mejora con Natural Language."""
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        image = vision.Image(content=image_bytes)
        response = client_vision.label_detection(image=image)
        labels = response.label_annotations
        descriptions = [label.description for label in labels]

        description_text = " ".join(descriptions) if descriptions else "No se pudieron identificar elementos distintivos en este cuadro."
        return analyze_text(description_text)

    def analyze_text(text):
        """Analiza el texto utilizando Google Cloud Natural Language para extraer entidades y mejorar la descripción."""
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        entities_response = client_language.analyze_entities(document=document)
        entities = entities_response.entities

        if entities:
            entity_names = [entity.name for entity in entities]
            enhanced_description = "En tu entorno pude reconocer lo siguiente  " + ", ".join(entity_names[:-1]) + " y " + entity_names[-1] + "."
        else:
            enhanced_description = text  # Usar texto original si no se encontraron entidades
        return enhanced_description

    def get_droidcam_frame(ip):
        """Obtiene un fotograma de DroidCam usando la IP especificada."""
        cap = cv2.VideoCapture(f"http://{ip}:4747/video")
        success, frame = cap.read()
        cap.release()
        return frame if success else None

    def process_droidcam_frame(ip, interval=20):
        """Procesa un fotograma de DroidCam cada intervalo de segundos."""
        while True:
            frame = get_droidcam_frame(ip)
            if frame is not None:
                description = get_image_description(frame)
                print(description)
                speak_text(description)
            else:
                error_message = "No se pudo obtener el fotograma de DroidCam."
                print(error_message)
                speak_text(error_message)
            time.sleep(interval)

    def listen_for_command(ip):
        """Escucha comandos de voz y actúa en consecuencia."""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Escuchando comando de voz...")
            try:
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio, language='es-ES')
                print(f"Reconocido: {text}")
                speak_text(f"Reconocido: {text}")
                if "dame contexto del video" in text.lower():
                    process_droidcam_frame(ip)
            except sr.UnknownValueError:
                error_message = "No se pudo entender el audio"
                print(error_message)
                speak_text(error_message)
            except sr.RequestError as e:
                error_message = f"Error de reconocimiento de voz; {e}"
                print(error_message)
                speak_text(error_message)

    listen_for_command(ip)

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
            if "aplicación 1" in command.lower() or "aplicación uno" in command.lower():
                main1()
            elif "aplicación 2" in command.lower() or "aplicación dos" in command.lower():
                main2()
            elif "aplicación 3" in command.lower() or "aplicación tres" in command.lower():
                main3("10.50.94.155")  # IP de DroidCam
            else:
                text_to_speech("Comando no reconocido")
        except sr.UnknownValueError:
            text_to_speech("No se pudo entender el audio")
        except sr.RequestError as e:
            text_to_speech(f"Error del servicio de reconocimiento de voz; {e}")

if __name__ == "__main__":
    text_to_speech("Aplicación iniciada. Di 'aplicación uno', 'aplicación dos' o 'aplicación tres' para abrir una aplicación.")
    recognize_speech_and_execute()
