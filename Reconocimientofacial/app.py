import cv2
import numpy as np
import streamlit as st
from PIL import Image
import face_recognition
import pandas as pd

# Establecer el color de fondo pastel
background_color = "#F5E6CC"

# Estilos CSS personalizados
st.markdown(
    f"""
    <style>
    .title {{
        font-size: 36px;
        font-weight: bold;
        color: #333333;
        text-align: center;
        margin-bottom: 20px;
    }}

    .subtitle {{
        font-size: 20px;
        color: #666666;
        text-align: center;
        margin-bottom: 20px;
    }}

    .upload-btn {{
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }}

    .upload-btn:hover {{
        background-color: #45a049;
    }}

    body {{
        background-color: {background_color};
        margin: 0;
        padding: 0;
    }}

    .stApp {{
        width: 90%;
        max-width: 800px;
        margin: auto;
        padding-top: 50px;
        padding-bottom: 50px;
    }}

    .stButton {{
        display: block;
        margin: 0 auto;
    }}

    .stDataFrame {{
        margin-top: 20px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Título
st.markdown("<h1 class='title'>Sistema de Reconocimiento Facial</h1>", unsafe_allow_html=True)


# Subtítulo
st.markdown("<p class='subtitle'> Este es un sistema de reconocimiento facial con Streamlit que toma asistencia automaticamente. Sube las imágenes que quieres reconocer de 1 por persona (máximo 30):</p>", unsafe_allow_html=True)

# Subir imágenes para reconocer
uploaded_images = st.file_uploader(" ", accept_multiple_files=True, type=["jpg", "jpeg", "png"], key="upload-btn")

# Lista para almacenar codificaciones de rostros de imágenes cargadas
known_face_encodings = []
# Lista para almacenar nombres de personas asociados con las imágenes cargadas
known_face_names = []

# DataFrame para almacenar la asistencia
if 'asistencia_df' not in st.session_state:
    st.session_state.asistencia_df = pd.DataFrame(columns=['Nombre', 'Estado'])

if uploaded_images is not None:
    if len(uploaded_images) > 30:
        st.warning("Solo se permiten un máximo de 30 imágenes. Solo se procesarán las primeras 30 imágenes.")
        uploaded_images = uploaded_images[:30]

    for uploaded_image in uploaded_images:
        image = np.array(Image.open(uploaded_image))
        st.image(image, caption=f"Imagen cargada: {uploaded_image.name}", use_column_width=True)

        # Convertir la imagen a RGB y encontrar codificaciones de rostros
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_image)

        if len(face_encodings) > 0:
            # Tomar solo la primera codificación de rostro (asumiendo que solo hay un rostro en la imagen cargada)
            known_face_encodings.append(face_encodings[0])
            # Usar el nombre de archivo como nombre de la persona asociada al rostro
            known_face_names.append(uploaded_image.name)

        if len(known_face_encodings) >= 30:
            break  # Limitar a 30 imágenes procesadas

    st.success(f"{len(known_face_encodings)} imágenes procesadas y rostros codificados.")

# Función para registrar la asistencia
def registrar_asistencia(nombre):
    if nombre not in st.session_state.asistencia_df['Nombre'].values:
        nuevo_registro = pd.DataFrame({'Nombre': [nombre], 'Estado': ['Presente']})
        st.session_state.asistencia_df = pd.concat([st.session_state.asistencia_df, nuevo_registro], ignore_index=True)

# Función para mostrar la cámara en vivo y realizar el reconocimiento facial
#Esta función se encarga de capturar video en tiempo real desde la cámara, procesar los fotogramas para detectar rostros y realizar el reconocimiento facial, mostrando los resultados en la interfaz de usuario de Streamlit.
def run_recognition():

    #Se inicializa la captura de video desde la cámara. El 0 indica que se está utilizando la cámara por defecto del sistema.
    video_capture = cv2.VideoCapture(0)

    #Se crean marcadores de posición en la interfaz de Streamlit para el video en tiempo real (frame_placeholder) y para la tabla de asistencia (table_placeholder).
    frame_placeholder = st.empty()
    table_placeholder = st.empty()

#Se inicia un bucle que se ejecuta continuamente mientras st.session_state.stop_camera sea False. Esto permite que el video en tiempo real continúe hasta que el usuario decida detenerlo.
    while not st.session_state.stop_camera:

       #Se captura un fotograma de la cámara. ret es un indicador de éxito de la captura, y frame contiene la imagen del fotograma.
       #Si la captura falla (ret es False), se muestra un mensaje de error y se rompe el bucle.

        ret, frame = video_capture.read()
        if not ret:
            st.error("Error al acceder a la cámara.")
            break

#OpenCV captura imágenes en formato BGR (azul, verde, rojo), pero la biblioteca face_recognition espera imágenes en formato RGB (rojo, verde, azul). Esta línea convierte el fotograma al formato esperado.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar rostros en el marco actual
       #face_locations encuentra las ubicaciones de todos los rostros en el fotograma.
       #face_encodings obtiene las codificaciones de los rostros en las ubicaciones detectadas.
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)


# Comparar las codificaciones de rostros detectadas con las codificaciones de rostros conocidas
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Comparar las codificaciones de rostros detectadas con las codificaciones de rostros conocidas
            if len(known_face_encodings) > 0:
                # Encontrar la coincidencia más cercana (similitud)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if face_distances[best_match_index] < 0.6:
                    name = known_face_names[best_match_index]
                    registrar_asistencia(name)
                else:
                    name = "Desconocido"
            else:
                name = "Desconocido"

            # Dibujar el cuadro del rostro y el nombre asociado
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Mostrar el video en tiempo real en Streamlit
        frame_placeholder.image(frame, channels="BGR")


        # Actualizar y mostrar la tabla de asistencia
        table_placeholder.dataframe(st.session_state.asistencia_df)

    video_capture.release()




# Botón para activar la cámara
if st.button("Activar cámara"):
    st.session_state.stop_camera = False
    run_recognition()