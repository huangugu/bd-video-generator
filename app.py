import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image

def generate_comic_video(img_array, cols, rows, duration_per_panel):
    height, width, _ = img_array.shape
    panel_w = width // cols
    panel_h = height // rows
    
    fps = 30
    total_frames = int(duration_per_panel * fps)
    
    # Fichier temporaire pour la sortie
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_file.name
    temp_file.close()
    
    # Utilisation de 'mp4v' qui est souvent plus compatible sur les serveurs Linux
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        st.error("Erreur lors de l'initialisation de l'encodeur vidéo.")
        return None
    
    progress_bar = st.progress(0)
    total_panels = cols * rows
    current_panel = 0

    # Parcours : De bas en haut (lignes), de gauche à droite (colonnes)
    for r in range(rows - 1, -1, -1):
        for c in range(cols):
            x_start = c * panel_w
            y_start = r * panel_h
            x_end = x_start + panel_w
            y_end = y_start + panel_h
            
            # Effet de mise en avant (cadre vert)
            overlay = img_array.copy()
            cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (0, 255, 0), 10)
            
            # Écriture des frames
            for _ in range(total_frames):
                out.write(overlay)
            
            current_panel += 1
            progress_bar.progress(current_panel / total_panels)

    out.release()
    return output_path

# --- Interface Streamlit ---
st.set_page_config(page_title="Générateur Vidéo BD", layout="centered")

st.title("🎬 Générateur de Vidéo Bande Dessinée")
st.markdown("Transformez votre planche de BD en vidéo animée case par case.")

uploaded_file = st.file_uploader("Choisissez une image (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Aperçu de la planche", use_column_width=True)
    
    # Conversion PIL -> OpenCV (BGR)
    img_array = np.array(image)
    
    # Gestion des canaux (Noir & Blanc ou RGBA)
    if len(img_array.shape) == 2: 
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4: 
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        cols = st.number_input("Colonnes", min_value=1, max_value=10, value=3)
    with col2:
        rows = st.number_input("Lignes", min_value=1, max_value=10, value=4)
    with col3:
        duration = st.number_input("Durée/case (sec)", min_value=0.5, max_value=10.0, value=1.5, step=0.5)

    if st.button(" Générer la vidéo", type="primary"):
        with st.spinner('Traitement en cours...'):
            try:
                video_path = generate_comic_video(img_array, cols, rows, duration)
                
                if video_path:
                    st.success("Vidéo générée !")
                    st.video(video_path)
                    
                    with open(video_path, "rb") as file:
                        st.download_button(
                            label="📥 Télécharger la vidéo (MP4)",
                            data=file,
                            file_name="ma_planche_bd.mp4",
                            mime="video/mp4"
                        )
            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")
                st.code(str(e))
else:
    st.info("⬆️ Veuillez uploader une image pour commencer.")
