import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image

def generate_comic_video_zoom(img_array, cols, rows, duration_per_panel, zoom_factor=1.5):
    """
    Génère une vidéo avec zoom sur chaque case, parcours de gauche à droite, puis haut en bas.
    """
    height, width, _ = img_array.shape
    panel_w = width // cols
    panel_h = height // rows
    
    fps = 30
    total_frames = int(duration_per_panel * fps)
    
    # Fichier temporaire
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_file.name
    temp_file.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        st.error("Erreur lors de l'initialisation de l'encodeur vidéo.")
        return None
    
    progress_bar = st.progress(0)
    total_panels = cols * rows
    current_panel = 0

    # Parcours : de HAUT en BAS, de GAUCHE à DROITE
    for r in range(rows):  # De haut en bas
        for c in range(cols):  # De gauche à droite
            x_start = c * panel_w
            y_start = r * panel_h
            x_end = x_start + panel_w
            y_end = y_start + panel_h
            
            # Calcul du centre de la case
            center_x = (x_start + x_end) // 2
            center_y = (y_start + y_end) // 2
            
            # Taille de la zone à afficher (zoom)
            zoom_w = int(panel_w / zoom_factor)
            zoom_h = int(panel_h / zoom_factor)
            
            # Coordonnées du rectangle zoomé (centré sur la case)
            zoom_x_start = max(0, center_x - zoom_w // 2)
            zoom_y_start = max(0, center_y - zoom_h // 2)
            zoom_x_end = min(width, zoom_x_start + zoom_w)
            zoom_y_end = min(height, zoom_y_start + zoom_h)
            
            # Extraire la zone zoomée
            zoomed_region = img_array[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
            
            # Redimensionner pour remplir l'écran (effet plein écran)
            resized_frame = cv2.resize(zoomed_region, (width, height), interpolation=cv2.INTER_AREA)
            
            # Ajouter un texte indiquant la case actuelle (optionnel)
            cv2.putText(resized_frame, f"Case {c+1}-{r+1}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Écrire les frames pour la durée souhaitée
            for _ in range(total_frames):
                out.write(resized_frame)
            
            current_panel += 1
            progress_bar.progress(current_panel / total_panels)

    out.release()
    return output_path

# --- Interface Streamlit ---
st.set_page_config(page_title="Générateur Vidéo BD - Zoom", layout="centered")

st.title("🎬 Générateur de Vidéo Bande Dessinée avec Zoom")
st.markdown("Créez une vidéo qui zoome sur chaque case, de gauche à droite et de haut en bas.")

uploaded_file = st.file_uploader("Choisissez une image (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Aperçu de la planche", use_column_width=True)
    
    img_array = np.array(image)
    if len(img_array.shape) == 2: 
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4: 
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cols = st.number_input("Colonnes", min_value=1, max_value=10, value=3)
    with col2:
        rows = st.number_input("Lignes", min_value=1, max_value=10, value=4)
    with col3:
        duration = st.number_input("Durée/case (sec)", min_value=0.5, max_value=10.0, value=1.5, step=0.5)
    with col4:
        zoom = st.slider("Facteur de zoom", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

    if st.button("  Générer la vidéo avec zoom", type="primary"):
        with st.spinner('Création de la vidéo en cours...'):
            try:
                video_path = generate_comic_video_zoom(img_array, cols, rows, duration, zoom)
                
                if video_path:
                    st.success("Vidéo générée avec succès !")
                    st.video(video_path)
                    
                    with open(video_path, "rb") as file:
                        st.download_button(
                            label="📥 Télécharger la vidéo (MP4)",
                            data=file,
                            file_name="ma_planche_bd_zoom.mp4",
                            mime="video/mp4"
                        )
            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")
                st.code(str(e))
else:
    st.info("⬆️ Veuillez uploader une image pour commencer.")
