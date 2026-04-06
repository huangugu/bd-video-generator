import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image

def detect_panels(img_array, min_area_ratio=0.05, max_area_ratio=0.4):
    """
    Détecte les cases de BD automatiquement en utilisant les contours OpenCV.
    """
    height, width, _ = img_array.shape
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    closed_edges = cv2.erode(dilated_edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    panels = []
    img_area = height * width
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        if min_area < area < max_area:
            aspect_ratio = w / float(h)
            if 0.3 < aspect_ratio < 3.0:
                margin = 5
                panels.append({
                    'x': max(0, x - margin),
                    'y': max(0, y - margin),
                    'w': min(width - x, w + 2 * margin),
                    'h': min(height - y, h + 2 * margin)
                })
    
    panels.sort(key=lambda p: p['y'])
    
    sorted_panels = []
    current_row = []
    if panels:
        current_y = panels[0]['y']
        row_threshold = height * 0.1
        
        for panel in panels:
            if abs(panel['y'] - current_y) <= row_threshold:
                current_row.append(panel)
            else:
                current_row.sort(key=lambda p: p['x'])
                sorted_panels.extend(current_row)
                current_row = [panel]
                current_y = panel['y']
        
        current_row.sort(key=lambda p: p['x'])
        sorted_panels.extend(current_row)
    
    return sorted_panels

def add_letterbox(frame, target_width, target_height):
    """
    Ajoute des bandes noires pour maintenir le ratio d'aspect sans déformation.
    """
    frame_h, frame_w = frame.shape[:2]
    
    # Calcul du ratio pour remplir sans déformer
    ratio = min(target_width / frame_w, target_height / frame_h)
    new_w = int(frame_w * ratio)
    new_h = int(frame_h * ratio)
    
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Créer un fond noir
    letterboxed = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Centrer l'image
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    
    letterboxed[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return letterboxed

def generate_comic_video_with_panels(img_array, panels, duration_per_panel, zoom_factor=1.2, background_mode="black"):
    """
    Génère la vidéo avec conservation des couleurs et ratios.
    background_mode: "black" ou "full_page"
    """
    height, width, _ = img_array.shape
    fps = 30
    total_frames = int(duration_per_panel * fps)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_file.name
    temp_file.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        st.error("Erreur lors de l'initialisation de l'encodeur vidéo.")
        return None
    
    progress_bar = st.progress(0)
    total_panels = len(panels)
    
    for i, panel in enumerate(panels):
        x, y, w, h = panel['x'], panel['y'], panel['w'], panel['h']
        
        if background_mode == "black":
            # Mode fond noir : zoom sur la case avec letterbox
            center_x = x + w // 2
            center_y = y + h // 2
            
            zoom_w = int(w / zoom_factor)
            zoom_h = int(h / zoom_factor)
            
            zoom_x_start = max(0, center_x - zoom_w // 2)
            zoom_y_start = max(0, center_y - zoom_h // 2)
            zoom_x_end = min(width, zoom_x_start + zoom_w)
            zoom_y_end = min(height, zoom_y_start + zoom_h)
            
            zoomed_region = img_array[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
            
            # Letterbox pour maintenir le ratio sans déformation
            final_frame = add_letterbox(zoomed_region, width, height)
            
        else:  # background_mode == "full_page"
            # Mode page complète : montre toute la BD avec cadre vert sur la case
            final_frame = img_array.copy()
            cv2.rectangle(final_frame, (x, y), (x + w, y + h), (0, 255, 0), 8)
        
        # Numéro de case (optionnel, discret)
        cv2.putText(final_frame, f"{i+1}/{total_panels}", (width - 100, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        for _ in range(total_frames):
            out.write(final_frame)
        
        progress_bar.progress((i + 1) / total_panels)
    
    out.release()
    return output_path

def draw_panel_preview(img_array, panels):
    """Dessine les rectangles des cases détectées pour l'aperçu."""
    preview = img_array.copy()
    for i, panel in enumerate(panels):
        x, y, w, h = panel['x'], panel['y'], panel['w'], panel['h']
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(preview, str(i+1), (x + 5, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return preview

# --- Interface Streamlit ---
st.set_page_config(page_title="Générateur Vidéo BD - IA", layout="wide")

st.title("🤖 Générateur de Vidéo BD avec Détection IA")
st.markdown("""
Détection automatique • **Couleurs originales préservées** • **Ratio d'aspect respecté** • Zoom intelligent
""")

uploaded_file = st.file_uploader("Choisissez une planche de BD (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    if len(img_array.shape) == 2: 
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4: 
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(image, caption="Image originale", use_column_width=True)
    
    with col2:
        st.subheader("⚙️ Paramètres")
        
        detection_mode = st.radio(
            "Mode de détection",
            ["Automatique (IA)", "Manuel (Grille)"],
            index=0
        )
        
        cols = st.number_input("Colonnes (mode manuel)", min_value=1, max_value=10, value=3)
        rows = st.number_input("Lignes (mode manuel)", min_value=1, max_value=10, value=4)
        
        duration = st.slider("Durée par case (sec)", min_value=0.5, max_value=5.0, value=1.5, step=0.5)
        zoom = st.slider("Facteur de zoom", min_value=1.0, max_value=3.0, value=1.2, step=0.1)
        
        background_mode = st.radio(
            "Style de fond",
            ["Fond noir (zoom plein écran)", "Page complète (case surlignée)"],
            index=0
        )
        
        min_area = st.slider("Sensibilité détection (min)", min_value=0.01, max_value=0.2, value=0.05, step=0.01)
        max_area = st.slider("Sensibilité détection (max)", min_value=0.2, max_value=0.8, value=0.4, step=0.05)

    st.divider()

    if detection_mode == "Automatique (IA)":
        panels = detect_panels(img_array, min_area, max_area)
        mode_info = f"🤖 **{len(panels)} cases détectées automatiquement**"
    else:
        h, w, _ = img_array.shape
        panel_w, panel_h = w // cols, h // rows
        panels = []
        for r in range(rows):
            for c in range(cols):
                panels.append({'x': c * panel_w, 'y': r * panel_h, 'w': panel_w, 'h': panel_h})
        mode_info = f"📐 **{len(panels)} cases (grille {cols}x{rows})**"
    
    st.info(mode_info)
    
    if panels:
        preview_img = draw_panel_preview(img_array, panels)
        st.image(preview_img, caption="Aperçu des cases détectées (numérotées dans l'ordre de lecture)", use_column_width=True)
        
        background_value = "black" if "Fond noir" in background_mode else "full_page"
        
        if st.button("🎬 Générer la vidéo", type="primary", use_container_width=True):
            with st.spinner('Génération de la vidéo en cours...'):
                try:
                    video_path = generate_comic_video_with_panels(
                        img_array, panels, duration, zoom, background_value
                    )
                    
                    if video_path:
                        st.success("✅ Vidéo générée avec succès !")
                        st.video(video_path)
                        
                        with open(video_path, "rb") as file:
                            st.download_button(
                                label="📥 Télécharger la vidéo (MP4)",
                                data=file,
                                file_name="bd_video_ia.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    st.code(str(e))
    else:
        st.warning("⚠️ Aucune case détectée. Essayez d'ajuster la sensibilité ou passez en mode Manuel.")

else:
    st.info("⬆️ Uploadez une image pour commencer")
    
    st.markdown("""
    ### ✨ Fonctionnalités
    - **Détection IA** : Contours et formes rectangulaires
    - **Couleurs originales** : Aucune altération des couleurs
    - **Ratio préservé** : Pas de déformation (letterbox si nécessaire)
    - **Deux styles** : Fond noir OU page complète avec surlignage
    - **Tri intelligent** : Ordre de lecture naturel
    """)
