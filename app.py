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

def zoom_interpolate(start_scale, end_scale, progress):
    """Interpolation fluide pour le zoom avec easing."""
    # Ease in-out pour un mouvement plus naturel
    if progress < 0.5:
        ease = 2 * progress * progress
    else:
        ease = 1 - pow(-2 * progress + 2, 2) / 2
    return start_scale + (end_scale - start_scale) * ease

def create_zoom_frame(img_array, center_x, center_y, scale, target_width, target_height):
    """
    Crée une frame avec zoom centré sur un point.
    scale = 1.0 → vue complète, scale > 1.0 → zoomé
    """
    height, width, _ = img_array.shape
    
    # Calcul de la zone à extraire selon le zoom
    view_w = int(width / scale)
    view_h = int(height / scale)
    
    # Centrer sur le point cible
    x_start = max(0, int(center_x - view_w / 2))
    y_start = max(0, int(center_y - view_h / 2))
    x_end = min(width, x_start + view_w)
    y_end = min(height, y_start + view_h)
    
    # Ajuster si on atteint les bords
    if x_end - x_start < view_w:
        if x_start == 0:
            x_end = min(width, view_w)
        else:
            x_start = max(0, x_end - view_w)
    
    if y_end - y_start < view_h:
        if y_start == 0:
            y_end = min(height, view_h)
        else:
            y_start = max(0, y_end - view_h)
    
    cropped = img_array[y_start:y_end, x_start:x_end]
    
    # Redimensionner pour remplir l'écran (sans déformation car on garde le ratio de l'image originale)
    resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    return resized

def generate_comic_video_cinematic(img_array, panels, duration_per_panel, max_zoom=2.0):
    """
    Génère une vidéo avec effet de caméra cinématique :
    - Vue complète de la page
    - Zoom fluide vers la case
    - Pause sur la case
    - Dézoom vers vue complète
    - Transition vers la case suivante
    """
    height, width, _ = img_array.shape
    fps = 30
    total_panels = len(panels)
    
    # Découpage du temps par case
    zoom_in_frames = int(duration_per_panel * fps * 0.3)  # 30% pour zoomer
    hold_frames = int(duration_per_panel * fps * 0.4)      # 40% en zoom
    zoom_out_frames = int(duration_per_panel * fps * 0.3)  # 30% pour dézoomer
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_file.name
    temp_file.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        st.error("Erreur lors de l'initialisation de l'encodeur vidéo.")
        return None
    
    progress_bar = st.progress(0)
    
    for i, panel in enumerate(panels):
        x, y, w, h = panel['x'], panel['y'], panel['w'], panel['h']
        
        # Centre de la case
        center_x = x + w // 2
        center_y = y + h // 2
        
        # 1. ZOOM IN (de la vue complète vers la case)
        for frame_idx in range(zoom_in_frames):
            progress = frame_idx / zoom_in_frames
            scale = zoom_interpolate(1.0, max_zoom, progress)
            frame = create_zoom_frame(img_array, center_x, center_y, scale, width, height)
            out.write(frame)
        
        # 2. HOLD (maintenir le zoom sur la case)
        for frame_idx in range(hold_frames):
            frame = create_zoom_frame(img_array, center_x, center_y, max_zoom, width, height)
            out.write(frame)
        
        # 3. ZOOM OUT (retour à la vue complète)
        for frame_idx in range(zoom_out_frames):
            progress = frame_idx / zoom_out_frames
            scale = zoom_interpolate(max_zoom, 1.0, progress)
            frame = create_zoom_frame(img_array, center_x, center_y, scale, width, height)
            out.write(frame)
        
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
st.set_page_config(page_title="Générateur Vidéo BD - Cinématique", layout="wide")

st.title("🎬 Générateur de Vidéo BD - Effet Cinématique")
st.markdown("""
**Détection IA** • **Zoom fluide avec la caméra** • **Pas de surlignage** • **Couleurs et ratios préservés**
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
        
        duration = st.slider("Durée par case (sec)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
        max_zoom = st.slider("Niveau de zoom max", min_value=1.5, max_value=5.0, value=2.5, step=0.5)

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
        
        if st.button("🎬 Générer la vidéo cinématique", type="primary", use_container_width=True):
            with st.spinner('Génération de la vidéo en cours... Cela peut prendre quelques minutes.'):
                try:
                    video_path = generate_comic_video_cinematic(img_array, panels, duration, max_zoom)
                    
                    if video_path:
                        st.success("✅ Vidéo générée avec succès !")
                        st.video(video_path)
                        
                        with open(video_path, "rb") as file:
                            st.download_button(
                                label="📥 Télécharger la vidéo (MP4)",
                                data=file,
                                file_name="bd_video_cinematique.mp4",
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
    - **Effet cinématique** : La caméra zoome et dézoome sur chaque case
    - **Mouvement fluide** : Transitions douces avec easing
    - **Aucun surlignage** : Juste la caméra qui se déplace
    - **Couleurs originales** : Aucune altération
    - **Ratio préservé** : Pas de déformation
    - **Détection IA** : Ou mode grille manuel
    """)
