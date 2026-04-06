import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image

def detect_panels(img_array, min_area_ratio=0.05, max_area_ratio=0.4):
    """Détection des cases de BD."""
    try:
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
        
        # Tri intelligent
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
    except Exception as e:
        st.error(f"Erreur détection: {e}")
        return []

def create_zoom_frame(img_array, center_x, center_y, scale, target_width, target_height):
    """Crée une frame zoomée de manière sécurisée."""
    try:
        height, width, _ = img_array.shape
        
        # Éviter les divisions par zéro et les scales trop extrêmes
        scale = max(1.0, min(scale, 10.0))
        
        view_w = int(width / scale)
        view_h = int(height / scale)
        
        # Coordonnées centrées
        x_start = int(center_x - view_w / 2)
        y_start = int(center_y - view_h / 2)
        x_end = x_start + view_w
        y_end = y_start + view_h
        
        # Ajustement aux bords de l'image
        if x_start < 0:
            x_start = 0
            x_end = min(width, view_w)
        if y_start < 0:
            y_start = 0
            y_end = min(height, view_h)
        if x_end > width:
            x_end = width
            x_start = max(0, width - view_w)
        if y_end > height:
            y_end = height
            y_start = max(0, height - view_h)
        
        # Vérification finale
        x_start = max(0, min(x_start, width - 1))
        y_start = max(0, min(y_start, height - 1))
        x_end = max(x_start + 1, min(x_end, width))
        y_end = max(y_start + 1, min(y_end, height))
        
        cropped = img_array[y_start:y_end, x_start:x_end]
        
        # Redimensionnement
        if cropped.size > 0:
            resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)
            return resized
        else:
            # Fallback: image noire en cas d'erreur
            return np.zeros((target_height, target_width, 3), dtype=np.uint8)
    except Exception as e:
        st.error(f"Erreur zoom: {e}")
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

def generate_comic_video_cinematic(img_array, panels, duration_per_panel, max_zoom=2.0):
    """Génération de vidéo avec effets cinématiques."""
    try:
        height, width, _ = img_array.shape
        fps = 30
        total_panels = len(panels)
        
        if total_panels == 0:
            st.error("Aucune case à traiter!")
            return None
        
        # Calcul des frames
        zoom_in_frames = max(1, int(duration_per_panel * fps * 0.3))
        hold_frames = max(1, int(duration_per_panel * fps * 0.4))
        zoom_out_frames = max(1, int(duration_per_panel * fps * 0.3))
        
        total_frames = total_panels * (zoom_in_frames + hold_frames + zoom_out_frames)
        
        # Fichier temporaire
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=tempfile.gettempdir())
        output_path = temp_file.name
        temp_file.close()
        
        # Encodeur vidéo
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            st.error("Erreur: Impossible d'ouvrir l'encodeur vidéo.")
            return None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        frame_count = 0
        
        for i, panel in enumerate(panels):
            try:
                x, y, w, h = panel['x'], panel['y'], panel['w'], panel['h']
                
                # Validation des coordonnées
                if w <= 0 or h <= 0:
                    st.warning(f"Case {i+1} invalide, ignorée")
                    continue
                
                center_x = x + w // 2
                center_y = y + h // 2
                
                status_text.text(f"Traitement case {i+1}/{total_panels}")
                
                # ZOOM IN
                for frame_idx in range(zoom_in_frames):
                    progress = frame_idx / max(1, zoom_in_frames - 1) if zoom_in_frames > 1 else 1
                    # Ease in-out
                    if progress < 0.5:
                        ease = 2 * progress * progress
                    else:
                        ease = 1 - pow(-2 * progress + 2, 2) / 2
                    scale = 1.0 + (max_zoom - 1.0) * ease
                    
                    frame = create_zoom_frame(img_array, center_x, center_y, scale, width, height)
                    out.write(frame)
                    frame_count += 1
                
                # HOLD
                for _ in range(hold_frames):
                    frame = create_zoom_frame(img_array, center_x, center_y, max_zoom, width, height)
                    out.write(frame)
                    frame_count += 1
                
                # ZOOM OUT
                for frame_idx in range(zoom_out_frames):
                    progress = frame_idx / max(1, zoom_out_frames - 1) if zoom_out_frames > 1 else 0
                    if progress < 0.5:
                        ease = 2 * progress * progress
                    else:
                        ease = 1 - pow(-2 * progress + 2, 2) / 2
                    scale = max_zoom - (max_zoom - 1.0) * ease
                    
                    frame = create_zoom_frame(img_array, center_x, center_y, scale, width, height)
                    out.write(frame)
                    frame_count += 1
                
                progress_bar.progress((i + 1) / total_panels)
                
            except Exception as e:
                st.error(f"Erreur case {i+1}: {e}")
                continue
        
        out.release()
        status_text.empty()
        
        # Vérification du fichier
        if os.path.getsize(output_path) == 0:
            st.error("Fichier vidéo vide!")
            return None
        
        return output_path
        
    except Exception as e:
        st.error(f"Erreur génération vidéo: {e}")
        return None

def draw_panel_preview(img_array, panels):
    """Aperçu des cases détectées."""
    try:
        preview = img_array.copy()
        for i, panel in enumerate(panels):
            x, y, w, h = panel['x'], panel['y'], panel['w'], panel['h']
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(preview, str(i+1), (x + 5, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return preview
    except:
        return img_array

import os

# --- Interface Streamlit ---
st.set_page_config(page_title="Générateur Vidéo BD - Cinématique", layout="wide")

st.title("🎬 Générateur de Vidéo BD - Effet Cinématique")
st.markdown("""
**Détection IA** • **Zoom fluide** • **Pas de surlignage** • **Stable et optimisé**
""")

uploaded_file = st.file_uploader("Choisissez une planche de BD (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        if len(img_array.shape) == 2: 
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4: 
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        
        st.success(f"Image chargée: {img_array.shape[1]}x{img_array.shape[0]} pixels")

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
            st.image(preview_img, caption="Aperçu des cases détectées", use_column_width=True)
            
            if st.button("🎬 Générer la vidéo cinématique", type="primary", use_container_width=True):
                with st.spinner('Génération en cours... Veuillez patienter.'):
                    try:
                        video_path = generate_comic_video_cinematic(img_array, panels, duration, max_zoom)
                        
                        if video_path and os.path.exists(video_path):
                            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                            st.success(f"✅ Vidéo générée! ({file_size:.1f} MB)")
                            st.video(video_path)
                            
                            with open(video_path, "rb") as file:
                                st.download_button(
                                    label="📥 Télécharger la vidéo (MP4)",
                                    data=file,
                                    file_name="bd_video_cinematique.mp4",
                                    mime="video/mp4",
                                    use_container_width=True
                                )
                        else:
                            st.error("Échec de la génération de vidéo.")
                    except Exception as e:
                        st.error(f"Erreur: {e}")
                        st.exception(e)
        else:
            st.warning("⚠️ Aucune case détectée. Ajustez la sensibilité ou utilisez le mode Manuel.")

    except Exception as e:
        st.error(f"Erreur lors du chargement de l'image: {e}")
        st.exception(e)

else:
    st.info("⬆️ Uploadez une image pour commencer")
