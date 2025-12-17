import cv2
import mediapipe as mp
import streamlit as st
import time
import json
import os
import numpy as np
import tempfile
import random
import math

# Konfigurasi Halaman
st.set_page_config(
    page_title="GESTURE DETECTION",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CSS ENHANCED & MODERN
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }

    .stApp { 
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Header dengan gradient modern */
    .header-enhanced {
        text-align: center;
        padding: 3rem 2rem 4rem 2rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        border-radius: 0 0 32px 32px;
        margin-bottom: 3rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    }
    
    .header-enhanced::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(59, 130, 246, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 50%, rgba(147, 51, 234, 0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .header-enhanced h1 {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -2px;
        margin-bottom: 1rem;
        position: relative;
        text-transform: uppercase;
    }
    
    .header-enhanced p {
        font-size: 1.2rem;
        color: #cbd5e1;
        font-weight: 400;
        position: relative;
        letter-spacing: 0.5px;
    }

    /* Upload Area dengan hover effect */
    .upload-zone {
        background: white;
        border: 3px dashed #cbd5e1;
        border-radius: 20px;
        padding: 4rem 3rem;
        text-align: center;
        margin: 2rem auto;
        max-width: 900px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .upload-zone::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .upload-zone:hover {
        border-color: #3b82f6;
        background: linear-gradient(135deg, #f0f9ff 0%, #dbeafe 100%);
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(59, 130, 246, 0.15);
    }
    
    .upload-zone:hover::before {
        left: 100%;
    }
    
    .upload-zone h2 {
        font-size: 1.8rem;
        color: #0f172a;
        margin-bottom: 0.8rem;
        font-weight: 700;
    }
    
    .upload-zone p {
        color: #64748b;
        font-size: 1.1rem;
    }

    /* Video Preview Card */
    .video-preview {
        background: white;
        padding: 2rem;
        border-radius: 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin: 2rem 0;
    }

    /* Loading Overlay dengan animasi */
    .loading-overlay {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 41, 59, 0.98) 100%);
        border-radius: 24px;
        padding: 4rem 3rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .loading-overlay::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, transparent 70%);
        animation: rotate 8s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-title {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .loading-subtitle {
        font-size: 1.1rem;
        color: #cbd5e1;
        margin-bottom: 2rem;
        position: relative;
        z-index: 1;
    }
    
    .loading-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
        position: relative;
        z-index: 1;
    }
    
    .loading-stat {
        background: rgba(255,255,255,0.05);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    .loading-stat-value {
        font-size: 2rem;
        font-weight: 800;
        color: #3b82f6;
        margin-bottom: 0.5rem;
    }
    
    .loading-stat-label {
        font-size: 0.9rem;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Progress Bar Custom */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        background-size: 200% 100%;
        animation: gradient-shift 2s ease infinite;
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    /* Result Card dengan shadow dinamis */
    .result-card {
        background: white;
        padding: 3rem;
        border-radius: 24px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.06);
        border: 1px solid #f1f5f9;
        margin-bottom: 2rem;
        transition: transform 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.1);
    }
    
    /* Score Circle dengan efek 3D */
    .score-circle {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1.5rem;
        color: white;
        font-size: 3.5rem;
        font-weight: 900;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2),
                    inset 0 -5px 20px rgba(0, 0, 0, 0.2);
        position: relative;
    }
    
    .score-circle::after {
        content: '';
        position: absolute;
        top: 10%;
        left: 10%;
        width: 40%;
        height: 40%;
        background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .score-excellent { 
        background: linear-gradient(135deg, #059669, #047857, #065f46);
    }
    .score-good { 
        background: linear-gradient(135deg, #16a34a, #15803d, #166534);
    }
    .score-average { 
        background: linear-gradient(135deg, #f59e0b, #d97706, #b45309);
    }
    .score-poor { 
        background: linear-gradient(135deg, #dc2626, #b91c1c, #991b1b);
    }
    
    /* Metric Cards dengan gradients */
    .metric-card {
        background: white;
        padding: 2rem 1.5rem;
        border-radius: 16px;
        border: 2px solid #f1f5f9;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        border-color: #3b82f6;
        box-shadow: 0 12px 32px rgba(59, 130, 246, 0.15);
    }
    
    .metric-card:hover::before {
        transform: scaleX(1);
    }
    
    .metric-card h3 {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }
    
    .metric-card h2 {
        color: #0f172a;
        font-size: 2.5rem;
        font-weight: 900;
        margin: 0;
        background: linear-gradient(135deg, #0f172a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Analysis Summary dengan border keren */
    .analysis-panel {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2.5rem;
        border-radius: 20px;
        border-left: 6px solid #3b82f6;
        margin-top: 2rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        position: relative;
    }
    
    .analysis-panel::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 150px;
        height: 150px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.05) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .analysis-panel h2 {
        color: #0f172a;
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .insight-item {
        background: white;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        border-radius: 12px;
        border-left: 4px solid #e2e8f0;
        line-height: 1.8;
        color: #334155;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .insight-item:hover {
        border-left-color: #3b82f6;
        transform: translateX(4px);
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.1);
    }

    /* Info Section dengan cards */
    .info-section {
        background: white;
        padding: 3rem;
        border-radius: 24px;
        margin-top: 3rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.06);
        border: 1px solid #f1f5f9;
    }
    
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 2rem;
        margin-top: 2rem;
    }
    
    .info-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.08);
        border-color: #3b82f6;
    }
    
    .info-card h3 {
        color: #0f172a;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .info-card ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .info-card li {
        color: #64748b;
        padding: 0.6rem 0;
        padding-left: 1.5rem;
        position: relative;
        line-height: 1.6;
    }
    
    .info-card li::before {
        content: '→';
        position: absolute;
        left: 0;
        color: #3b82f6;
        font-weight: bold;
    }

    /* Button Styles */
    div.stButton > button {
        border-radius: 12px;
        font-weight: 700;
        padding: 1rem 3rem;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        width: 100%;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(59, 130, 246, 0.4);
    }
    
    .main .block-container {
        padding-top: 0;
        max-width: 1400px;
    }
    
    /* Animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .result-card, .analysis-panel {
        animation: slideInUp 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_mediapipe():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_face = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    return mp_drawing, mp_pose, mp_face, mp_hands

# ==========================================
# LOGIC ANALISIS REALISTIS (KETAT & MATEMATIS)
# ==========================================
class RealisticVideoAnalyzer:
    def __init__(self):
        self.frame_count = 0
        
        # Metrics Tracking
        self.total_hand_movement = 0
        self.max_hand_spread = 0
        self.eye_contact_frames = 0
        self.smile_frames = 0
        self.good_posture_frames = 0
        self.head_stability_variance = []
        
        # State sebelumnya untuk menghitung delta (gerakan)
        self.prev_hand_l = None
        self.prev_hand_r = None

    def calculate_distance(self, p1, p2):
        """Menghitung jarak Euclidean antar dua landmark"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def analyze_frame(self, pose_res, face_res, hand_res):
        self.frame_count += 1
        
        # --- 1. ANALISIS FACE (Fokus & Ekspresi) ---
        if face_res.multi_face_landmarks:
            landmarks = face_res.multi_face_landmarks[0].landmark
            
            # Eye Contact: Hidung relatif di tengah mata secara horizontal
            nose_tip = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            
            face_center_x = (left_eye.x + right_eye.x) / 2
            
            # Toleransi ketat: Hidung harus benar-benar di tengah
            if abs(nose_tip.x - face_center_x) < 0.05 and 0.35 < nose_tip.x < 0.65:
                self.eye_contact_frames += 1
                
            # Head Stability Data Collection
            self.head_stability_variance.append((nose_tip.x, nose_tip.y))
            
            # Smile Detection (Rasio Lebar vs Tinggi Mulut)
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            left_mouth = landmarks[61]
            right_mouth = landmarks[291]
            
            mouth_width = self.calculate_distance(left_mouth, right_mouth)
            mouth_height = self.calculate_distance(upper_lip, lower_lip)
            
            if mouth_height > 0 and (mouth_width / mouth_height) > 3.0:
                self.smile_frames += 1

        # --- 2. ANALISIS POSE (Postur) ---
        if pose_res.pose_landmarks:
            landmarks = pose_res.pose_landmarks.landmark
            
            l_shoulder = landmarks[11]
            r_shoulder = landmarks[12]
            
            # Cek Bahu Sejajar (Toleransi kemiringan rendah)
            shoulder_slope = abs(l_shoulder.y - r_shoulder.y)
            if shoulder_slope < 0.04: 
                self.good_posture_frames += 1

            # Fallback Tracking Gerakan Tangan via Pose
            l_wrist = landmarks[15]
            r_wrist = landmarks[16]
            
            if self.prev_hand_l:
                move_l = self.calculate_distance(l_wrist, self.prev_hand_l)
                move_r = self.calculate_distance(r_wrist, self.prev_hand_r)
                current_hand_movement = (move_l + move_r) * 50 
                self.total_hand_movement += current_hand_movement
            
            self.prev_hand_l = l_wrist
            self.prev_hand_r = r_wrist

        # --- 3. ANALISIS TANGAN DETIL (Sebaran) ---
        if hand_res.multi_hand_landmarks:
            for hand_lm in hand_res.multi_hand_landmarks:
                wrist = hand_lm.landmark[0]
                dist_from_center = abs(wrist.x - 0.5)
                if dist_from_center > self.max_hand_spread:
                    self.max_hand_spread = dist_from_center

    def calculate_final_scores(self, duration_seconds):
        if self.frame_count == 0: return self._get_default_scores()

        # --- NORMALISASI DATA ---
        eye_contact_ratio = self.eye_contact_frames / self.frame_count
        posture_ratio = self.good_posture_frames / self.frame_count
        smile_ratio = self.smile_frames / self.frame_count
        
        # Movement Metrics
        avg_movement = self.total_hand_movement / max(1, duration_seconds)
        
        # Head Stability
        stability_penalty = 0
        if len(self.head_stability_variance) > 1:
            head_x = [p[0] for p in self.head_stability_variance]
            head_y = [p[1] for p in self.head_stability_variance]
            variance = np.std(head_x) + np.std(head_y)
            if variance > 0.05: stability_penalty = min(25, variance * 400)
        
        # --- SISTEM PENILAIAN KETAT ---
        
        # 1. ABILITY
        base_ability = 65
        ability_bonus = (eye_contact_ratio * 20) + (posture_ratio * 15)
        ability_penalty = 0
        if eye_contact_ratio < 0.5: ability_penalty += 20
        if posture_ratio < 0.6: ability_penalty += 10
        score_ability = base_ability + ability_bonus - ability_penalty
        
        # 2. INTELLIGENCE
        base_intelligence = 70
        move_score = 0
        if 8 <= avg_movement <= 20: move_score = 15
        elif avg_movement < 5: move_score = -15
        elif avg_movement > 25: move_score = -20
        score_intelligence = base_intelligence + move_score - stability_penalty

        # 3. PERSONALITY
        base_personality = 60
        smile_bonus = smile_ratio * 25
        openness_bonus = min(15, self.max_hand_spread * 50)
        stiffness_penalty = 0
        if avg_movement < 4: stiffness_penalty = 20
        score_personality = base_personality + smile_bonus + openness_bonus - stiffness_penalty

        # 4. ATTITUDE
        base_attitude = 68
        posture_bonus = posture_ratio * 20
        attitude_penalty = stability_penalty
        if posture_ratio < 0.4: attitude_penalty += 25
        score_attitude = base_attitude + posture_bonus - attitude_penalty

        # 5. EMOTIONAL INTELLIGENCE
        base_emotional = 62
        connection_bonus = (eye_contact_ratio * 15) + (smile_ratio * 15)
        disconnect_penalty = 0
        if eye_contact_ratio < 0.3 and smile_ratio < 0.1:
            disconnect_penalty = 25
        score_emotional = base_emotional + connection_bonus - disconnect_penalty

        def clamp(val): return min(95, max(35, val))

        scores = {
            'ability': clamp(score_ability),
            'intelligence': clamp(score_intelligence),
            'personality': clamp(score_personality),
            'attitude': clamp(score_attitude),
            'emotional_intelligence': clamp(score_emotional)
        }
        scores['overall'] = sum(scores.values()) / 5
        return scores

    def _get_default_scores(self):
        return {k: 50 for k in ['ability', 'intelligence', 'personality', 'attitude', 'emotional_intelligence', 'overall']}

def get_score_class(score):
    if score >= 85: return "score-excellent"
    elif score >= 75: return "score-good"
    elif score >= 60: return "score-average"
    else: return "score-poor"

def get_score_label(score):
    if score >= 85: return "Sangat Baik (Luar Biasa)"
    elif score >= 75: return "Baik (Profesional)"
    elif score >= 60: return "Cukup (Standar)"
    else: return "Perlu Banyak Peningkatan"

def analyze_video_realistic(video_file, progress_bar, status_text, loading_container):
    analyzer = RealisticVideoAnalyzer()
    mp_drawing, mp_pose, mp_face, mp_hands = initialize_mediapipe()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        temp_path = tmp_file.name
    
    try:
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30
        
        duration = total_frames / fps
        
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        face = mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        frame_idx = 0
        skip_frames = 4
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1
            if frame_idx % skip_frames != 0: continue
            
            progress = frame_idx / total_frames
            progress_bar.progress(min(progress, 1.0))
            
            # Update loading stats
            processed_frames = frame_idx // skip_frames
            status_text.text(f"Frame {frame_idx}/{total_frames}")
            
            # Update stats in loading overlay
            loading_container.markdown(f"""
            <div class="loading-stats">
                <div class="loading-stat">
                    <div class="loading-stat-value">{int(progress * 100)}%</div>
                    <div class="loading-stat-label">Progress</div>
                </div>
                <div class="loading-stat">
                    <div class="loading-stat-value">{frame_idx}</div>
                    <div class="loading-stat-label">Frames Processed</div>
                </div>
                <div class="loading-stat">
                    <div class="loading-stat-value">{duration:.1f}s</div>
                    <div class="loading-stat-label">Video Duration</div>
                </div>
                <div class="loading-stat">
                    <div class="loading-stat-value">{fps}</div>
                    <div class="loading-stat-label">FPS</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            res_pose = pose.process(frame_rgb)
            res_face = face.process(frame_rgb)
            res_hand = hands.process(frame_rgb)
            
            analyzer.analyze_frame(res_pose, res_face, res_hand)
        
        video_duration = total_frames / fps
        final_scores = analyzer.calculate_final_scores(video_duration)
        
        face_ratio = analyzer.eye_contact_frames / max(1, analyzer.frame_count)
        insights = generate_analysis_insights(final_scores, analyzer, video_duration, face_ratio)
        
        results = {
            'scores': final_scores,
            'video_info': {
                'duration': video_duration,
                'total_frames': total_frames,
                'fps': fps,
                'face_visibility': face_ratio
            },
            'insights': insights
        }
        return results
        
    finally:
        cap.release()
        try:
            os.unlink(temp_path)
        except:
            pass

def generate_analysis_insights(scores, analyzer, duration, face_ratio):
    insights = []
    overall = scores['overall']
    
    if overall >= 85:
        insights.append("Performa luar biasa! Konsistensi gestur dan fokus sangat profesional.")
    elif overall >= 75:
        insights.append("Penampilan yang solid dan profesional. Pertahankan konsistensi ini.")
    elif overall >= 60:
        insights.append("Sudah memenuhi standar dasar, namun ada beberapa kebiasaan yang perlu diperbaiki.")
    else:
        insights.append("Perlu perbaikan signifikan pada dasar-dasar komunikasi non-verbal (kontak mata/postur).")
    
    avg_move = analyzer.total_hand_movement / max(1, duration)
    
    if face_ratio < 0.5:
        insights.append("Kontak mata sangat kurang. Anda terlalu sering melihat ke arah lain.")
    elif face_ratio > 0.85:
        insights.append("Kontak mata sangat baik dan fokus terjaga.")

    if avg_move > 25:
        insights.append("Terdeteksi gerakan tubuh yang berlebihan/gelisah. Cobalah lebih tenang.")
    elif avg_move < 5:
        insights.append("Tubuh terlalu kaku. Cobalah gunakan gestur tangan alami untuk menekankan poin.")
        
    smile_ratio = analyzer.smile_frames / max(1, analyzer.frame_count)
    if smile_ratio < 0.15:
        insights.append("Ekspresi cenderung datar. Jangan lupa tersenyum sesekali untuk kesan hangat.")

    metric_names = {
        'ability': 'Ability',
        'intelligence': 'Intelligence',
        'personality': 'Personality',
        'attitude': 'Attitude',
        'emotional_intelligence': 'Emotional Intelligence'
    }
    
    scores_only = {k: v for k, v in scores.items() if k != 'overall'}
    lowest_score = min(scores_only.items(), key=lambda x: x[1])
    
    if lowest_score[1] < 70:
         insights.append(f"Fokus perbaikan utama pada: {metric_names[lowest_score[0]]} (Skor: {lowest_score[1]:.0f}%).")
    
    return insights

def main():
    # Header
    st.markdown("""
    <div class="header-enhanced">
        <h1>Gesture Detection</h1>
        <p>Analisis Gestur Profesional dengan AI Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    main_container = st.container()
    
    with main_container:
        st.markdown("""
        <div class="upload-zone">
            <h2>Upload Video Presentasi Anda</h2>
            <p>Format: MP4, AVI, MOV • Durasi optimal: 1-5 menit • Pastikan wajah terlihat jelas</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Pilih file video",
            type=['mp4', 'avi', 'mov'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            st.markdown('<div class="video-preview">', unsafe_allow_html=True)
            st.video(uploaded_file)
            st.markdown('</div>', unsafe_allow_html=True)
            
            col_spacer1, col_btn, col_spacer2 = st.columns([1, 2, 1])
            with col_btn:
                if st.button("Mulai Analisis AI", type="primary", use_container_width=True):
                    # Loading Overlay
                    loading_placeholder = st.empty()
                    loading_placeholder.markdown("""
                    <div class="loading-overlay">
                        <div class="loading-title">Menganalisis Video Anda</div>
                        <div class="loading-subtitle">AI sedang memproses gestur, ekspresi wajah, dan postur tubuh...</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    stats_container = st.empty()
                    
                    try:
                        results = analyze_video_realistic(uploaded_file, progress_bar, status_text, stats_container)
                        
                        # Clear loading
                        loading_placeholder.empty()
                        progress_bar.empty()
                        status_text.empty()
                        stats_container.empty()
                        
                        # Results Display
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        
                        overall_score = results['scores']['overall']
                        score_class = get_score_class(overall_score)
                        score_label = get_score_label(overall_score)
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.markdown(f"""
                            <div class="{score_class} score-circle">
                                {overall_score:.0f}
                            </div>
                            <h2 style="text-align: center; color: #0f172a; margin: 0; font-size: 2.2rem; font-weight: 800;">Skor Total</h2>
                            <p style="text-align: center; color: #64748b; font-size: 1.3rem; margin-top: 0.8rem; font-weight: 600;">{score_label}</p>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("---")
                        st.markdown('<h3 style="color: #0f172a; margin-bottom: 2rem; font-size: 1.8rem; font-weight: 800; text-align: center;">Rincian Penilaian Komprehensif</h3>', unsafe_allow_html=True)
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        metrics = [
                            ('Ability', 'ability'),
                            ('Intelligence', 'intelligence'),
                            ('Personality', 'personality'),
                            ('Attitude', 'attitude'),
                            ('Emotional Intelligence', 'emotional_intelligence')
                        ]
                        
                        for i, (label, key) in enumerate(metrics):
                            score = results['scores'][key]
                            border_color = "#e2e8f0"
                            if score < 60: border_color = "#fca5a5"
                            elif score >= 80: border_color = "#86efac"

                            with [col1, col2, col3, col4, col5][i]:
                                st.markdown(f"""
                                <div class="metric-card" style="border-color: {border_color};">
                                    <h3>{label}</h3>
                                    <h2>{score:.0f}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Analysis Panel
                        st.markdown(f"""
                        <div class="analysis-panel">
                            <h2>Evaluasi & Rekomendasi Perbaikan</h2>
                            <div style="margin-top: 1.5rem;">
                                {"".join([f'<div class="insight-item">{insight}</div>' for insight in results['insights']])}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Video Info
                        st.markdown("<br>", unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        
                        info_items = [
                            ("Durasi Video", f"{results['video_info']['duration']:.1f} detik"),
                            ("Total Frame", f"{results['video_info']['total_frames']:,}"),
                            ("Frame Rate", f"{results['video_info']['fps']} FPS"),
                            ("Visibilitas Wajah", f"{results['video_info']['face_visibility']*100:.1f}%")
                        ]
                        
                        for i, (label, value) in enumerate(info_items):
                            with [col1, col2, col3, col4][i]:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{label}</h3>
                                    <h2 style="font-size: 1.5rem;">{value}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        loading_placeholder.empty()
                        progress_bar.empty()
                        status_text.empty()
                        stats_container.empty()
                        st.error(f"Terjadi kesalahan saat analisis: {str(e)}")

if __name__ == "__main__":
    main()