"""
TennisAI — Vision Pipeline v3 (Full-Session Analysis)
──────────────────────────────────────────────────────
Basado en v2. Cambios en v3:

  Fase A (roadmap fixes):
    - Issue #3: coverage_ratio expuesto en mediapipe_result_db
                (frames_analyzed / expected_frames a 30fps)
    - Issue #5: processing_gaps_percent expuesto en compression_meta
                ((windows_found - clips_processed) / windows_found)

  Fase B (pendiente): TrackNetV3 reemplaza ball tracker
  Fase C (pendiente): ViTPose-Huge reemplaza MediaPipe (requiere HF token)
  Fase D (pendiente): Stroke Phase Detector

Deploy:
  modal deploy vision_pipeline_v3.py

Secrets requeridos en Modal:
  - supabase-key  →  SUPABASE_URL, SUPABASE_SERVICE_KEY
"""

import modal
import json
import uuid
from pathlib import Path
from starlette.requests import Request

app = modal.App("tennis-vision-pipeline-v3")

# ── Imagen con OpenCV + YOLO + MediaPipe + Ball Tracker + FFmpeg ───
# v3: MediaPipe para pose estimation (ya no ViTPose — está gated en HF)
vision_image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands(
        "apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 ffmpeg"
    )
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "ultralytics==8.2.0",
        "mediapipe==0.10.14",     # Pose estimation
        "opencv-python-headless",
        "numpy==1.26.4",
        "huggingface_hub==0.34.0",  # Para descargar ball tracker model
        "httpx==0.27.0",
        "fastapi[standard]",
    )
)


# ─── HELPERS SUPABASE ─────────────────────────────────────────
def supabase_patch(url: str, key: str, table: str, record_id: str, data: dict) -> bool:
    import httpx
    resp = httpx.patch(
        f"{url}/rest/v1/{table}?id=eq.{record_id}",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json=data,
        timeout=30,
    )
    return resp.status_code in (200, 204)


def supabase_post(url: str, key: str, table: str, data: dict) -> dict | None:
    import httpx
    resp = httpx.post(
        f"{url}/rest/v1/{table}",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
        json=data,
        timeout=30,
    )
    if resp.status_code in (200, 201):
        saved = resp.json()
        return saved[0] if saved else None
    return None


# ─── FASE 0-A: TRIAJE POR CPU (OpenCV) ───────────────────────
def extract_action_windows(
    video_path: str,
    fps_sample: int   = 2,
    threshold: float  = 1.5,
    pad_before: float = 3.0,
    pad_after:  float = 5.0,
) -> list[tuple[float, float]]:
    """
    Escanea el video a baja resolución/fps buscando picos de movimiento.
    Retorna lista de tuplas (start_sec, end_sec).

    fps_sample  : cuántos frames por segundo analizar (2 es suficiente)
    threshold   : sensibilidad — bajar si se pierden golpes, subir si hay
                  demasiados falsos positivos por viento o público.
    pad_before  : segundos antes del pico de movimiento a incluir.
    pad_after   : segundos después del pico a incluir.
    """
    import cv2
    import numpy as np

    cap          = cv2.VideoCapture(video_path)
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    frame_jump   = max(1, int(fps_original / fps_sample))

    motion_peaks: list[float] = []
    prev_gray = None
    frame_idx = 0

    while True:
        # Saltar frames para leer solo a fps_sample
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Bajar resolución para CPU rápida
        small = cv2.resize(frame, (640, 360))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray  = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is not None:
            delta  = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            score  = float(np.sum(thresh)) / float(thresh.size)

            if score > threshold:
                timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                motion_peaks.append(timestamp_sec)

        prev_gray  = gray
        frame_idx += frame_jump

    cap.release()

    if not motion_peaks:
        # Fallback: si no se detecta nada, procesar el video completo
        cap2          = cv2.VideoCapture(video_path)
        total_frames  = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        fps2          = cap2.get(cv2.CAP_PROP_FPS)
        total_seconds = total_frames / fps2 if fps2 > 0 else 0
        cap2.release()
        print("⚠️ Triaje no detectó movimiento — procesando video completo como fallback")
        return [(0.0, total_seconds)]

    # Consolidar picos en ventanas con padding
    windows: list[list[float]] = []
    for t in sorted(motion_peaks):
        start = max(0.0, t - pad_before)
        end   = t + pad_after
        if not windows or start > windows[-1][1]:
            windows.append([start, end])
        else:
            windows[-1][1] = max(windows[-1][1], end)  # Extender si se solapan

    return [(w[0], w[1]) for w in windows]


# ─── FASE 1-GPU: INFERENCIA POR CLIP ─────────────────────────
@app.function(
    image=vision_image,
    timeout=600,
    memory=4096,
    gpu="T4",
     secrets=[modal.Secret.from_name("huggingface-secret")],
)
def process_single_clip(
    clip_bytes:      bytes,
    time_offset:     float,
    store_landmarks: bool = True,
    dominant_hand:   str  = "right",
) -> dict:
    """
    Corre MediaPipe + YOLO + BallTracker sobre un clip de ~8 segundos.
    time_offset: el start_sec del clip en la sesión original.
    Retorna impactos y frames con timestamp_global correcto.
    """
    import cv2
    import numpy as np
    import tempfile, os
    import torch
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download

    # ── Volcar bytes a disco temporal ─────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(clip_bytes)
        clip_path = f.name

    try:
        # ── Helpers compartidos ───────────────────────────────
        def calc_angle(a, b, c):
            a, b, c = np.array(a), np.array(b), np.array(c)
            ba, bc  = a - b, c - b
            cosine  = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return round(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))), 1)

        # ── YOLO (primero — detección de jugador) ────
        # YOLO corre a 5fps para stroke hints + bbox del jugador.
        # La bbox no es esencial para MediaPipe, pero se guarda para contexto.
        TARGET_FPS_YOLO = 5
        CONF_THRESHOLD  = 0.5

        yolo_model       = YOLO("yolov8n.pt")
        cap              = cv2.VideoCapture(clip_path)
        fps_y            = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_every_y   = max(1, int(round(fps_y / TARGET_FPS_YOLO)))
        yolo_frames: list[dict] = []
        position_history: list[dict] = []
        # bbox_by_frame: frame_count → (x1,y1,x2,y2) en píxeles — para contexto/logging
        bbox_by_frame: dict[int, tuple] = {}
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % sample_every_y == 0:
                results     = yolo_model(frame, classes=[0], verbose=False)
                main_player = None
                raw_bbox    = None
                for r in results:
                    players = []
                    for box in r.boxes:
                        conf = round(float(box.conf[0]), 2)
                        if conf < CONF_THRESHOLD:
                            continue
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        w, h = x2 - x1, y2 - y1
                        players.append({
                            "confidence":   conf,
                            "center_x":     round((x1 + x2) / 2, 1),
                            "center_y":     round((y1 + y2) / 2, 1),
                            "width":        round(w, 1),
                            "height":       round(h, 1),
                            "aspect_ratio": round(h / (w + 1e-6), 2),
                            "_bbox":        (x1, y1, x2, y2),
                        })
                    if players:
                        best = max(players, key=lambda p: p["confidence"])
                        raw_bbox    = best.pop("_bbox")
                        main_player = best

                if main_player and raw_bbox:
                    bbox_by_frame[frame_count] = raw_bbox
                    local_ts   = frame_count / fps_y
                    global_ts  = round(local_ts + time_offset, 3)
                    stroke_hint = None
                    position_history.append({
                        "cx": main_player["center_x"],
                        "cy": main_player["center_y"],
                        "aspect_ratio": main_player["aspect_ratio"],
                    })
                    if len(position_history) >= 3:
                        recent  = position_history[-3:]
                        dx      = abs(recent[-1]["cx"] - recent[0]["cx"])
                        dy      = abs(recent[-1]["cy"] - recent[0]["cy"])
                        avg_asp = sum(p["aspect_ratio"] for p in recent) / 3
                        if dy > 30 and avg_asp < 1.8:
                            stroke_hint = "posible_saque_o_smash"
                        elif dx > 40:
                            stroke_hint = "posible_forehand_o_backhand"
                        elif dx < 10 and dy < 10:
                            stroke_hint = "posicion_base"
                        else:
                            stroke_hint = "movimiento_general"

                    yolo_frames.append({
                        "frame":            frame_count,
                        "timestamp":        global_ts,
                        "timestamp_local":  round(local_ts, 3),
                        "player":           main_player,
                        "stroke_hint":      stroke_hint,
                    })
            frame_count += 1
        cap.release()

        # ── MediaPipe Pose (Heavy) ────────────────────────────────
        from mediapipe.python.solutions.pose import Pose as MPPose

        TARGET_FPS_MP = 30
        VIS_THRESHOLD = 0.6

        pose         = MPPose(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
        )

        cap          = cv2.VideoCapture(clip_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration     = total_frames / fps
        sample_every = max(1, int(round(fps / TARGET_FPS_MP)))

        mp_frames: list[dict] = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % sample_every == 0:
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if result.pose_landmarks:
                    lms = result.pose_landmarks.landmark

                    def pt(idx):
                        return [lms[idx].x, lms[idx].y, lms[idx].z]

                    key_idxs   = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
                    visibility = round(float(sum(lms[i].visibility for i in key_idxs) / len(key_idxs)), 2)

                    if visibility >= VIS_THRESHOLD:
                        local_ts  = frame_count / fps
                        global_ts = round(local_ts + time_offset, 3)

                        entry = {
                            "frame":           frame_count,
                            "timestamp":       global_ts,
                            "timestamp_local": round(local_ts, 3),
                            "angles": {
                                "right_elbow": calc_angle(pt(12), pt(14), pt(16)),
                                "left_elbow":  calc_angle(pt(11), pt(13), pt(15)),
                                "right_knee":  calc_angle(pt(24), pt(26), pt(28)),
                                "left_knee":   calc_angle(pt(23), pt(25), pt(27)),
                                "right_hip":   calc_angle(pt(12), pt(24), pt(26)),
                                "left_hip":    calc_angle(pt(11), pt(23), pt(25)),
                            },
                            "shoulder_alignment": round(
                                abs(lms[11].y - lms[12].y) * 100, 2
                            ),
                            "visibility": visibility,
                        }
                        if store_landmarks:
                            entry["landmarks_3d"] = [
                                {
                                    "x":          round(float(lms[i].x), 4),
                                    "y":          round(float(lms[i].y), 4),
                                    "z":          round(float(lms[i].z), 4),
                                    "visibility": round(float(lms[i].visibility), 3),
                                }
                                for i in range(33)
                            ]
                        mp_frames.append(entry)

            frame_count += 1

        cap.release()
        pose.close()

                # ── Ball Tracker (v3) ───────────────────────────────────
        # Cambio 1: 25fps en lugar de 10fps → diff_ms promedio ~20ms vs ~60ms
        # Cambio 3: umbral dinámico — conf>=0.3 siempre acepta;
        #           conf>=0.15 acepta solo si es consistente con trayectoria previa
        TARGET_FPS_BALL     = 25
        CONF_BALL_PRIMARY   = 0.3
        CONF_BALL_FALLBACK  = 0.15
        MAX_JUMP_PIXELS     = 250   # máximo desplazamiento esperado a 25fps (~200px/frame a tope)

        ball_model_path = hf_hub_download(
            repo_id="RJTPP/tennis-ball-detection", filename="best.pt", repo_type="model"
        )
        ball_model  = YOLO(ball_model_path)
        cap         = cv2.VideoCapture(clip_path)
        fps_b       = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_b    = max(1, int(round(fps_b / TARGET_FPS_BALL)))
        ball_frames: list[dict]   = []
        ball_positions: list[dict] = []   # solo detecciones aceptadas, para extrapolación
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % sample_b == 0:
                # Correr YOLO con umbral bajo para capturar candidatos de baja confianza
                results  = ball_model(frame, verbose=False, conf=CONF_BALL_FALLBACK)
                ball     = None
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf     = round(float(box.conf[0]), 2)
                        center_x = round((x1 + x2) / 2, 1)
                        center_y = round((y1 + y2) / 2, 1)
                        size     = round(((x2 - x1) + (y2 - y1)) / 2, 1)
                        if ball is None or conf > ball["confidence"]:
                            ball = {"confidence": conf, "center_x": center_x, "center_y": center_y, "size": size}

                # Cambio 3: filtro dinámico por confianza + consistencia de trayectoria
                if ball is not None:
                    accepted = False
                    if ball["confidence"] >= CONF_BALL_PRIMARY:
                        accepted = True   # confianza alta — siempre aceptar
                    elif ball["confidence"] >= CONF_BALL_FALLBACK and len(ball_positions) >= 2:
                        # Baja confianza — aceptar solo si está cerca de la posición extrapolada
                        p1 = ball_positions[-2]
                        p2 = ball_positions[-1]
                        expected_x = p2["center_x"] + (p2["center_x"] - p1["center_x"])
                        expected_y = p2["center_y"] + (p2["center_y"] - p1["center_y"])
                        dist = float(np.sqrt(
                            (ball["center_x"] - expected_x) ** 2 +
                            (ball["center_y"] - expected_y) ** 2
                        ))
                        accepted = dist <= MAX_JUMP_PIXELS
                    if not accepted:
                        ball = None

                local_ts  = frame_count / fps_b
                global_ts = round(local_ts + time_offset, 3)

                if ball is not None:
                    if ball_positions:
                        prev = ball_positions[-1]
                        dx   = ball["center_x"] - prev["center_x"]
                        dy   = ball["center_y"] - prev["center_y"]
                        ball["speed_pixels"] = round(float(np.sqrt(dx**2 + dy**2)), 1)
                    else:
                        ball["speed_pixels"] = None
                    ball_positions.append(ball)

                ball_frames.append({
                    "frame":            frame_count,
                    "timestamp":        global_ts,
                    "timestamp_local":  round(local_ts, 3),
                    "ball_detected":    ball is not None,
                    "ball":             ball,
                })
            frame_count += 1
        cap.release()

        # ── Detectar impactos en este clip ────────────────────
        cap_tmp = cv2.VideoCapture(clip_path)
        frame_h = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_tmp.release()
        clip_impacts = _detect_impacts_from_clip(ball_frames, mp_frames, yolo_frames, time_offset, dominant_hand, frame_h)

        print(
            f"  📎 Clip offset={time_offset:.1f}s | "
            f"mp={len(mp_frames)} | yolo={len(yolo_frames)} | "
            f"ball={len(ball_frames)} | impactos={len(clip_impacts)}"
        )

        return {
            "time_offset":    time_offset,
            "duration":       duration,
            "mp_frames":      mp_frames,
            "yolo_frames":    yolo_frames,
            "ball_frames":    ball_frames,
            "impact_frames":  clip_impacts,
        }

    finally:
        if os.path.exists(clip_path):
            os.unlink(clip_path)


def _classify_stroke_from_landmarks(
    landmarks_3d:  list[dict],
    dominant_hand: str = "right",
) -> str:
    """
    Distingue forehand / backhand / saque_o_smash usando la geometría
    de la pose en el momento exacto del impacto.

    Forehand/backhand: si la muñeca dominante está al mismo lado del
    centro de cadera que el hombro dominante → forehand, si no → backhand.

    Saque/smash: si la muñeca dominante está claramente por encima del
    hombro dominante (y menor en coords normalizadas donde 0 = arriba).
    """
    # Índices MediaPipe 33 (pose estimation):
    #  9=L_wrist  10=R_wrist  |  5=L_shoulder  6=R_shoulder  |  11=L_hip  12=R_hip
    wrist_idx    = 10 if dominant_hand == "right" else 9
    shoulder_idx =  6 if dominant_hand == "right" else 5

    wrist_x    = landmarks_3d[wrist_idx]["x"]
    wrist_y    = landmarks_3d[wrist_idx]["y"]
    shoulder_y = landmarks_3d[shoulder_idx]["y"]
    hip_center = (landmarks_3d[11]["x"] + landmarks_3d[12]["x"]) / 2

    # Saque/smash: muñeca por encima del hombro (umbral 0.05 en coords norm.)
    if wrist_y < shoulder_y - 0.05:
        return "saque_o_smash"

    # Forehand vs backhand
    if dominant_hand == "right":
        return "forehand" if wrist_x > hip_center else "backhand"
    else:
        return "forehand" if wrist_x < hip_center else "backhand"


def _compute_stroke_phases(
    impact_frame: dict,
    mp_frames:    list[dict],
    dominant_hand: str = "right",
) -> dict:
    """
    Calcula las 4 fases del golpe usando velocidad angular del codo dominante.

    Ventana: ±1 segundo alrededor del frame de impacto (30 frames a 30fps).
    - Preparación  : frame con ángulo mínimo de codo (máxima flexión) ANTES del impacto
    - Aceleración  : ventana entre preparación e impacto
    - Impacto      : el frame ya detectado
    - Follow-through: frame con ángulo máximo de codo DESPUÉS del impacto

    ROM = ángulo_follow_through - ángulo_preparación = rango real de movimiento.
    """
    elbow_key = "right_elbow" if dominant_hand == "right" else "left_elbow"
    impact_ts  = impact_frame["timestamp"]
    impact_angle = impact_frame["angles"].get(elbow_key, 0.0)

    WINDOW_SEC = 1.0  # ±1 segundo alrededor del impacto

    # Frames dentro de la ventana, ordenados por timestamp
    window = sorted(
        [f for f in mp_frames if abs(f["timestamp"] - impact_ts) <= WINDOW_SEC],
        key=lambda f: f["timestamp"],
    )
    if len(window) < 3:
        return {}

    impact_idx = min(range(len(window)), key=lambda i: abs(window[i]["timestamp"] - impact_ts))

    # Frames antes del impacto (preparación + aceleración)
    pre_frames  = window[:impact_idx]
    # Frames después del impacto (follow-through)
    post_frames = window[impact_idx + 1:]

    if not pre_frames or not post_frames:
        return {}

    # Preparación: mínimo ángulo de codo antes del impacto
    prep_frame  = min(pre_frames,  key=lambda f: f["angles"].get(elbow_key, 999))
    prep_angle  = prep_frame["angles"].get(elbow_key, 0.0)

    # Follow-through: máximo ángulo de codo después del impacto
    ft_frame    = max(post_frames, key=lambda f: f["angles"].get(elbow_key, 0))
    ft_angle    = ft_frame["angles"].get(elbow_key, 0.0)

    accel_frames = impact_idx - window.index(prep_frame)

    return {
        "prep_frame":                prep_frame["frame"],
        "prep_timestamp":            prep_frame["timestamp"],
        "prep_angle_elbow":          round(prep_angle, 1),
        "impact_angle_elbow":        round(impact_angle, 1),
        "followthrough_frame":       ft_frame["frame"],
        "followthrough_timestamp":   ft_frame["timestamp"],
        "followthrough_angle_elbow": round(ft_angle, 1),
        "rom_degrees":               round(ft_angle - prep_angle, 1),
        "accel_frames":              accel_frames,
    }


def _detect_impacts_from_clip(
    ball_frames:   list[dict],
    mp_frames:     list[dict],
    yolo_frames:   list[dict],
    time_offset:   float,
    dominant_hand: str = "right",
    frame_height:  int = 768,
) -> list[dict]:
    """
    Detecta impactos dentro de un clip individual.
    Los timestamps ya vienen en global (sumado time_offset en process_single_clip).
    Asigna stroke_type calculado desde landmarks_3d (preciso), con fallback
    a stroke_hint de YOLO si no hay landmarks disponibles.
    """
    import numpy as np

    detected = [f for f in ball_frames if f["ball_detected"] and f["ball"].get("speed_pixels")]
    if len(detected) < 3 or not mp_frames:
        return []

    timestamps = [f["timestamp"] for f in detected]

    # Cambio 2: criterio de impacto por cambio de dirección del vector velocidad.
    # Dot product negativo entre vectores consecutivos = pelota invirtió trayectoria = impacto real.
    # Más preciso que spike de velocidad: elimina falsos positivos de rebote en piso
    # y saltos geométricos por frames perdidos.
    xs = [f["ball"]["center_x"] for f in detected]
    ys = [f["ball"]["center_y"] for f in detected]

    impact_ts: list[float] = []
    for i in range(1, len(detected) - 1):
        vx_prev = xs[i]     - xs[i - 1]
        vy_prev = ys[i]     - ys[i - 1]
        vx_curr = xs[i + 1] - xs[i]
        vy_curr = ys[i + 1] - ys[i]
        dot = vx_prev * vx_curr + vy_prev * vy_curr

        # Solo considera cambio de dirección si la pelota tiene velocidad suficiente (no ruido)
        speed_prev = (vx_prev**2 + vy_prev**2)**0.5
        if dot < 0 and speed_prev > 15:  # Agregó filtro de velocidad mínima
            impact_ts.append(timestamps[i])

    # Fallback: si no hay cambios de dirección clara, usar los 3 frames de mayor velocidad
    if not impact_ts:
        speeds   = [f["ball"]["speed_pixels"] for f in detected]
        sorted_d = sorted(detected, key=lambda f: f["ball"]["speed_pixels"], reverse=True)
        impact_ts = [f["timestamp"] for f in sorted_d[:3]]

    impacts = []
    for ts in impact_ts:
        closest    = min(mp_frames, key=lambda f: abs(f["timestamp"] - ts))
        diff_ms    = round(abs(closest["timestamp"] - ts) * 1000)
        ball_speed = next(f["ball"]["speed_pixels"] for f in detected if f["timestamp"] == ts)

        if diff_ms <= 200 and ball_speed >= 20:
            # Filtro de codo: descartar si codo está muy cerrado (preparación, no impacto)
            elbow_key = "left_elbow" if dominant_hand == "left" else "right_elbow"
            elbow_angle = closest.get("angles", {}).get(elbow_key, 0)

            # Ángulos válidos para impacto (no preparación):
            # - Forehand/Backhand impacto: codo 100-170° (extendido)
            # - Saque impacto: codo 140-180° (completamente extendido)
            if elbow_angle < 95:
                print(f"  ⛔ ts={ts:.2f}s descartado — fase preparación (codo={elbow_angle:.0f}° < 95°)")
                continue

            # Filtro oponente: pelota en mitad superior = golpe del oponente
            ball_frame = next((f for f in detected if f["timestamp"] == ts), None)
            if ball_frame:
                b_cy = ball_frame["ball"]["center_y"]
                if b_cy < frame_height * 0.5:
                    print(f"  ⛔ ts={ts:.2f}s descartado — oponente (b_cy={b_cy:.0f} < {frame_height*0.5:.0f}px)")
                    continue
            # stroke_type: calcular desde landmarks si están disponibles (preciso),
            # si no, heredar stroke_hint de YOLO (impreciso pero mejor que nada)
            stroke_type = None
            if closest.get("landmarks_3d"):
                stroke_type = _classify_stroke_from_landmarks(
                    closest["landmarks_3d"], dominant_hand
                )
            elif yolo_frames:
                closest_yolo = min(yolo_frames, key=lambda f: abs(f["timestamp"] - ts))
                yolo_diff_ms = abs(closest_yolo["timestamp"] - ts) * 1000
                if yolo_diff_ms <= 500:
                    raw_hint = closest_yolo.get("stroke_hint", "")
                    if "forehand" in raw_hint or "backhand" in raw_hint:
                        stroke_type = "forehand_o_backhand"   # no podemos distinguir sin landmarks
                    elif "saque" in raw_hint or "smash" in raw_hint:
                        stroke_type = "saque_o_smash"

            # Fase D: calcular fases del golpe (prep → accel → impacto → follow-through)
            stroke_phases = _compute_stroke_phases(closest, mp_frames, dominant_hand)

            entry = {
                "impact_timestamp":    ts,
                "mediapipe_frame":     closest["frame"],
                "mediapipe_timestamp": closest["timestamp"],
                "diff_ms":             diff_ms,
                "ball_speed_pixels":   ball_speed,
                "angles":              closest["angles"],
                "shoulder_alignment":  closest.get("shoulder_alignment"),
                "visibility":          closest.get("visibility"),
                "stroke_type":         stroke_type,   # "forehand" | "backhand" | "saque_o_smash" | None
                "stroke_phases":       stroke_phases or None,
            }
            if closest.get("landmarks_3d"):
                entry["landmarks_3d"] = closest["landmarks_3d"]
            impacts.append(entry)

    # Deduplicar por mediapipe_frame
    seen, unique = set(), []
    for f in sorted(impacts, key=lambda x: x["impact_timestamp"]):
        if f["mediapipe_frame"] not in seen:
            seen.add(f["mediapipe_frame"])
            unique.append(f)
    return unique


# ─── ORQUESTADOR FULL-SESSION ─────────────────────────────────
@app.function(
    image=vision_image,
    timeout=7200,                          # 2 horas máximo para sesiones largas
    memory=2048,
    secrets=[modal.Secret.from_name("supabase-key")],
)
def run_vision_pipeline(
    video_bytes:        bytes,
    session_type:       str  = "mezcla",
    user_id:            str  = None,
    previous_session:   dict = None,
    camera_orientation: str  = None,
    equipment_used:     dict = None,
    dominant_hand:      str  = "right",
    store_landmarks:    bool = True,
    vision_job_id:      str  = None,   # pasado desde el endpoint; si es None se genera aquí (local entrypoint)
) -> dict:
    """
    Orquestador principal — reemplaza al de v1.
    Firma idéntica para que el endpoint HTTP y los tests no cambien.
    """
    import concurrent.futures, os, uuid, subprocess, tempfile, sys

    supabase_url  = os.environ["SUPABASE_URL"]
    supabase_key  = os.environ["SUPABASE_SERVICE_KEY"]

    # Si viene del endpoint ya tiene vision_job_id y el registro fue creado allí
    # Si viene del local_entrypoint (testing) lo generamos aquí
    if not vision_job_id:
        vision_job_id = str(uuid.uuid4())
        supabase_post(supabase_url, supabase_key, "vision_results", {
            "id":                 vision_job_id,
            "user_id":            user_id,
            "session_type":       session_type,
            "status":             "vision_processing",
            "camera_orientation": camera_orientation,
            "equipment_used":     equipment_used or {},
            "dominant_hand":      dominant_hand,
        })

    print(f"🚀 Full-Session Pipeline v2 — job: {vision_job_id} | sesión: {session_type} | user: {user_id}")

    # ── Fase 0-A: Volcar video a disco y triaje CPU ───────────
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        video_path = tmp.name

    print("🔍 Fase 0-A: Triaje de acción por CPU...")
    windows = extract_action_windows(video_path)
    print(f"✅ {len(windows)} ventanas de acción detectadas:")
    for i, (s, e) in enumerate(windows):
        print(f"   Clip {i+1}: {s:.1f}s → {e:.1f}s (dur: {e-s:.1f}s)")

    # ── Fase 0-B: Recortar clips con FFmpeg (-c copy) ─────────
    print("✂️  Fase 0-B: Recortando clips con FFmpeg...")
    clip_paths: list[tuple[str, float]] = []   # (path, start_sec)

    for idx, (start, end) in enumerate(windows):
        clip_path = f"/tmp/clip_{idx:04d}.mp4"
        duration  = end - start
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(duration),
                "-c", "copy",           # Sin recodificación = instantáneo
                clip_path,
            ],
            capture_output=True,
        )
        if result.returncode == 0:
            clip_paths.append((clip_path, start))
        else:
            print(f"⚠️ FFmpeg falló en clip {idx}: {result.stderr.decode()[:200]}")

    print(f"✅ {len(clip_paths)}/{len(windows)} clips recortados correctamente")

    # ── Fase 1-GPU: Procesar clips en paralelo distribuido ────
    print(f"🚀 Fase 1-GPU: Enviando {len(clip_paths)} clips a GPU (T4)...")

    all_mp_frames:     list[dict] = []
    all_yolo_frames:   list[dict] = []
    all_ball_frames:   list[dict] = []
    all_impact_frames: list[dict] = []

    # Usar ThreadPoolExecutor para paralelizar llamadas .remote()
    def process_clip_remote(args):
        clip_path, start_sec = args
        with open(clip_path, "rb") as f:
            clip_bytes = f.read()
        return process_single_clip.remote(clip_bytes, time_offset=start_sec, store_landmarks=store_landmarks, dominant_hand=dominant_hand)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(process_clip_remote, arg): arg
            for arg in clip_paths
        }
        for future in concurrent.futures.as_completed(futures):
            clip_path, start_sec = futures[future]
            try:
                clip_result = future.result()
                if clip_result:
                    # Regla C: concatenar listas planas (Regla del documento)
                    all_mp_frames.extend(clip_result.get("mp_frames",     []))
                    all_yolo_frames.extend(clip_result.get("yolo_frames",  []))
                    all_ball_frames.extend(clip_result.get("ball_frames",  []))
                    all_impact_frames.extend(clip_result.get("impact_frames", []))
            except Exception as e:
                print(f"❌ Error en clip offset={start_sec:.1f}s: {e}")

    # Limpiar archivos temporales
    os.unlink(video_path)
    for clip_path, _ in clip_paths:
        try:
            os.unlink(clip_path)
        except Exception:
            pass

    # ── Ordenar todos los frames por timestamp global ─────────
    all_mp_frames.sort(key=lambda f: f["timestamp"])
    all_yolo_frames.sort(key=lambda f: f["timestamp"])
    all_ball_frames.sort(key=lambda f: f["timestamp"])

    # Deduplicar impactos globalmente (pueden solapar entre clips adyacentes)
    all_impact_frames.sort(key=lambda f: f["impact_timestamp"])
    seen_mp_frames, unique_impacts = set(), []
    for imp in all_impact_frames:
        key = round(imp["impact_timestamp"], 1)   # Deduplicar por ventana de 100ms
        if key not in seen_mp_frames:
            seen_mp_frames.add(key)
            unique_impacts.append(imp)
    all_impact_frames = unique_impacts

    print(f"✅ Fase 1-GPU completada:")
    print(f"   MediaPipe frames: {len(all_mp_frames)}")
    print(f"   YOLO frames:      {len(all_yolo_frames)}")
    print(f"   Ball frames:      {len(all_ball_frames)}")
    print(f"   Impactos totales: {len(all_impact_frames)}")

    # ── Fase 2: Calcular summary global (Regla B del documento) ──
    # Solo frames de juego real (descartamos tiempos muertos → promedios más precisos)
    def _avg(frames: list[dict], angle_key: str) -> float:
        vals = [f["angles"][angle_key] for f in frames if angle_key in f.get("angles", {})]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    total_duration = sum(e - s for s, e in windows)

    # Frames slim para Supabase (sin landmarks_3d para no inflar el payload)
    mp_frames_slim   = [{k: v for k, v in f.items() if k != "landmarks_3d"} for f in all_mp_frames]
    yolo_frames_slim = all_yolo_frames
    ball_frames_slim = all_ball_frames

    def _mb(obj): return round(sys.getsizeof(json.dumps(obj)) / (1024 * 1024), 2)
    print(
        f"📦 Payload — mp: {_mb(mp_frames_slim)}MB | "
        f"yolo: {_mb(yolo_frames_slim)}MB | "
        f"ball: {_mb(ball_frames_slim)}MB"
    )

    mediapipe_result = {
        "duration_seconds": round(total_duration, 1),
        "frames_analyzed":  len(all_mp_frames),
        "sample_fps":       30,
        "store_landmarks":  store_landmarks,
        "summary": {
            "avg_right_elbow":        _avg(all_mp_frames, "right_elbow"),
            "avg_left_elbow":         _avg(all_mp_frames, "left_elbow"),
            "avg_right_knee":         _avg(all_mp_frames, "right_knee"),
            "avg_left_knee":          _avg(all_mp_frames, "left_knee"),
            "avg_right_hip":          _avg(all_mp_frames, "right_hip"),
            "avg_left_hip":           _avg(all_mp_frames, "left_hip"),
            "avg_shoulder_alignment": round(
                sum(f.get("shoulder_alignment", 0) for f in all_mp_frames) / len(all_mp_frames), 2
            ) if all_mp_frames else 0.0,
        },
        "frames": mp_frames_slim,
    }

    stroke_counts: dict[str, int] = {}
    for f in all_yolo_frames:
        h = f.get("stroke_hint")
        if h:
            stroke_counts[h] = stroke_counts.get(h, 0) + 1

    yolo_result = {
        "total_frames_analyzed":  len(all_yolo_frames),
        "detection_rate_percent": round(len(all_yolo_frames) / max(len(all_ball_frames), 1) * 100, 1),
        "stroke_hints_summary":   stroke_counts,
        "avg_player_position": {
            "center_x": round(sum(f["player"]["center_x"] for f in all_yolo_frames) / len(all_yolo_frames), 1) if all_yolo_frames else 0,
            "center_y": round(sum(f["player"]["center_y"] for f in all_yolo_frames) / len(all_yolo_frames), 1) if all_yolo_frames else 0,
        },
        "frames": yolo_frames_slim,
    }

    ball_detected = [f for f in all_ball_frames if f["ball_detected"]]
    det_pct       = round(len(ball_detected) / max(len(all_ball_frames), 1) * 100, 1)
    speeds        = [f["ball"]["speed_pixels"] for f in ball_detected if f["ball"] and f["ball"].get("speed_pixels")]

    ball_result = {
        "total_frames_analyzed":       len(all_ball_frames),
        "ball_detection_rate_percent": det_pct,
        "sample_fps":                  10,
        "avg_ball_speed_pixels":       round(sum(speeds) / len(speeds), 1) if speeds else 0,
        "max_ball_speed_pixels":       round(max(speeds), 1) if speeds else 0,
        "frames":                      ball_frames_slim,
    }

    # ── Fase 3: Guardar en Supabase ───────────────────────────
    # Los frames crudos (mp_frames_slim, yolo_frames_slim, ball_frames_slim) NO se guardan
    # en Supabase — solo el summary y los impact_frames (que sí incluyen landmarks_3d).
    # Esto mantiene el payload por debajo del límite de 1MB/fila de Supabase JSONB.
    # Los agentes solo necesitan el summary para los promedios globales y los
    # impact_frames para el análisis por golpe.
    # Issue #3: coverage_ratio = frames capturados / frames esperados a 30fps
    expected_frames  = mediapipe_result["duration_seconds"] * 30
    coverage_ratio   = round(mediapipe_result["frames_analyzed"] / expected_frames, 3) if expected_frames > 0 else 0.0
    coverage_quality = "high" if coverage_ratio > 0.8 else "low" if coverage_ratio < 0.5 else "medium"
    print(f"📊 Coverage MediaPipe: {coverage_ratio:.1%} ({coverage_quality})")

    mediapipe_result_db = {
        "duration_seconds": mediapipe_result["duration_seconds"],
        "frames_analyzed":  mediapipe_result["frames_analyzed"],
        "sample_fps":       mediapipe_result["sample_fps"],
        "store_landmarks":  mediapipe_result["store_landmarks"],
        "coverage_ratio":   coverage_ratio,    # proporción de frames capturados vs esperados a 30fps
        "coverage_quality": coverage_quality,  # "high" | "medium" | "low"
        "summary":          mediapipe_result["summary"],
        # "frames" omitido — no se guarda en DB
    }
    yolo_result_db = {k: v for k, v in yolo_result.items() if k != "frames"}
    ball_result_db  = {k: v for k, v in ball_result.items()  if k != "frames"}

    # Issue #5: processing_gaps_percent = clips que fallaron / windows detectadas
    windows_found   = len(windows)
    clips_processed = len(clip_paths)
    gaps_percent    = round((1 - clips_processed / windows_found) * 100, 1) if windows_found > 0 else 0.0
    if gaps_percent > 5:
        print(f"⚠️  Processing gaps: {gaps_percent:.1f}% de clips fallaron ({windows_found - clips_processed}/{windows_found})")
    else:
        print(f"✅ Processing gaps: {gaps_percent:.1f}% (ok)")

    def _mb(obj): return round(sys.getsizeof(json.dumps(obj)) / (1024 * 1024), 2)
    print(
        f"📦 Payload DB — mp_summary: {_mb(mediapipe_result_db)}MB | "
        f"yolo: {_mb(yolo_result_db)}MB | ball: {_mb(ball_result_db)}MB | "
        f"impacts: {_mb(all_impact_frames)}MB"
    )

    vision_data = {
        "status":             "vision_done",
        "mediapipe_result":   mediapipe_result_db,
        "yolo_result":        yolo_result_db,
        "ball_result":        ball_result_db,
        "impact_frames":      all_impact_frames,
        "camera_orientation": camera_orientation,
        "equipment_used":     equipment_used or {},
        "dominant_hand":      dominant_hand,
        "compression_meta": {
            "mode":                   "full_session_v3",
            "windows_found":          windows_found,
            "clips_processed":        clips_processed,
            "processing_gaps_percent": gaps_percent,  # % de windows que no se procesaron
            "action_seconds":         round(total_duration, 1),
        },
    }

    ok = supabase_patch(supabase_url, supabase_key, "vision_results", vision_job_id, vision_data)
    print(f"{'✅' if ok else '❌'} vision_results actualizado — status: vision_done")

    # ── Disparar agents_pipeline igual que v1 ─────────────────
    print("🤖 Disparando agents_pipeline...")
    try:
        run_agents = modal.Function.from_name(
            "tennis-agents-pipeline",
            "run_agents_pipeline",
        )
        run_agents.spawn(
            vision_job_id      = vision_job_id,
            session_type       = session_type,
            user_id            = user_id,
            previous_session   = previous_session,
            camera_orientation = camera_orientation,
            equipment_used     = equipment_used,
            dominant_hand      = dominant_hand,
        )
        print("✅ agents_pipeline disparado de forma asíncrona")
    except Exception as e:
        err = f"tennis-agents-pipeline no encontrado: {e}"
        print(f"❌ {err}")
        supabase_patch(supabase_url, supabase_key, "vision_results", vision_job_id,
                       {"status": "error", "error_message": err})
        return {
            "vision_job_id": vision_job_id,
            "status":        "error",
            "error":         err,
        }

    return {
        "vision_job_id":    vision_job_id,
        "status":           "vision_done",
        "duration_seconds": round(total_duration, 1),
        "frames_analyzed":  len(all_mp_frames),
        "windows_found":    len(windows),
        "clips_processed":  len(clip_paths),
        "total_impacts":    len(all_impact_frames),
    }


# ─── ENDPOINT HTTP (recibe multipart/form-data) ────────────────
@app.function(
    image=vision_image,
    timeout=900,
    memory=1024,
    secrets=[modal.Secret.from_name("supabase-key")],
)
@modal.fastapi_endpoint(method="POST")
async def vision_endpoint(request: Request):
    """
    Recibe el video como multipart/form-data en el campo 'video'.
    """
    from fastapi.responses import JSONResponse
    import httpx

    try:
        # Parsear multipart/form-data
        form_data = await request.form()
        video_file = form_data.get("video")
        
        if not video_file:
            return {"error": "Falta campo 'video' en form-data"}
        
        video_bytes = await video_file.read()
        session_type = form_data.get("session_type", "mezcla")
        user_id = form_data.get("user_id")
        camera_orientation = form_data.get("camera_orientation")
        equipment_used = form_data.get("equipment_used")
        dominant_hand = form_data.get("dominant_hand", "right")
        store_landmarks = str(form_data.get("store_landmarks", "true")).lower() == "true"
        
    except Exception as e:
        # Fallback: intentar JSON con video_url
        try:
            body = await request.json()
            video_url = body.get("video_url")
            if not video_url:
                return {"error": f"Form parsing error: {str(e)}. JSON fallback failed: no video_url"}
            
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.get(video_url)
                video_bytes = response.content
            
            session_type = body.get("session_type", "mezcla")
            user_id = body.get("user_id")
            camera_orientation = body.get("camera_orientation")
            equipment_used = body.get("equipment_used")
            dominant_hand = body.get("dominant_hand", "right")
            store_landmarks = body.get("store_landmarks", True)
        except Exception as e2:
            return {"error": f"Failed both form and JSON parsing: {str(e2)}"}

    import os
    vision_job_id = str(uuid.uuid4())

    # Crear registro inicial en Supabase desde el endpoint (antes del spawn)
    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_SERVICE_KEY"]
    import httpx as _httpx
    _httpx.post(
        f"{supabase_url}/rest/v1/vision_results",
        headers={
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        json={
            "id":                 vision_job_id,
            "user_id":            user_id,
            "session_type":       session_type,
            "status":             "vision_processing",
            "camera_orientation": camera_orientation,
            "equipment_used":     json.loads(equipment_used) if isinstance(equipment_used, str) else (equipment_used or {}),
            "dominant_hand":      dominant_hand,
        },
        timeout=15,
    )

    # ── Día 3: buscar sesión anterior del usuario activo ────────────────────────
    previous_session = None
    if user_id:
        try:
            prev_resp = _httpx.get(
                f"{supabase_url}/rest/v1/sessions"
                f"?user_id=eq.{user_id}"
                f"&select=global_score,scores_detalle,synthesizer_metadata,created_at"
                f"&order=created_at.desc&limit=1",
                headers={
                    "apikey": supabase_key,
                    "Authorization": f"Bearer {supabase_key}",
                },
                timeout=10,
            )
            rows = prev_resp.json() if prev_resp.status_code == 200 else []
            previous_session = rows[0] if rows else None
            print(f"📊 Sesión anterior: {'encontrada (' + str(previous_session.get('created_at','?'))[:10] + ')' if previous_session else 'ninguna (primera sesión)'}")
        except Exception as e:
            print(f"⚠️  Lookup sesión anterior falló (no crítico): {e}")

    # Spawn asíncrono — retorna vision_job_id en ~2s para que el frontend haga polling
    await run_vision_pipeline.spawn.aio(
        video_bytes,
        session_type,
        user_id,
        previous_session,
        camera_orientation,
        equipment_used,
        dominant_hand,
        store_landmarks,
        vision_job_id,
    )

    return JSONResponse({"vision_job_id": vision_job_id, "status": "vision_processing"}, headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "POST, OPTIONS", "Access-Control-Allow-Headers": "*"})


# ─── ENDPOINT POLLING (idéntico a v1) ────────────────────────
@app.function(
    image=vision_image,
    timeout=30,
    memory=256,
    secrets=[modal.Secret.from_name("supabase-key")],
)
@modal.fastapi_endpoint(method="GET")
async def status_endpoint(vision_job_id: str) -> dict:
    import httpx, os

    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_SERVICE_KEY"]

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{supabase_url}/rest/v1/vision_results"
            f"?id=eq.{vision_job_id}"
            f"&select=id,status,session_id,error_message,created_at",
            headers={
                "apikey":        supabase_key,
                "Authorization": f"Bearer {supabase_key}",
            },
        )

    if resp.status_code != 200 or not resp.json():
        return {"error": "Job no encontrado", "vision_job_id": vision_job_id}

    row = resp.json()[0]
    return {
        "vision_job_id": vision_job_id,
        "status":        row.get("status"),
        "session_id":    row.get("session_id"),
        "error_message": row.get("error_message"),
        "created_at":    row.get("created_at"),
    }


# ─── LOCAL ENTRYPOINT ────────────────────────────────────────
@app.local_entrypoint()
def main(
    video_path:         str  = "test.mp4",
    session_type:       str  = "mezcla",
    camera_orientation: str  = None,
    dominant_hand:      str  = "right",
    store_landmarks:    bool = True,
):
    video_bytes = Path(video_path).read_bytes()
    print(f"📹 Full-Session Pipeline v2 — {video_path} | {session_type} | mano: {dominant_hand}")
    result = run_vision_pipeline.remote(
        video_bytes,
        session_type,
        None,
        None,
        camera_orientation,
        None,
        dominant_hand,
        store_landmarks,
    )
    print(f"\n✅ vision_job_id:   {result['vision_job_id']}")
    print(f"   status:          {result['status']}")
    print(f"   duración acción: {result.get('duration_seconds')}s")
    print(f"   frames:          {result.get('frames_analyzed')}")
    print(f"   ventanas:        {result.get('windows_found')}")
    print(f"   clips GPU:       {result.get('clips_processed')}")
    print(f"   impactos:        {result.get('total_impacts')}")
