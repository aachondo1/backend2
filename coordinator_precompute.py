"""
TennisAI — Pre-cómputo para agent_coordinator
═══════════════════════════════════════════════
Funciones Python puras (sin LLM) que corren en run_agents_pipeline()
ANTES de llamar a agent_coordinator.remote().

Producen datos estadísticos y contextuales listos para inyectar en el
prompt del coordinador, reduciendo la carga cognitiva del LLM y
garantizando que los cálculos numéricos sean deterministas.

Orden de ejecución en run_agents_pipeline():

    # 1. Limpiar ruido biomecánico PRIMERO — antes de cualquier cálculo
    mediapipe_clean, noise_report = detect_and_clean_noise(mediapipe_result)

    # 2. Calcular estadísticas sobre datos ya limpios
    stroke_stats      = compute_stroke_stats(mediapipe_clean, yolo_result, dominant_hand)
    tactical_context  = compute_tactical_context(yolo_result)
    fatigue_context   = compute_fatigue_context(mediapipe_clean, dominant_hand)
    data_quality      = build_data_quality_report(mediapipe_clean, yolo_result,
                                                   ball_result, impact_frames)

    # noise_report se inyecta dentro de data_quality antes de llamar al coordinador
    data_quality["noise_report"] = noise_report

    coordinator_result = agent_coordinator.remote(
        mediapipe_slim, yolo_slim, ball_slim, session_type,
        camera_orientation, equipment_used, dominant_hand,
        stroke_stats     = stroke_stats,
        tactical_context = tactical_context,
        fatigue_context  = fatigue_context,
        data_quality     = data_quality,
    )
"""

import math
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS INTERNOS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_avg(values: list[float]) -> float:
    """Promedio seguro sobre lista; retorna 0.0 si vacía."""
    return round(sum(values) / len(values), 2) if values else 0.0


def _safe_std(values: list[float]) -> float:
    """Desviación estándar muestral (n-1); retorna 0.0 si < 2 elementos."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return round(math.sqrt(variance), 2)


# Mapeo de qué ángulo de codo/rodilla/cadera priorizar según golpe + mano dominante.
# Retorna (elbow_key, knee_key, hip_key, guide_elbow_key)
# guide_elbow_key: brazo NO dominante, relevante para backhand (brazo guía).
def _dominant_keys(stroke: str, dominant_hand: str) -> dict:
    """
    Retorna los keys de ángulo relevantes para un golpe dado y mano dominante.

    Lógica:
      forehand  → brazo dominante es el de golpe
      saque     → brazo dominante es el de golpe
      backhand  → brazo dominante es el de apoyo; el contralateral es el guía

    Returns:
        dict con keys: dom_elbow, dom_knee, dom_hip, guide_elbow
    """
    is_left = dominant_hand == "left"

    if stroke in ("forehand", "saque"):
        # Ambos usan el brazo dominante como brazo de golpe — el mapeo de keys es idéntico.
        # NOTA DE INTERPRETACIÓN (para los agentes downstream):
        #   forehand → dom_elbow evalúa el ángulo en el momento de impacto
        #              (semi-flectado ~90-120° para generar topspin).
        #   saque    → dom_elbow evalúa la extensión máxima en el punto de contacto
        #              (lo más cercano a 180° = brazo completamente extendido).
        #   Esta distinción NO vive aquí; la aplican agent_forehand y agent_saque
        #   al interpretar avg_dom_elbow / std_dom_elbow.
        return {
            "dom_elbow":   "left_elbow"  if is_left else "right_elbow",
            "dom_knee":    "left_knee"   if is_left else "right_knee",
            "dom_hip":     "left_hip"    if is_left else "right_hip",
            "guide_elbow": "right_elbow" if is_left else "left_elbow",  # no prioritario
        }
    elif stroke == "backhand":
        # En backhand el brazo GUÍA es el contralateral al dominante
        return {
            "dom_elbow":   "right_elbow" if is_left else "left_elbow",   # brazo guía
            "dom_knee":    "left_knee"   if is_left else "right_knee",   # pierna dominante sigue siendo base
            "dom_hip":     "left_hip"    if is_left else "right_hip",
            "guide_elbow": "left_elbow"  if is_left else "right_elbow",  # apoyo (dominante)
        }
    else:
        # fallback genérico
        return {
            "dom_elbow":   "left_elbow"  if is_left else "right_elbow",
            "dom_knee":    "left_knee"   if is_left else "right_knee",
            "dom_hip":     "left_hip"    if is_left else "right_hip",
            "guide_elbow": "right_elbow" if is_left else "left_elbow",
        }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 0 — detect_and_clean_noise
# ─────────────────────────────────────────────────────────────────────────────

# Ángulos articulares monitoreados para detección de glitches.
# A 30fps un cambio físicamente posible máximo es ~35° entre frames consecutivos.
# Cualquier delta mayor indica un error de estimación de ViTPose/MediaPipe.
_NOISE_ANGLE_KEYS = [
    "right_elbow", "left_elbow",
    "right_knee",  "left_knee",
    "right_hip",   "left_hip",
    "right_shoulder", "left_shoulder",
]
_NOISE_DELTA_THRESHOLD = 35.0   # grados por frame (a 30fps)


def detect_and_clean_noise(
    mediapipe_data: dict,
    delta_threshold: float = _NOISE_DELTA_THRESHOLD,
) -> tuple[dict, dict]:
    """
    Detecta y elimina frames con glitches biomecánicos antes de cualquier cálculo.

    Un frame se considera contaminado si CUALQUIER ángulo articular cambia más
    de `delta_threshold` grados respecto al frame anterior (ordenado por timestamp).
    Este salto es físicamente imposible a 30fps y corresponde a un error de
    estimación del modelo de pose (ViTPose / MediaPipe).

    Estrategia de limpieza:
      - El frame contaminado se elimina del resultado.
      - Si hay dos frames contaminados consecutivos (burst de ruido), se eliminan ambos.
      - Los frames limpiados se reportan como metadata en noise_report.

    Args:
        mediapipe_data:   dict original con "frames" lista y otros campos.
        delta_threshold:  delta máximo permitido en grados entre frames consecutivos.

    Returns:
        (mediapipe_clean, noise_report)

        mediapipe_clean: copia de mediapipe_data con "frames" ya filtrados.
        noise_report: {
            "frames_original":    int,
            "frames_removed":     int,
            "removal_rate":       float,          # 0-1
            "noise_detected":     bool,
            "contaminated_frames": [              # lista de frames removidos
                {
                    "frame":       int,
                    "timestamp":   float,
                    "joint":       str,            # articulación con el salto
                    "delta_deg":   float,          # magnitud del salto
                    "prev_angle":  float,
                    "bad_angle":   float,
                }
            ],
            "joints_affected":    list[str],      # articulaciones con más glitches
            "recommendation":     str,            # texto para data_quality prompt
        }
    """
    import copy

    mp_frames = mediapipe_data.get("frames", [])

    if len(mp_frames) < 2:
        # Sin suficientes frames para comparar — devolver intacto
        noise_report = {
            "frames_original":     len(mp_frames),
            "frames_removed":      0,
            "removal_rate":        0.0,
            "noise_detected":      False,
            "contaminated_frames": [],
            "joints_affected":     [],
            "recommendation":      "Insuficientes frames para análisis de ruido.",
        }
        return mediapipe_data, noise_report

    frames_sorted = sorted(mp_frames, key=lambda f: f["timestamp"])

    contaminated_indices: set[int] = set()   # índices en frames_sorted
    contamination_log: list[dict]  = []
    joint_hit_count: dict[str, int] = {}

    # ── Paso 1: detectar saltos imposibles entre frames consecutivos ──
    for i in range(1, len(frames_sorted)):
        prev_f = frames_sorted[i - 1]
        curr_f = frames_sorted[i]

        prev_angles = prev_f.get("angles", {})
        curr_angles = curr_f.get("angles", {})

        for joint in _NOISE_ANGLE_KEYS:
            prev_val = prev_angles.get(joint)
            curr_val = curr_angles.get(joint)

            if prev_val is None or curr_val is None:
                continue

            delta = abs(curr_val - prev_val)
            if delta > delta_threshold:
                contaminated_indices.add(i)
                joint_hit_count[joint] = joint_hit_count.get(joint, 0) + 1

                contamination_log.append({
                    "frame":      curr_f.get("frame"),
                    "timestamp":  curr_f.get("timestamp"),
                    "joint":      joint,
                    "delta_deg":  round(delta, 1),
                    "prev_angle": round(prev_val, 1),
                    "bad_angle":  round(curr_val, 1),
                })
                break   # un solo joint contaminado es suficiente para marcar el frame

    # ── Paso 2: expandir burst — si i y i+1 ambos contaminados, agregar i+1 ──
    # Evita que un glitch de 2 frames pase: el frame N+1 compara contra el N malo
    # y podría parecer limpio si el modelo "rebota" al valor original.
    expanded = set(contaminated_indices)
    for idx in contaminated_indices:
        if idx + 1 < len(frames_sorted):
            # Re-verificar el frame siguiente contra el frame previo limpio
            prev_clean_idx = idx - 1
            while prev_clean_idx in contaminated_indices and prev_clean_idx >= 0:
                prev_clean_idx -= 1

            if prev_clean_idx < 0:
                continue

            prev_angles = frames_sorted[prev_clean_idx].get("angles", {})
            next_angles = frames_sorted[idx + 1].get("angles", {})

            for joint in _NOISE_ANGLE_KEYS:
                prev_val = prev_angles.get(joint)
                next_val = next_angles.get(joint)
                if prev_val is None or next_val is None:
                    continue
                if abs(next_val - prev_val) > delta_threshold:
                    expanded.add(idx + 1)
                    break

    # ── Paso 3: construir mediapipe_clean sin los frames contaminados ──
    clean_frames = [
        f for i, f in enumerate(frames_sorted)
        if i not in expanded
    ]

    mediapipe_clean = copy.copy(mediapipe_data)
    mediapipe_clean["frames"] = clean_frames

    # ── Paso 4: construir noise_report ──
    frames_original = len(frames_sorted)
    frames_removed  = len(expanded)
    removal_rate    = round(frames_removed / frames_original, 3) if frames_original else 0.0

    joints_affected = sorted(joint_hit_count, key=joint_hit_count.get, reverse=True)

    if removal_rate == 0.0:
        recommendation = "Sin ruido biomecánico detectado — ángulos confiables."
    elif removal_rate < 0.05:
        recommendation = (
            f"Ruido leve: {frames_removed} frames eliminados ({round(removal_rate*100, 1)}%). "
            f"Articulaciones afectadas: {', '.join(joints_affected[:3])}. "
            "Ángulos calculados sobre datos limpios."
        )
    elif removal_rate < 0.15:
        recommendation = (
            f"⚠️ Ruido moderado: {frames_removed} frames eliminados ({round(removal_rate*100, 1)}%). "
            f"Articulaciones más afectadas: {', '.join(joints_affected[:3])}. "
            "Interpretar ángulos con cautela — posible oclusión parcial o movimiento rápido."
        )
    else:
        recommendation = (
            f"🚨 Ruido alto: {frames_removed} frames eliminados ({round(removal_rate*100, 1)}%). "
            f"Articulaciones más afectadas: {', '.join(joints_affected[:3])}. "
            "La calidad del video o el ángulo de cámara dificultan la estimación de pose. "
            "Scores biomecánicos son estimaciones de baja confianza."
        )

    noise_report = {
        "frames_original":     frames_original,
        "frames_removed":      frames_removed,
        "removal_rate":        removal_rate,
        "noise_detected":      frames_removed > 0,
        "contaminated_frames": contamination_log,
        "joints_affected":     joints_affected,
        "recommendation":      recommendation,
    }

    return mediapipe_clean, noise_report


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 1 — compute_stroke_stats
# ─────────────────────────────────────────────────────────────────────────────

def compute_stroke_stats(
    mediapipe_data: dict,
    yolo_data:      dict,
    dominant_hand:  str = "right",
) -> dict:
    """
    Calcula avg y std de ángulos biomecánicos por tipo de golpe,
    priorizando el brazo dominante según stroke + lateralidad.

    Estrategia de asignación de frames a golpes:
      1. Usa stroke_hints de YOLO para identificar ventanas de tiempo por golpe.
      2. Para cada ventana, busca frames de MediaPipe dentro de ±0.5s.
      3. Filtra frames con visibility < 0.6 (ruido de MediaPipe).

    Args:
        mediapipe_data: dict con keys "frames" (lista) y "summary"
        yolo_data:      dict con keys "frames" (lista) y "stroke_hints_summary"
        dominant_hand:  "right" | "left"

    Returns:
        dict por stroke ("forehand", "backhand", "saque") con estadísticas.
        Ejemplo:
        {
            "forehand": {
                "n_frames": 18,
                "avg_dom_elbow": 112.3,  "std_dom_elbow": 14.1,
                "avg_dom_knee":  143.7,  "std_dom_knee":  8.2,
                "avg_dom_hip":   155.1,  "std_dom_hip":   6.8,
                "avg_shoulder_alignment": 3.9,
                "std_shoulder_alignment": 2.4,   # ← motor de consistencia
                "avg_guide_elbow": 138.2,         # backhand: brazo guía
                "dominant_hand": "right",
                "dom_elbow_key": "right_elbow",
                "low_quality_frames_rejected": 3,
            }
        }
    """
    mp_frames   = mediapipe_data.get("frames", [])
    yolo_frames = yolo_data.get("frames", [])

    if not mp_frames:
        return {}

    # ── Paso 1: construir índice timestamp → frame de MediaPipe ──
    # Usamos índice entero (frame number) como clave para evitar float comparison
    mp_by_frame = {f["frame"]: f for f in mp_frames}
    mp_sorted   = sorted(mp_frames, key=lambda f: f["timestamp"])

    # ── Paso 2: detectar ventanas de tiempo por tipo de golpe desde YOLO ──
    # Un "hint" de forehand/backhand en un frame YOLO abre una ventana ±0.5s
    WINDOW_S = 0.5  # segundos alrededor del hint para buscar frames MediaPipe

    stroke_windows: dict[str, list[tuple[float, float]]] = {
        "forehand": [],
        "backhand": [],
        "saque":    [],
    }

    for yf in yolo_frames:
        hint = yf.get("stroke_hint", "") or ""
        ts   = yf.get("timestamp", 0.0)

        if "forehand" in hint:
            stroke_windows["forehand"].append((ts - WINDOW_S, ts + WINDOW_S))
        elif "backhand" in hint:
            stroke_windows["backhand"].append((ts - WINDOW_S, ts + WINDOW_S))
        elif "saque" in hint or "smash" in hint:
            stroke_windows["saque"].append((ts - WINDOW_S, ts + WINDOW_S))

    # ── Paso 3: para cada golpe, recolectar frames MediaPipe en sus ventanas ──
    MIN_VISIBILITY = 0.6

    result = {}

    for stroke, windows in stroke_windows.items():
        if not windows:
            continue

        keys = _dominant_keys(stroke, dominant_hand)

        # Colapsar ventanas solapadas (merge de intervalos)
        windows_sorted = sorted(windows, key=lambda w: w[0])
        merged = [windows_sorted[0]]
        for start, end in windows_sorted[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Recolectar frames dentro de alguna ventana, con filtro de visibilidad
        stroke_frames    = []
        rejected_low_vis = 0

        for f in mp_sorted:
            ts = f["timestamp"]
            in_window = any(w_start <= ts <= w_end for w_start, w_end in merged)
            if not in_window:
                continue

            vis = f.get("visibility", 1.0)
            if vis < MIN_VISIBILITY:
                rejected_low_vis += 1
                continue

            stroke_frames.append(f)

        if not stroke_frames:
            continue

        # ── Paso 4: calcular estadísticas sobre los frames del golpe ──
        def _extract(key: str) -> list[float]:
            return [
                f["angles"][key]
                for f in stroke_frames
                if key in f.get("angles", {})
            ]

        dom_elbow_vals   = _extract(keys["dom_elbow"])
        dom_knee_vals    = _extract(keys["dom_knee"])
        dom_hip_vals     = _extract(keys["dom_hip"])
        guide_elbow_vals = _extract(keys["guide_elbow"])
        shoulder_vals    = [f.get("shoulder_alignment", 0.0) for f in stroke_frames
                            if f.get("shoulder_alignment") is not None]

        result[stroke] = {
            # Conteo y calidad
            "n_frames":                    len(stroke_frames),
            "low_quality_frames_rejected": rejected_low_vis,

            # Brazo dominante (golpe principal)
            "avg_dom_elbow":  _safe_avg(dom_elbow_vals),
            "std_dom_elbow":  _safe_std(dom_elbow_vals),

            # Pierna y cadera dominante
            "avg_dom_knee":   _safe_avg(dom_knee_vals),
            "std_dom_knee":   _safe_std(dom_knee_vals),
            "avg_dom_hip":    _safe_avg(dom_hip_vals),
            "std_dom_hip":    _safe_std(dom_hip_vals),

            # Rotación de hombros — motor de consistencia y potencia
            "avg_shoulder_alignment": _safe_avg(shoulder_vals),
            "std_shoulder_alignment": _safe_std(shoulder_vals),

            # Brazo guía (especialmente relevante en backhand a dos manos)
            "avg_guide_elbow": _safe_avg(guide_elbow_vals),
            "std_guide_elbow": _safe_std(guide_elbow_vals),

            # Metadata para que el agente sepa qué brazo se analizó
            "dominant_hand":  dominant_hand,
            "dom_elbow_key":  keys["dom_elbow"],
            "guide_elbow_key": keys["guide_elbow"],
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — infer_backhand_grip
# ─────────────────────────────────────────────────────────────────────────────

def infer_backhand_grip(
    impact_frames: list[dict],
    dominant_hand: str = "right",
    proximity_threshold: float = 0.07,
    min_landmarks_required: int = 3,
) -> dict:
    """
    Infiere si el backhand del jugador es a UNA o DOS manos analizando
    la proximidad de ambas muñecas en los impact_frames de backhand.

    Lógica:
      - MediaPipe landmark 15 = muñeca izquierda
      - MediaPipe landmark 16 = muñeca derecha
      - En un backhand a dos manos, ambas muñecas convergen hacia la
        raqueta: la distancia euclidiana entre ellas es pequeña (~0.05-0.10
        en coordenadas normalizadas 0-1).
      - En un backhand a una mano, la muñeca no dominante se separa
        (estabilizador atrás o al costado): distancia mayor (~0.15-0.25).

    Args:
        impact_frames:          lista de todos los impact_frames (se filtra por backhand)
        dominant_hand:          "right" | "left"
        proximity_threshold:    distancia euclidiana máxima para clasificar como 2 manos
        min_landmarks_required: mínimo de frames con landmarks para confiar en la inferencia

    Returns:
        {
            "grip":              "two_handed" | "one_handed" | "unknown",
            "confidence":        float,   # 0-1, fracción de frames que votaron por el grip
            "frames_analyzed":   int,
            "avg_wrist_distance": float,  # distancia promedio entre muñecas
            "biomechanical_note": str,    # texto listo para el prompt del agente
        }
    """
    import math

    _STROKE_MAP = {"backhand": "backhand"}
    is_left = dominant_hand == "left"

    # Índices MediaPipe
    DOM_WRIST   = 15 if is_left else 16   # muñeca dominante
    GUIDE_WRIST = 16 if is_left else 15   # muñeca no dominante (guía)

    bh_frames = [
        f for f in impact_frames
        if f.get("stroke_type") == "backhand" and f.get("landmarks_3d")
    ]

    if len(bh_frames) < min_landmarks_required:
        return {
            "grip":               "unknown",
            "confidence":         0.0,
            "frames_analyzed":    len(bh_frames),
            "avg_wrist_distance": 0.0,
            "biomechanical_note": (
                "No hay suficientes frames con landmarks para determinar "
                "si el backhand es a una o dos manos. "
                "Analiza el golpe con criterios generales."
            ),
        }

    distances   = []
    two_handed_votes = 0

    for f in bh_frames:
        lm = f["landmarks_3d"]
        try:
            dom_x   = lm[DOM_WRIST]["x"]
            dom_y   = lm[DOM_WRIST]["y"]
            guide_x = lm[GUIDE_WRIST]["x"]
            guide_y = lm[GUIDE_WRIST]["y"]
            dist    = math.sqrt((dom_x - guide_x) ** 2 + (dom_y - guide_y) ** 2)
            distances.append(dist)
            if dist <= proximity_threshold:
                two_handed_votes += 1
        except (IndexError, KeyError, TypeError):
            continue

    if not distances:
        return {
            "grip":               "unknown",
            "confidence":         0.0,
            "frames_analyzed":    0,
            "avg_wrist_distance": 0.0,
            "biomechanical_note": "Landmarks inválidos — no se pudo inferir el grip.",
        }

    avg_dist   = round(sum(distances) / len(distances), 3)
    confidence = round(two_handed_votes / len(distances), 2)
    grip       = "two_handed" if confidence >= 0.5 else "one_handed"

    if grip == "two_handed":
        biomechanical_note = (
            f"BACKHAND A DOS MANOS (confianza {int(confidence*100)}%, "
            f"distancia promedio entre muñecas: {avg_dist:.3f}). "
            "La mano NO dominante actúa como MOTOR PRINCIPAL — no solo como estabilizador. "
            "Evaluar: tracción activa de la mano guía, rotación bilateral de caderas, "
            "extensión simétrica de ambos brazos en follow-through. "
            "El codo guía debe extenderse hacia adelante en el impacto (no quedarse atrás). "
            "Un ángulo de codo guía > 140° en impacto indica extensión correcta."
        )
    else:
        biomechanical_note = (
            f"BACKHAND A UNA MANO (confianza {int((1-confidence)*100)}%, "
            f"distancia promedio entre muñecas: {avg_dist:.3f}). "
            "La mano NO dominante actúa como ESTABILIZADOR pasivo durante el swing. "
            "Evaluar: rotación completa del hombro dominante, extensión del brazo de golpe, "
            "posición de la mano guía en la preparación (debe soltar la raqueta antes del swing). "
            "Un ángulo de codo dominante entre 150-170° en impacto indica extensión correcta. "
            "La cadera debe rotar más agresivamente que en el backhand a dos manos."
        )

    return {
        "grip":               grip,
        "confidence":         confidence,
        "frames_analyzed":    len(distances),
        "avg_wrist_distance": avg_dist,
        "biomechanical_note": biomechanical_note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — infer_forehand_grip
# ─────────────────────────────────────────────────────────────────────────────

# Rangos de ángulo del codo dominante en impacto por grip.
# El ángulo aumenta conforme el grip rota hacia western:
#   eastern       → brazo más cerrado, plano de golpe frontal  (~80-110°)
#   semi_western  → rango intermedio, topspin moderado         (~100-135°)
#   western       → windshield wiper, codo abierto, topspin alto (~135+°)
#
# LIMITACIÓN: esto es una proxy indirecta — no hay landmark de grip en MediaPipe.
# La cámara lateral puede comprimir el ángulo real hasta ~10-15°.
# Confidence < 0.65 → se devuelve "unknown" para no sesgar el análisis.
_FH_GRIP_ELBOW_THRESHOLDS = {
    # (lower_bound, upper_bound) del avg_dom_elbow en impacto
    "eastern":      (70,  112),
    "semi_western": (100, 138),
    "western":      (130, 180),
}
# Zona de solapamiento entre grips: si el ángulo cae en overlap, se usa std_dev
# para desempatar. Alta variabilidad (std > 20°) → semi_western por defecto.
_FH_GRIP_CONFIDENCE_THRESHOLD = 0.65


def infer_forehand_grip(
    impact_frames:           list[dict],
    dominant_hand:           str   = "right",
    min_impacts_required:    int   = 3,
    confidence_threshold:    float = _FH_GRIP_CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Infiere el grip de forehand (eastern / semi_western / western) analizando
    el ángulo del codo dominante en los impact_frames de forehand.

    A diferencia de infer_backhand_grip (que usa distancia de muñecas),
    el forehand no tiene un landmark de grip directo en MediaPipe.
    Usamos el avg_dom_elbow en impacto como proxy biomecánico:

      - Eastern (~80-112°):
          Brazo relativamente cerrado en impacto. Plano de golpe más frontal.
          Jugadores que golpean "de plano" con poco topspin.
      - Semi-western (~100-138°):
          Rango intermedio. El más común en tenis moderno amateur y profesional.
          Topspin moderado-alto. Rango con overlap con los otros dos.
      - Western / Full-western (>130°):
          Codo claramente abierto. Windshield wiper marcado.
          Produce mucho topspin. Djokovic, Nadal, Alcaraz operan en 140-160°.
          Los rangos ATP estándar (90-120°) NO aplican para este grip.

    Estrategia de clasificación:
      1. Calcular avg y std del codo dominante en impactos de forehand.
      2. Determinar en qué rango cae el avg.
      3. En zonas de overlap (100-112° y 130-138°): usar std_dev para desempatar.
         std > 20° en overlap → semi_western (grip intermedio más probable).
      4. Si avg fuera de todos los rangos, o n < min_impacts_required,
         o confianza < threshold → devolver "unknown".

    LIMITACIÓN IMPORTANTE:
      El ángulo de codo medido en 2D puede estar comprimido hasta 10-15° en
      vista lateral. Esto sesga los valores hacia grips más "cerrados" de lo real.
      El campo `camera_compression_note` en el output advierte al agente sobre esto.

    Args:
        impact_frames:        lista de todos los impact_frames (se filtra por forehand)
        dominant_hand:        "right" | "left"
        min_impacts_required: mínimo de impactos de forehand para confiar en la inferencia
        confidence_threshold: fracción mínima de impactos que deben votar por el grip (0-1)

    Returns:
        {
            "grip":                  "eastern" | "semi_western" | "western" | "unknown",
            "confidence":            float,    # 0-1
            "frames_analyzed":       int,
            "avg_elbow_at_impact":   float,    # ángulo promedio del codo dominante
            "std_elbow_at_impact":   float,    # desviación estándar
            "biomechanical_note":    str,      # texto listo para el prompt del agente
            "camera_compression_note": str,    # advertencia sobre compresión 2D
        }
    """
    _STROKE_MAP = {"forehand": "forehand"}
    is_left    = dominant_hand == "left"
    elbow_key  = "left_elbow" if is_left else "right_elbow"

    fh_frames = [
        f for f in impact_frames
        if f.get("stroke_type") == "forehand"
    ]

    # ── Caso: insuficientes impactos ──────────────────────────────────────────
    if len(fh_frames) < min_impacts_required:
        return {
            "grip":                    "unknown",
            "confidence":              0.0,
            "frames_analyzed":         len(fh_frames),
            "avg_elbow_at_impact":     0.0,
            "std_elbow_at_impact":     0.0,
            "biomechanical_note": (
                f"Insuficientes impactos de forehand ({len(fh_frames)}) para inferir grip "
                f"(mínimo requerido: {min_impacts_required}). "
                "Analiza el forehand con rangos estándar de referencia general."
            ),
            "camera_compression_note": "",
        }

    # ── Extraer ángulos del codo dominante en impacto ─────────────────────────
    elbow_vals = []
    for f in fh_frames:
        # Los impact_frames pueden tener los ángulos en dos estructuras posibles:
        # directamente en f[elbow_key] (formato del vision pipeline)
        # o en f["angles"][elbow_key] (formato normalizado del coordinador)
        val = f.get(elbow_key) or f.get("angles", {}).get(elbow_key)
        if val is not None:
            elbow_vals.append(float(val))

    if not elbow_vals:
        return {
            "grip":                    "unknown",
            "confidence":              0.0,
            "frames_analyzed":         len(fh_frames),
            "avg_elbow_at_impact":     0.0,
            "std_elbow_at_impact":     0.0,
            "biomechanical_note": (
                "No se encontraron ángulos de codo en los impactos de forehand. "
                "Analiza el forehand con rangos estándar de referencia general."
            ),
            "camera_compression_note": "",
        }

    avg_elbow = _safe_avg(elbow_vals)
    std_elbow = _safe_std(elbow_vals)

    # ── Clasificar grip por rango ─────────────────────────────────────────────
    # Contar "votos" — cuántos impactos individuales caen en cada rango.
    # Permite calcular confidence como fracción de votos para el grip ganador.
    votes = {"eastern": 0, "semi_western": 0, "western": 0}

    for val in elbow_vals:
        if val < 112:
            votes["eastern"] += 1
        elif val > 130:
            votes["western"] += 1
        else:
            votes["semi_western"] += 1

    total_votes = sum(votes.values())
    winner_grip = max(votes, key=votes.get)
    confidence  = round(votes[winner_grip] / total_votes, 2) if total_votes else 0.0

    # ── Desempate en zonas de overlap ─────────────────────────────────────────
    # Si el avg cae en overlap (100-112° o 130-138°) y los votos están divididos,
    # usar std_dev como criterio secundario.
    eastern_votes     = votes["eastern"]
    sw_votes          = votes["semi_western"]
    western_votes     = votes["western"]

    in_lower_overlap  = 100 <= avg_elbow <= 112   # eastern / semi_western
    in_upper_overlap  = 130 <= avg_elbow <= 138   # semi_western / western

    if in_lower_overlap and eastern_votes > 0 and sw_votes > 0:
        # Alta variabilidad en overlap bajo → semi_western más probable
        winner_grip = "semi_western" if std_elbow > 20 else "eastern"
        confidence  = round(max(eastern_votes, sw_votes) / total_votes, 2)

    elif in_upper_overlap and sw_votes > 0 and western_votes > 0:
        # Alta variabilidad en overlap alto → semi_western más probable
        winner_grip = "western" if std_elbow <= 20 else "semi_western"
        confidence  = round(max(sw_votes, western_votes) / total_votes, 2)

    # ── Aplicar threshold de confianza ────────────────────────────────────────
    if confidence < confidence_threshold:
        grip = "unknown"
    else:
        grip = winner_grip

    # ── Construir biomechanical_note ──────────────────────────────────────────
    n = len(elbow_vals)

    if grip == "eastern":
        biomechanical_note = (
            f"FOREHAND EASTERN (confianza {int(confidence*100)}%, "
            f"codo promedio en impacto: {avg_elbow:.1f}° ±{std_elbow:.1f}°, n={n}). "
            "Golpe plano con poco topspin. "
            "Rango óptimo de codo en impacto: 80-112°. "
            "Evaluar: rotación de hombros completa, punto de contacto adelantado, "
            "transferencia de peso al frente. "
            "Un ángulo de codo > 115° puede indicar preparación tardía o impacto retrasado."
        )
    elif grip == "semi_western":
        biomechanical_note = (
            f"FOREHAND SEMI-WESTERN (confianza {int(confidence*100)}%, "
            f"codo promedio en impacto: {avg_elbow:.1f}° ±{std_elbow:.1f}°, n={n}). "
            "El grip más frecuente en tenis moderno. Topspin moderado-alto. "
            "Rango óptimo de codo en impacto: 100-138°. "
            "Evaluar: brushing ascendente de la raqueta, cadera rotando antes que el hombro, "
            "follow-through sobre el hombro no dominante. "
            "Un ángulo de codo < 95° puede indicar golpe demasiado plano para este grip."
        )
    elif grip == "western":
        biomechanical_note = (
            f"FOREHAND WESTERN / FULL-WESTERN (confianza {int(confidence*100)}%, "
            f"codo promedio en impacto: {avg_elbow:.1f}° ±{std_elbow:.1f}°, n={n}). "
            "⚠️ IMPORTANTE: Los rangos ATP estándar de codo (90-120°) NO aplican para este grip. "
            "Djokovic, Nadal y Alcaraz con western grip operan en 140-160° en impacto — es técnica CORRECTA. "
            "Rango óptimo de codo en impacto para western: 130-165°. "
            "Evaluar: windshield wiper pronunciado en follow-through, "
            "punto de contacto más adelantado que en eastern, alta rotación de muñeca."
        )
    else:  # unknown
        biomechanical_note = (
            f"GRIP DE FOREHAND NO DETERMINADO "
            f"(confianza insuficiente: {int(confidence*100)}%, "
            f"codo promedio en impacto: {avg_elbow:.1f}° ±{std_elbow:.1f}°, n={n}). "
            "Usar rangos de referencia general: óptimo de codo en impacto 90-130°. "
            "Si el jugador usa mucho topspin visualmente, los rangos estándar pueden subestimar "
            "ángulos de codo más abiertos (130-160°) que son correctos para grips occidentales."
        )

    # ── Nota de compresión 2D ─────────────────────────────────────────────────
    camera_compression_note = (
        "COMPRESIÓN 2D: El ángulo de codo medido en vista 2D puede estar subestimado "
        "hasta 10-15° respecto al ángulo real en 3D. "
        "Si el grip detectado está en la frontera entre categorías, "
        "considerar el rango del grip superior como posible alternativa."
    )

    return {
        "grip":                    grip,
        "confidence":              confidence,
        "frames_analyzed":         n,
        "avg_elbow_at_impact":     round(avg_elbow, 1),
        "std_elbow_at_impact":     round(std_elbow, 1),
        "biomechanical_note":      biomechanical_note,
        "camera_compression_note": camera_compression_note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 1b — compute_stroke_stats_from_impacts
# ─────────────────────────────────────────────────────────────────────────────

def compute_stroke_stats_from_impacts(
    impact_frames: list[dict],
    dominant_hand: str = "right",
) -> dict:
    """
    Calcula avg y std_dev de ángulos biomecánicos por golpe usando
    directamente los impact_frames del vision pipeline.

    Alternativa a compute_stroke_stats() cuando yolo_frames no están
    disponibles (no se guardan en Supabase). Los impact_frames sí se
    persisten y ya tienen stroke_type + angles + visibility.

    Normalización de stroke_type:
      "forehand"     → "forehand"
      "backhand"     → "backhand"
      "saque_o_smash"→ "saque"

    Args:
        impact_frames: lista de dicts con keys:
            impact_timestamp, angles, shoulder_alignment,
            visibility, stroke_type, ball_speed_pixels, diff_ms
        dominant_hand: "right" | "left"

    Returns:
        Mismo formato que compute_stroke_stats():
        {
            "forehand": {
                "n_impacts": int,
                "avg_dom_elbow":  float, "std_dom_elbow":  float,
                "avg_dom_knee":   float, "std_dom_knee":   float,
                "avg_dom_hip":    float, "std_dom_hip":    float,
                "avg_guide_elbow":float, "std_guide_elbow":float,
                "avg_shoulder_alignment": float,
                "std_shoulder_alignment": float,
                "avg_ball_speed": float, "std_ball_speed": float,
                "dominant_hand":  str,
                "dom_elbow_key":  str,
                "guide_elbow_key":str,
                "low_quality_frames_rejected": int,
            },
            ...
        }
    """
    # Normalizar stroke_type al vocabulario interno
    _STROKE_MAP = {
        "forehand":      "forehand",
        "backhand":      "backhand",
        "saque_o_smash": "saque",
        "saque":         "saque",
        "smash":         "saque",
    }

    MIN_VISIBILITY = 0.6
    MAX_DIFF_MS    = 100   # solo impactos bien sincronizados con la pelota

    # Agrupar por golpe normalizado
    buckets: dict[str, list[dict]] = {"forehand": [], "backhand": [], "saque": []}
    rejected: dict[str, int]       = {"forehand": 0,  "backhand": 0,  "saque": 0}

    for imp in impact_frames:
        raw_type = imp.get("stroke_type") or ""
        stroke   = _STROKE_MAP.get(raw_type)
        if not stroke:
            continue  # tipo desconocido o None — ignorar

        vis     = imp.get("visibility") or 1.0
        diff_ms = imp.get("diff_ms")    or 0

        if vis < MIN_VISIBILITY or diff_ms > MAX_DIFF_MS:
            rejected[stroke] = rejected.get(stroke, 0) + 1
            continue

        buckets[stroke].append(imp)

    result = {}

    for stroke, frames in buckets.items():
        if not frames:
            continue

        keys = _dominant_keys(stroke, dominant_hand)

        def _extract_angle(key: str) -> list[float]:
            return [
                f["angles"][key]
                for f in frames
                if key in f.get("angles", {})
            ]

        dom_elbow_vals   = _extract_angle(keys["dom_elbow"])
        dom_knee_vals    = _extract_angle(keys["dom_knee"])
        dom_hip_vals     = _extract_angle(keys["dom_hip"])
        guide_elbow_vals = _extract_angle(keys["guide_elbow"])
        shoulder_vals    = [
            f["shoulder_alignment"] for f in frames
            if f.get("shoulder_alignment") is not None
        ]
        ball_speed_vals  = [
            f["ball_speed_pixels"] for f in frames
            if f.get("ball_speed_pixels")
        ]

        result[stroke] = {
            # Conteo
            "n_impacts":                   len(frames),
            "low_quality_frames_rejected": rejected.get(stroke, 0),

            # Brazo dominante
            "avg_dom_elbow":  _safe_avg(dom_elbow_vals),
            "std_dom_elbow":  _safe_std(dom_elbow_vals),

            # Pierna y cadera dominante
            "avg_dom_knee":   _safe_avg(dom_knee_vals),
            "std_dom_knee":   _safe_std(dom_knee_vals),
            "avg_dom_hip":    _safe_avg(dom_hip_vals),
            "std_dom_hip":    _safe_std(dom_hip_vals),

            # Hombros — motor de consistencia
            "avg_shoulder_alignment": _safe_avg(shoulder_vals),
            "std_shoulder_alignment": _safe_std(shoulder_vals),

            # Brazo guía (backhand)
            "avg_guide_elbow": _safe_avg(guide_elbow_vals),
            "std_guide_elbow": _safe_std(guide_elbow_vals),

            # Velocidad de pelota en impactos de este golpe
            "avg_ball_speed": _safe_avg(ball_speed_vals),
            "std_ball_speed": _safe_std(ball_speed_vals),

            # Metadata de lateralidad
            "dominant_hand":   dominant_hand,
            "dom_elbow_key":   keys["dom_elbow"],
            "guide_elbow_key": keys["guide_elbow"],
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 2 — compute_tactical_context
# ─────────────────────────────────────────────────────────────────────────────

def compute_tactical_context(yolo_data: dict) -> dict:
    """
    Infiere el contexto táctico del jugador a partir de los datos de YOLO.

    Usa:
      - avg_player_position: center_y normalizado (0=top, 1=bottom del frame)
        como proxy de posición en cancha (red vs fondo).
      - stroke_hints_summary: distribución de golpes detectados.
      - Historial de posiciones frame a frame para contar aproximaciones a la red.

    Convención de center_y en el video:
      - Jugador filmado desde el fondo: values cercanos a 0.5 = línea de fondo,
        values bajos = cerca de la red (se acerca a cámara y sube en frame).
      - Usamos umbral de center_y < 0.35 como "zona de red".

    Returns:
        {
            "dominant_position": "baseline" | "net" | "mixed",
            "net_approaches": int,
            "stroke_distribution": {"forehand": 0.6, "backhand": 0.3, "saque": 0.1},
            "avg_center_y": float,
            "implication": str    ← texto listo para prompt del coordinador
        }
    """
    yolo_frames = yolo_data.get("frames", [])
    hints_summary = yolo_data.get("stroke_hints_summary", {})
    avg_pos = yolo_data.get("avg_player_position", {})

    # ── Posición dominante ────────────────────────────────────
    NET_THRESHOLD      = 0.35   # center_y normalizado: por debajo = zona de red
    BASELINE_THRESHOLD = 0.55   # por encima = zona de fondo

    avg_center_y = avg_pos.get("center_y", 0.5)

    if avg_center_y < NET_THRESHOLD:
        dominant_position = "net"
    elif avg_center_y > BASELINE_THRESHOLD:
        dominant_position = "baseline"
    else:
        dominant_position = "mixed"

    # ── Contar aproximaciones a la red (transiciones hacia zona de red) ──
    net_approaches   = 0
    was_at_baseline  = False

    for f in sorted(yolo_frames, key=lambda x: x.get("timestamp", 0)):
        player = f.get("player", {})
        cy     = player.get("center_y", 0.5)

        if cy > BASELINE_THRESHOLD:
            was_at_baseline = True
        elif cy < NET_THRESHOLD and was_at_baseline:
            net_approaches  += 1
            was_at_baseline  = False

    # ── Distribución de golpes ────────────────────────────────
    # stroke_hints_summary tiene keys como "posible_forehand_o_backhand",
    # "posible_saque_o_smash", "posicion_base", "movimiento_general"
    total_strokes = sum(hints_summary.values()) or 1

    # Separar hints exclusivos de hints ambiguos (forehand_o_backhand)
    fhbh_count  = sum(v for k, v in hints_summary.items() if "forehand_o_backhand" in k)
    fh_only     = sum(v for k, v in hints_summary.items()
                      if "forehand" in k and "forehand_o_backhand" not in k)
    bh_only     = sum(v for k, v in hints_summary.items()
                      if "backhand" in k and "forehand_o_backhand" not in k)
    sq_count    = sum(v for k, v in hints_summary.items() if "saque" in k or "smash" in k)

    # Los hints forehand_o_backhand los distribuimos 50/50 como estimación
    fh_total = fh_only + fhbh_count * 0.5
    bh_total = bh_only + fhbh_count * 0.5

    stroke_distribution = {
        "forehand": round(fh_total / total_strokes, 2),
        "backhand": round(bh_total / total_strokes, 2),
        "saque":    round(sq_count  / total_strokes, 2),
    }

    # ── Implicación táctica para el prompt ───────────────────
    if dominant_position == "baseline":
        implication = (
            "Jugador de fondo de cancha. "
            "Evaluar forehand y backhand con tolerancia normal en tiempo de preparación. "
            "Priorizar análisis de groundstrokes sobre volleys."
        )
    elif dominant_position == "net":
        implication = (
            "Jugador de red / voleo activo. "
            "Reducir peso del análisis de groundstrokes — "
            "priorizar control y tiempo de reacción sobre potencia."
        )
    else:
        implication = (
            "Jugador mixto (fondo + red). "
            "Balancear análisis entre groundstrokes y voleo. "
            f"Se detectaron {net_approaches} aproximaciones a la red."
        )

    if sq_count > 0:
        sq_pct = round(sq_count / total_strokes * 100)
        implication += f" {sq_pct}% de los hints corresponden a saque/smash."

    return {
        "dominant_position":    dominant_position,
        "net_approaches":       net_approaches,
        "stroke_distribution":  stroke_distribution,
        "avg_center_y":         round(avg_center_y, 3),
        "implication":          implication,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 3 — compute_fatigue_context
# ─────────────────────────────────────────────────────────────────────────────

def compute_fatigue_context(
    mediapipe_data: dict,
    dominant_hand:  str = "right",
    window_pct:     float = 0.10,
) -> dict:
    """
    Detecta degradación técnica por fatiga comparando el inicio vs el final
    de la sesión a nivel global (todos los frames, independiente del golpe).

    Compara el primer `window_pct`% y el último `window_pct`% de frames
    ordenados por timestamp. Analiza articulaciones clave asociadas a fatiga
    muscular real en tenis:
      - Rodilla dominante  → flexión de piernas (primer indicador de fatiga)
      - Cadera dominante   → rotación y estabilidad del tronco
      - Alineación hombros → preparación y rotación de hombros

    Una rodilla con ángulo MAYOR al final indica MENOS flexión
    (el jugador se pone más erguido — señal clásica de fatiga de cuádriceps).

    Args:
        mediapipe_data: dict con "frames" lista (ya limpiado por detect_and_clean_noise)
        dominant_hand:  "right" | "left"
        window_pct:     fracción de frames a comparar en cada extremo (default 10%)

    Returns:
        {
            "fatigue_detected":        bool,
            "window_size":             int,       # frames en cada ventana
            "knee_start_avg":          float,     # ángulo rodilla dominante inicio
            "knee_end_avg":            float,     # ángulo rodilla dominante final
            "knee_degradation_pct":    float,     # >0 = más erguido (peor) al final
            "hip_start_avg":           float,
            "hip_end_avg":             float,
            "hip_degradation_pct":     float,
            "shoulder_start_avg":      float,
            "shoulder_end_avg":        float,
            "shoulder_degradation_pct": float,
            "overall_fatigue_score":   float,     # 0-1 compuesto
            "narrative":               str,       # texto listo para prompt
        }
    """
    mp_frames = mediapipe_data.get("frames", [])

    # Necesitamos mínimo 20 frames para que las ventanas sean representativas
    MIN_FRAMES = 20
    if len(mp_frames) < MIN_FRAMES:
        return {
            "fatigue_detected":         False,
            "window_size":              0,
            "knee_start_avg":           0.0,
            "knee_end_avg":             0.0,
            "knee_degradation_pct":     0.0,
            "hip_start_avg":            0.0,
            "hip_end_avg":              0.0,
            "hip_degradation_pct":      0.0,
            "shoulder_start_avg":       0.0,
            "shoulder_end_avg":         0.0,
            "shoulder_degradation_pct": 0.0,
            "overall_fatigue_score":    0.0,
            "narrative": "Insuficientes frames para análisis de fatiga.",
        }

    frames_sorted = sorted(mp_frames, key=lambda f: f["timestamp"])
    n             = len(frames_sorted)
    window_size   = max(1, int(n * window_pct))

    start_frames = frames_sorted[:window_size]
    end_frames   = frames_sorted[-window_size:]

    # ── Determinar keys según mano dominante ──────────────────
    is_left   = dominant_hand == "left"
    knee_key  = "left_knee"  if is_left else "right_knee"
    hip_key   = "left_hip"   if is_left else "right_hip"

    def _avg_angle(frames: list[dict], key: str) -> float:
        vals = [
            f["angles"][key]
            for f in frames
            if key in f.get("angles", {})
            and f.get("visibility", 1.0) >= 0.6
        ]
        return _safe_avg(vals)

    def _avg_shoulder(frames: list[dict]) -> float:
        vals = [
            f["shoulder_alignment"]
            for f in frames
            if f.get("shoulder_alignment") is not None
            and f.get("visibility", 1.0) >= 0.6
        ]
        return _safe_avg(vals)

    # ── Calcular promedios inicio / fin ───────────────────────
    knee_start = _avg_angle(start_frames, knee_key)
    knee_end   = _avg_angle(end_frames,   knee_key)

    hip_start  = _avg_angle(start_frames, hip_key)
    hip_end    = _avg_angle(end_frames,   hip_key)

    sh_start   = _avg_shoulder(start_frames)
    sh_end     = _avg_shoulder(end_frames)

    # ── Calcular degradación porcentual ───────────────────────
    # Rodilla: ángulo mayor al final = menos flexión = degradación positiva
    # Cadera:  ángulo menor al final = menos rotación = degradación positiva
    # Hombros: alineación mayor al final = hombros más cerrados = degradación positiva

    def _pct_change(start: float, end: float, invert: bool = False) -> float:
        """Retorna % de cambio. invert=True cuando subir el valor es degradación."""
        if start == 0.0:
            return 0.0
        delta = (end - start) / start * 100
        return round(delta if invert else -delta, 1)

    knee_deg_pct = _pct_change(knee_start, knee_end, invert=True)   # más erguido = peor
    hip_deg_pct  = _pct_change(hip_start,  hip_end,  invert=False)  # menos rotación = peor
    sh_deg_pct   = _pct_change(sh_start,   sh_end,   invert=True)   # hombros más cerrados = peor

    # ── Score compuesto de fatiga ──────────────────────────────
    # Solo penalizamos degradaciones positivas (el jugador empeoró)
    # Ponderación: rodilla 50%, cadera 30%, hombros 20%
    # Normalizamos: 20% de degradación = score 1.0 (techo)
    NORM_PCT = 20.0

    knee_score = min(1.0, max(0.0, knee_deg_pct / NORM_PCT)) * 0.50
    hip_score  = min(1.0, max(0.0, hip_deg_pct  / NORM_PCT)) * 0.30
    sh_score   = min(1.0, max(0.0, sh_deg_pct   / NORM_PCT)) * 0.20

    overall_fatigue_score = round(knee_score + hip_score + sh_score, 3)
    fatigue_detected      = overall_fatigue_score >= 0.15  # umbral mínimo de señal

    # ── Narrativa para el prompt del coordinador ──────────────
    parts = []

    if knee_deg_pct >= 5.0:
        parts.append(
            f"El jugador redujo la flexión de rodillas un {knee_deg_pct}% "
            f"al final de la sesión (rodilla: {knee_start}° → {knee_end}°). "
            "Señal clásica de fatiga de cuádriceps."
        )
    if hip_deg_pct >= 5.0:
        parts.append(
            f"La rotación de cadera disminuyó un {hip_deg_pct}% "
            f"({hip_start}° → {hip_end}°). "
            "El jugador perdió transferencia de energía desde el tronco."
        )
    if sh_deg_pct >= 5.0:
        parts.append(
            f"La preparación de hombros se cerró un {sh_deg_pct}% "
            f"({sh_start}° → {sh_end}°). "
            "Probable fatiga de hombro o pérdida de concentración táctica."
        )

    if not parts:
        narrative = (
            "Sin señales claras de fatiga técnica entre el inicio y el final "
            f"de la sesión (ventanas de {window_size} frames cada una)."
        )
    else:
        intro = (
            f"⚠️ Fatiga técnica detectada (score {overall_fatigue_score:.2f}/1.0). "
            f"Comparando primer y último {round(window_pct*100)}% de la sesión: "
        )
        narrative = intro + " ".join(parts)

    return {
        "fatigue_detected":         fatigue_detected,
        "window_size":              window_size,
        "knee_start_avg":           knee_start,
        "knee_end_avg":             knee_end,
        "knee_degradation_pct":     knee_deg_pct,
        "hip_start_avg":            hip_start,
        "hip_end_avg":              hip_end,
        "hip_degradation_pct":      hip_deg_pct,
        "shoulder_start_avg":       sh_start,
        "shoulder_end_avg":         sh_end,
        "shoulder_degradation_pct": sh_deg_pct,
        "overall_fatigue_score":    overall_fatigue_score,
        "narrative":                narrative,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 4 — build_data_quality_report
# ─────────────────────────────────────────────────────────────────────────────

def build_data_quality_report(
    mediapipe_data: dict,
    yolo_data:      dict,
    ball_data:      dict,
    impact_frames:  list[dict],
) -> dict:
    """
    Calcula métricas de cobertura y calidad de cada fuente de datos.

    Incluye ball_sync_rate: fracción de impactos cuyo frame MediaPipe más
    cercano tiene diff_ms <= 100ms. Es la métrica clave para la validación
    post-LLM (sync_impact_quality_check).

    Args:
        mediapipe_data: dict con "frames_analyzed" y "frames" lista
        yolo_data:      dict con "total_frames_analyzed" y "detection_rate_percent"
        ball_data:      dict con "total_frames_analyzed" y "ball_detection_rate_percent"
        impact_frames:  lista de impactos detectados por el vision pipeline

    Returns:
        {
            "mediapipe_coverage":        float,  # 0-1
            "ball_detection_coverage":   float,  # 0-1
            "yolo_detection_rate":       float,  # 0-1
            "low_visibility_frames_pct": float,  # 0-1
            "impacts_total":             int,
            "impacts_with_ball_sync":    int,    # diff_ms <= 100
            "impacts_high_quality":      int,    # diff_ms <= 100 AND visibility >= 0.6
            "ball_sync_rate":            float,  # 0-1  ← semilla para validación LLM
            "overall_quality_score":     float,  # 0-1 compuesto
            "recommendation":            str,    # texto para el prompt
        }
    """
    # ── MediaPipe coverage ────────────────────────────────────
    mp_frames        = mediapipe_data.get("frames", [])
    frames_analyzed  = mediapipe_data.get("frames_analyzed", len(mp_frames)) or 1

    # Frames con visibility >= 0.6 (confiables para ángulos)
    high_vis_frames  = [f for f in mp_frames if f.get("visibility", 0) >= 0.6]
    low_vis_frames   = [f for f in mp_frames if f.get("visibility", 0) < 0.6]

    mediapipe_coverage      = round(len(high_vis_frames) / frames_analyzed, 3)
    low_visibility_pct      = round(len(low_vis_frames)  / frames_analyzed, 3) if mp_frames else 0.0

    # ── Ball detection coverage ───────────────────────────────
    ball_total    = ball_data.get("total_frames_analyzed", 1) or 1
    ball_det_pct  = ball_data.get("ball_detection_rate_percent", 0)
    ball_coverage = round(ball_det_pct / 100, 3)

    # ── YOLO detection rate ───────────────────────────────────
    yolo_det_pct  = yolo_data.get("detection_rate_percent", 0)
    yolo_rate     = round(yolo_det_pct / 100, 3)

    # ── Ball sync: impactos con diff_ms <= 100ms ──────────────
    # diff_ms ya está calculado en el vision pipeline para cada impact_frame
    # (distancia temporal entre el evento de pelota y el frame MediaPipe más cercano)
    SYNC_THRESHOLD_MS  = 100
    VIS_THRESHOLD      = 0.6

    impacts_total           = len(impact_frames)
    impacts_with_ball_sync  = sum(
        1 for imp in impact_frames
        if imp.get("diff_ms", 999) <= SYNC_THRESHOLD_MS
    )
    impacts_high_quality    = sum(
        1 for imp in impact_frames
        if imp.get("diff_ms", 999) <= SYNC_THRESHOLD_MS
        and imp.get("visibility", 0) >= VIS_THRESHOLD
    )

    ball_sync_rate = round(impacts_with_ball_sync / impacts_total, 3) if impacts_total else 0.0

    # ── Score compuesto de calidad global ─────────────────────
    # Ponderación: MediaPipe es la fuente más importante (40%),
    # ball sync define confianza en potencia (35%), YOLO en clasificación (25%)
    overall_quality_score = round(
        mediapipe_coverage * 0.40
        + ball_sync_rate   * 0.35
        + yolo_rate        * 0.25,
        3,
    )

    # ── Recomendación para el prompt ─────────────────────────
    parts = []

    if mediapipe_coverage >= 0.75:
        parts.append("MediaPipe confiable (≥75% frames con buena visibilidad)")
    elif mediapipe_coverage >= 0.5:
        parts.append("MediaPipe parcialmente confiable (50-75% frames)")
    else:
        parts.append("⚠️ MediaPipe con baja cobertura (<50%) — ángulos son estimaciones")

    if ball_sync_rate >= 0.7:
        parts.append("pelota bien sincronizada con pose (≥70% impactos)")
    elif ball_sync_rate >= 0.4:
        parts.append("pelota moderadamente sincronizada — usar potencia con cautela")
    else:
        parts.append("⚠️ baja sincronización pelota-pose — no scorear potencia con confianza")

    if low_visibility_pct > 0.3:
        parts.append(f"⚠️ {round(low_visibility_pct*100)}% frames con visibilidad baja (oclusión o contraluz)")

    recommendation = " | ".join(parts)

    return {
        "mediapipe_coverage":        mediapipe_coverage,
        "ball_detection_coverage":   ball_coverage,
        "yolo_detection_rate":       yolo_rate,
        "low_visibility_frames_pct": low_visibility_pct,
        "impacts_total":             impacts_total,
        "impacts_with_ball_sync":    impacts_with_ball_sync,
        "impacts_high_quality":      impacts_high_quality,
        "ball_sync_rate":            ball_sync_rate,
        "overall_quality_score":     overall_quality_score,
        "recommendation":            recommendation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 4 — sync_impact_quality_check  (post-LLM, corre en run_agents_pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def sync_impact_quality_check(
    coordinator_result: dict,
    impact_frames:      list[dict],
    data_quality:       dict,
) -> dict:
    """
    Validación post-LLM: compara los frames de impacto que el LLM asignó
    en frames_by_stroke[stroke]["impacto"] contra los impactos reales del
    ball tracker (impact_frames con diff_ms).

    Si el frame de impacto del LLM no tiene un impact_frame del ball tracker
    dentro de ±3 frames Y diff_ms <= 100ms, lo marca como no validado y
    penaliza data_quality.ball_detection_coverage para ese golpe.

    Modifica coordinator_result["data_quality"] in-place y retorna
    el resultado enriquecido con campo "impact_validation" por golpe.

    Args:
        coordinator_result: output del agent_coordinator (se modifica in-place)
        impact_frames:      lista raw del vision pipeline
        data_quality:       dict ya calculado por build_data_quality_report

    Returns:
        coordinator_result enriquecido con:
        coordinator_result["data_quality"]["impact_validation"] = {
            "forehand": {"ball_validated": True,  "matched_frame": 37, "diff_ms": 45},
            "backhand": {"ball_validated": False, "matched_frame": None, "diff_ms": None,
                         "warning": "No impact_frame del ball tracker dentro de ±3 frames"},
        }
    """
    FRAME_TOLERANCE = 3     # frames de margen (a 30fps = 100ms)
    SYNC_MS_LIMIT   = 100   # igual que en build_data_quality_report

    frames_by_stroke = coordinator_result.get("frames_by_stroke", {})
    impact_validation = {}

    # Indexar impact_frames del ball tracker por mediapipe_frame para lookup O(1)
    ball_impact_by_mp_frame = {
        imp.get("mediapipe_frame"): imp
        for imp in impact_frames
        if imp.get("mediapipe_frame") is not None
    }

    for stroke, phase_dict in frames_by_stroke.items():
        # frames_by_stroke puede ser lista plana (viejo formato) o dict de fases (nuevo)
        if isinstance(phase_dict, dict):
            impact_frame_indices = phase_dict.get("impacto", [])
        elif isinstance(phase_dict, list):
            impact_frame_indices = phase_dict  # fallback: tratar toda la lista como impacto
        else:
            continue

        if not impact_frame_indices:
            # ── FALLBACK: LLM no asignó frames pero el pre-cómputo sí tiene impactos ──
            # Si el ball tracker tiene impactos reales de este golpe con diff_ms confiable,
            # usarlos directamente en lugar de marcar como no validado.
            # Esto ocurre cuando mediapipe_coverage=0% y el LLM no puede asignar frames,
            # pero el vision pipeline sí detectó impactos reales con sincronización buena.
            precomputed_for_stroke = [
                imp for imp in impact_frames
                if imp.get("stroke_type") == stroke
                and imp.get("diff_ms") is not None
                and imp.get("diff_ms") <= SYNC_MS_LIMIT
                and imp.get("ball_speed_pixels", 0) > 0
            ]
            if precomputed_for_stroke:
                # Tomar el impacto con mayor velocidad de pelota (el más representativo)
                best = max(precomputed_for_stroke, key=lambda x: x.get("ball_speed_pixels", 0))
                impact_validation[stroke] = {
                    "ball_validated":  True,
                    "matched_frame":   best.get("mediapipe_frame"),
                    "diff_ms":         best.get("diff_ms"),
                    "ball_speed":      best.get("ball_speed_pixels"),
                    "fallback_used":   True,
                    "warning": (
                        "LLM no asignó frame de impacto — usando impacto del pre-cómputo "
                        f"(frame {best.get('mediapipe_frame')}, diff_ms={best.get('diff_ms')}, "
                        f"speed={best.get('ball_speed_pixels'):.1f}px/frame)."
                    ),
                }
            else:
                impact_validation[stroke] = {
                    "ball_validated": False,
                    "matched_frame":  None,
                    "diff_ms":        None,
                    "warning":        "LLM no asignó frame de impacto para este golpe",
                }
            continue

        # Tomar el primer frame de impacto asignado por el LLM para este golpe
        llm_impact_frame = impact_frame_indices[0] if impact_frame_indices else None

        if llm_impact_frame is None:
            impact_validation[stroke] = {
                "ball_validated": False,
                "matched_frame":  None,
                "diff_ms":        None,
                "warning":        "Frame de impacto LLM es None",
            }
            continue

        # Buscar un impact_frame del ball tracker dentro de ±FRAME_TOLERANCE frames
        matched_imp = None
        for offset in range(FRAME_TOLERANCE + 1):
            for candidate_frame in [llm_impact_frame + offset, llm_impact_frame - offset]:
                if candidate_frame in ball_impact_by_mp_frame:
                    candidate = ball_impact_by_mp_frame[candidate_frame]
                    if candidate.get("diff_ms", 999) <= SYNC_MS_LIMIT:
                        matched_imp = candidate
                        break
            if matched_imp:
                break

        if matched_imp:
            impact_validation[stroke] = {
                "ball_validated": True,
                "matched_frame":  matched_imp.get("mediapipe_frame"),
                "diff_ms":        matched_imp.get("diff_ms"),
                "ball_speed":     matched_imp.get("ball_speed_pixels"),
            }
        else:
            impact_validation[stroke] = {
                "ball_validated": False,
                "matched_frame":  None,
                "diff_ms":        None,
                "warning": (
                    f"No se encontró impact_frame del ball tracker "
                    f"dentro de ±{FRAME_TOLERANCE} frames del impacto LLM (frame {llm_impact_frame}). "
                    f"Potencia de este golpe no es confiable."
                ),
            }

    # ── Escribir validación en data_quality del coordinator ──
    if "data_quality" not in coordinator_result:
        coordinator_result["data_quality"] = {}

    coordinator_result["data_quality"].update(data_quality)
    coordinator_result["data_quality"]["impact_validation"] = impact_validation

    # Penalizar overall_quality_score si hay golpes sin validar
    unvalidated = sum(
        1 for v in impact_validation.values() if not v["ball_validated"]
    )
    total_validated = len(impact_validation) or 1
    validation_rate = (total_validated - unvalidated) / total_validated

    current_score = coordinator_result["data_quality"].get("overall_quality_score", 1.0)
    # Penalización proporcional: si 50% sin validar, baja 15% el score
    penalty = (1 - validation_rate) * 0.15
    coordinator_result["data_quality"]["overall_quality_score"] = round(
        max(0.0, current_score - penalty), 3
    )

    return coordinator_result


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 5 — compute_fatigue_by_stroke
# ─────────────────────────────────────────────────────────────────────────────

def compute_fatigue_by_stroke(
    impact_frames: list[dict],
    dominant_hand: str = "right",
    window_n:      int = 3,
) -> dict:
    """
    Detecta degradación técnica por fatiga comparando los primeros N
    impactos vs los últimos N impactos de cada golpe.

    A diferencia de compute_fatigue_context() que opera sobre todos los
    frames de MediaPipe, esta función opera sobre impact_frames (persistidos
    en Supabase) y entrega el análisis por golpe específico.

    Args:
        impact_frames: lista de impact_frames con stroke_type + angles + timestamp
        dominant_hand: "right" | "left"
        window_n:      cantidad de impactos a comparar en cada extremo (default 3)

    Returns:
        {
            "forehand": {
                "n_impacts":          int,
                "fatigue_detected":   bool,
                "knee_start_avg":     float,   # primeros N impactos
                "knee_end_avg":       float,   # últimos N impactos
                "knee_delta":         float,   # end - start (>0 = más erguido = peor)
                "elbow_start_avg":    float,
                "elbow_end_avg":      float,
                "elbow_delta":        float,   # >0 = más abierto al final = posible fatiga
                "shoulder_start_avg": float,
                "shoulder_end_avg":   float,
                "shoulder_delta":     float,
                "narrative":          str,     # texto listo para prompt
            },
            "backhand": { ... },
            "saque":    { ... },
        }
    """
    _STROKE_MAP = {
        "forehand":      "forehand",
        "backhand":      "backhand",
        "saque_o_smash": "saque",
        "saque":         "saque",
    }

    is_left    = dominant_hand == "left"
    elbow_key  = "left_elbow"  if is_left else "right_elbow"
    knee_key   = "left_knee"   if is_left else "right_knee"

    # Agrupar por golpe, ordenados por timestamp
    buckets: dict[str, list[dict]] = {"forehand": [], "backhand": [], "saque": []}
    for imp in sorted(impact_frames, key=lambda f: f.get("impact_timestamp", 0)):
        stroke = _STROKE_MAP.get(imp.get("stroke_type") or "")
        if stroke:
            buckets[stroke].append(imp)

    result = {}

    for stroke, frames in buckets.items():
        if len(frames) < window_n * 2:
            # No hay suficientes impactos para comparar
            result[stroke] = {
                "n_impacts":        len(frames),
                "fatigue_detected": False,
                "narrative":        f"Insuficientes impactos de {stroke} para análisis de fatiga (mín {window_n*2}).",
            }
            continue

        early = frames[:window_n]
        late  = frames[-window_n:]

        def _avg_angle(subset, key):
            vals = [f.get("angles", {}).get(key) for f in subset if f.get("angles", {}).get(key)]
            return round(sum(vals) / len(vals), 1) if vals else 0.0

        def _avg_shoulder(subset):
            vals = [f.get("shoulder_alignment") for f in subset if f.get("shoulder_alignment") is not None]
            return round(sum(vals) / len(vals), 1) if vals else 0.0

        knee_start     = _avg_angle(early, knee_key)
        knee_end       = _avg_angle(late,  knee_key)
        knee_delta     = round(knee_end - knee_start, 1)

        elbow_start    = _avg_angle(early, elbow_key)
        elbow_end      = _avg_angle(late,  elbow_key)
        elbow_delta    = round(elbow_end - elbow_start, 1)

        shoulder_start = _avg_shoulder(early)
        shoulder_end   = _avg_shoulder(late)
        shoulder_delta = round(shoulder_end - shoulder_start, 1)

        # Fatiga detectada si rodilla se extiende (más erguido) O
        # alineación de hombros empeora (más desalineados)
        fatigue_detected = (knee_delta > 5) or (shoulder_delta > 3)

        # Narrativa lista para el prompt
        if not fatigue_detected:
            narrative = f"Sin señales de fatiga en {stroke} — técnica estable a lo largo de la sesión."
        else:
            parts = []
            if knee_delta > 5:
                parts.append(
                    f"rodilla {'+' if knee_delta > 0 else ''}{knee_delta}° al final "
                    f"(menos flexión — señal clásica de fatiga de cuádriceps)"
                )
            if shoulder_delta > 3:
                parts.append(
                    f"alineación de hombros {'+' if shoulder_delta > 0 else ''}{shoulder_delta}° al final "
                    f"(rotación de tronco degradada)"
                )
            narrative = f"⚠️ Fatiga detectada en {stroke}: {' | '.join(parts)}."

        result[stroke] = {
            "n_impacts":          len(frames),
            "fatigue_detected":   fatigue_detected,
            "knee_start_avg":     knee_start,
            "knee_end_avg":       knee_end,
            "knee_delta":         knee_delta,
            "elbow_start_avg":    elbow_start,
            "elbow_end_avg":      elbow_end,
            "elbow_delta":        elbow_delta,
            "shoulder_start_avg": shoulder_start,
            "shoulder_end_avg":   shoulder_end,
            "shoulder_delta":     shoulder_delta,
            "fatigue_detected":   fatigue_detected,
            "narrative":          narrative,
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 6 — compute_player_position_context
# ─────────────────────────────────────────────────────────────────────────────

def compute_player_position_context(yolo_data: dict) -> dict:
    """
    Extrae la posición promedio del jugador en cancha desde yolo_result.
    Versión liviana de compute_tactical_context() orientada a los especialistas.

    yolo_result["avg_player_position"] se persiste en Supabase (a diferencia
    de yolo_result["frames"] que se omite). Se puede leer directamente.

    Returns:
        {
            "avg_center_x":       float,   # 0-1, horizontal (0=izq, 1=der)
            "avg_center_y":       float,   # 0-1, vertical (0=arriba/red, 1=abajo/fondo)
            "dominant_position":  str,     # "baseline" | "net" | "mixed"
            "position_note":      str,     # texto listo para prompt del especialista
        }
    """
    avg_pos  = yolo_data.get("avg_player_position", {})
    center_x = avg_pos.get("center_x", 0.5)
    center_y = avg_pos.get("center_y", 0.5)

    NET_THRESHOLD      = 0.35
    BASELINE_THRESHOLD = 0.55

    if center_y < NET_THRESHOLD:
        dominant_position = "net"
        position_note = (
            f"Jugador predominantemente en zona de RED (center_y={center_y:.2f}). "
            "Los golpes se ejecutaron con menor tiempo de preparación y mayor urgencia. "
            "Flexión de rodillas puede ser menor por necesidad de reacción rápida — "
            "ser más tolerante con flexión insuficiente."
        )
    elif center_y > BASELINE_THRESHOLD:
        dominant_position = "baseline"
        position_note = (
            f"Jugador predominantemente en FONDO de cancha (center_y={center_y:.2f}). "
            "Tiempo de preparación estándar — aplicar criterios biomecánicos normales. "
            "Mayor flexión de rodillas esperada para generar potencia desde el fondo."
        )
    else:
        dominant_position = "mixed"
        position_note = (
            f"Jugador en POSICIÓN MIXTA (center_y={center_y:.2f}). "
            "Mezcla de golpes de fondo y transición — evaluar cada golpe en contexto."
        )

    return {
        "avg_center_x":      round(center_x, 3),
        "avg_center_y":      round(center_y, 3),
        "dominant_position": dominant_position,
        "position_note":     position_note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 7 — compute_phase_angles
# ─────────────────────────────────────────────────────────────────────────────
#
# ARQUITECTURA (v2 — Opción A):
#   La función ya NO depende de mediapipe_data["frames"] (omitido deliberadamente
#   en DB para mantener el payload < 1 MB).  Opera directamente sobre impact_frames,
#   que el vision pipeline guarda intactos en Supabase con:
#
#     impact_frame = {
#       "angles":             { "right_elbow": float, "right_knee": float, ... },
#       "shoulder_alignment": float,
#       "visibility":         float,
#       "stroke_type":        "forehand" | "backhand" | "saque" | ...,
#       "stroke_phases": {
#         "prep_frame":                int,
#         "prep_timestamp":            float,
#         "prep_angle_elbow":          float,   ← ángulo de codo en prep
#         "impact_angle_elbow":        float,
#         "followthrough_frame":       int,
#         "followthrough_timestamp":   float,
#         "followthrough_angle_elbow": float,   ← ángulo de codo en ft
#         "rom_degrees":               float,
#         "accel_frames":              int,
#       }
#     }
#
#   Qué se puede calcular con esta fuente:
#     ✅ impacto.dom_elbow / dom_knee / dom_hip / shoulder_alignment  (angles del impacto)
#     ✅ preparacion.dom_elbow   (stroke_phases.prep_angle_elbow)
#     ✅ preparacion.dom_hip     (stroke_phases.prep_angle_hip)
#     ✅ preparacion.shoulder_alignment (stroke_phases.prep_shoulder_alignment)
#     ✅ followthrough.dom_elbow (stroke_phases.followthrough_angle_elbow)
#     ✅ followthrough.dom_hip   (stroke_phases.ft_angle_hip)
#     ✅ followthrough.shoulder_alignment (stroke_phases.ft_shoulder_alignment)
#     ✅ delta_elbow_extension, delta_hip_rotation, delta_shoulder_rotation — todos disponibles
#     ⚠️  preparacion/ft.dom_knee → None (no guardado; valor diagnóstico bajo entre fases)
#
#   Resultado práctico: phase_angles deja de ser {} y provee deltas reales de
#   cadena cinética al LLM. delta_elbow_extension es el delta más diagnóstico
#   (aceleración del brazo), y ahora está disponible para todos los golpes.


def compute_phase_angles(
    impact_frames: list[dict],
    dominant_hand: str = "right",
    min_ball_speed_pct: float = 0.0,  # 0.0 = usar todos; 0.40 = filtrar golpes lentos
    # Parámetros legacy (ignorados, mantenidos para backward-compat)
    frames_by_stroke: dict | None = None,
    mediapipe_data:   dict | None = None,
) -> dict:
    """
    Calcula ángulos por fase biomecánica para cada tipo de golpe.

    Fuente de datos: `impact_frames` (lista de dicts producida por el vision
    pipeline y guardada íntegra en Supabase).  Ya NO usa mediapipe_data["frames"],
    que se omite de DB para mantener el payload < 1 MB.

    Por cada impact_frame con stroke_phases válido extrae:
      - fase "impacto"      → angles completo del frame de impacto
      - fase "preparacion"  → prep_angle_elbow del stroke_phases
      - fase "followthrough"→ followthrough_angle_elbow del stroke_phases

    Promedia los valores de todos los impactos del mismo stroke_type y calcula
    deltas de cadena cinética vía _compute_phase_deltas.

    ⚠️  ESTRATIFICACIÓN OPCIONAL POR VELOCIDAD DE PELOTA:
      - Si min_ball_speed_pct > 0.0, solo se usan impactos en el top
        (100% - min_ball_speed_pct) de ball_speed.
      - Esto filtra golpes lentos (taps, peloteos suaves) que contaminan
        los promedios de técnica con posiciones relajadas.
      - Default es 0.0 (usar todos los impactos) para máxima robustez
        cuando hay pocos datos.

    Args:
        impact_frames:      lista de dicts del vision pipeline (top_5 + worst_10 + mid_5)
        dominant_hand:      "right" | "left"
        min_ball_speed_pct: fracción del máximo de velocidad para filtrar
                           impactos lentos (0.0-1.0; default 0.0 = sin filtro)
        frames_by_stroke:   ignorado (legacy) — no borrar para no romper callers
        mediapipe_data:     ignorado (legacy) — no borrar para no romper callers

    Returns:
        {
            "forehand": {
                "phase_data_available": bool,
                "phases_computed":      list[str],
                "phases_insufficient":  list[str],
                "source":               "impact_frames",   # para debugging
                "n_impacts_used":       int,
                "n_impacts_total":      int,
                "speed_threshold_used": float | None,  # si se aplicó filtro
                "angles": {
                    "preparacion": {
                        "dom_elbow":          float | None,
                        "dom_knee":           None,   # no disponible en stroke_phases
                        "dom_hip":            None,
                        "shoulder_alignment": None,
                        "n_frames":           int,
                        "low_confidence":     bool,
                    },
                    "impacto": {
                        "dom_elbow":          float,
                        "dom_knee":           float | None,
                        "dom_hip":            float | None,
                        "shoulder_alignment": float | None,
                        "n_frames":           int,
                        "low_confidence":     bool,
                    },
                    "followthrough": { ... },
                },
                "deltas": { "delta_elbow_extension": float, ... },
                "fallback_note": str,
            },
            ...
        }
    """
    is_left   = dominant_hand == "left"
    elbow_key = "left_elbow"  if is_left else "right_elbow"
    knee_key  = "left_knee"   if is_left else "right_knee"
    hip_key   = "left_hip"    if is_left else "right_hip"

    # ── Normalizar stroke_type a las tres categorías canónicas ───────────────
    # El vision pipeline puede producir "forehand_o_backhand" cuando no hay
    # landmarks suficientes para distinguir — se descarta para no contaminar.
    _STROKE_CANONICAL = {
        "forehand":  "forehand",
        "backhand":  "backhand",
        "saque":     "saque",
        "saque_o_smash": "saque",   # smash se trata como saque biomecánicamente
    }

    # ── Agrupar impact_frames por stroke_type ────────────────────────────────
    by_stroke: dict[str, list[dict]] = {"forehand": [], "backhand": [], "saque": []}

    for f in (impact_frames or []):
        vis = f.get("visibility", 1.0) or 1.0
        if vis < 0.6:
            continue   # frame de baja calidad — descartar

        raw_type  = f.get("stroke_type") or ""
        canonical = _STROKE_CANONICAL.get(raw_type)
        if canonical is None:
            continue   # tipo no reconocido o ambiguo

        by_stroke[canonical].append(f)

    result = {}

    for stroke, frames in by_stroke.items():
        if not frames:
            continue

        # ── FILTRADO OPCIONAL POR VELOCIDAD DE PELOTA ──────────────────────────
        frames_to_use = frames
        speed_threshold_used = None
        
        if min_ball_speed_pct > 0.0:
            # Encontrar el máximo de ball_speed en este golpe
            ball_speeds = [
                f.get("ball_speed_pixels", 0) 
                for f in frames 
                if f.get("ball_speed_pixels") is not None
            ]
            if ball_speeds:
                max_speed = max(ball_speeds)
                speed_threshold = max_speed * min_ball_speed_pct
                frames_to_use = [
                    f for f in frames 
                    if f.get("ball_speed_pixels", 0) >= speed_threshold
                ]
                speed_threshold_used = round(speed_threshold, 1)
        
        n_impacts_total = len(frames)
        n_impacts_used = len(frames_to_use)
        
        if not frames_to_use:
            # Todos los frames fueron filtrados — usar el original
            frames_to_use = frames
            n_impacts_used = len(frames)

        # ── Acumular ángulos por fase ─────────────────────────────────────────
        # impacto: ángulos completos del frame de impacto
        imp_elbow:   list[float] = []
        imp_knee:    list[float] = []
        imp_hip:     list[float] = []
        imp_shoulder: list[float] = []

        # preparacion y followthrough: codo + hip + shoulder_alignment (desde stroke_phases)
        prep_elbow:   list[float] = []
        prep_hip:     list[float] = []
        prep_shoulder: list[float] = []
        ft_elbow:     list[float] = []
        ft_hip:       list[float] = []
        ft_shoulder:  list[float] = []

        for f in frames_to_use:
            angles = f.get("angles") or {}
            sp     = f.get("stroke_phases") or {}

            # ── Fase impacto ──────────────────────────────────────────────────
            e = angles.get(elbow_key)
            k = angles.get(knee_key)
            h = angles.get(hip_key)
            s = f.get("shoulder_alignment")

            if e is not None: imp_elbow.append(e)
            if k is not None: imp_knee.append(k)
            if h is not None: imp_hip.append(h)
            if s is not None: imp_shoulder.append(s)

            # ── Fase preparación ──────────────────────────────────────────────
            pe  = sp.get("prep_angle_elbow")
            ph  = sp.get("prep_angle_hip")
            ps  = sp.get("prep_shoulder_alignment")
            if pe is not None:
                prep_elbow.append(pe)
            if ph is not None:
                prep_hip.append(ph)
            if ps is not None:
                prep_shoulder.append(ps)

            # ── Fase follow-through ───────────────────────────────────────────
            fe  = sp.get("followthrough_angle_elbow")
            fh  = sp.get("ft_angle_hip")
            fs  = sp.get("ft_shoulder_alignment")
            if fe is not None:
                ft_elbow.append(fe)
            if fh is not None:
                ft_hip.append(fh)
            if fs is not None:
                ft_shoulder.append(fs)

        # ── Construir phase_angles dict ───────────────────────────────────────
        # Se necesita al menos 1 valor de codo en impacto para considerar el stroke.
        if not imp_elbow:
            continue

        MIN_IMPACTS = 1   # con 1 impacto ya es útil; el LLM interpreta con cautela

        phase_angles:    dict = {}
        phases_computed: list = []
        phases_insuf:    list = []

        # ── PREPARACIÓN ──────────────────────────────────────────────────────
        if prep_elbow:
            phase_angles["preparacion"] = {
                "dom_elbow":          round(_safe_avg(prep_elbow), 1),
                "dom_hip":            round(_safe_avg(prep_hip), 1)      if prep_hip      else None,
                "shoulder_alignment": round(_safe_avg(prep_shoulder), 2) if prep_shoulder else None,
                "dom_knee":           None,   # no guardado en stroke_phases
                "n_frames":           len(prep_elbow),
                "low_confidence":     len(prep_elbow) < 3,
            }
            phases_computed.append("preparacion")
        else:
            phases_insuf.append("preparacion")

        # ── IMPACTO ───────────────────────────────────────────────────────────
        # Datos más ricos: ángulos completos del frame de impacto real
        phase_angles["impacto"] = {
            "dom_elbow":          round(_safe_avg(imp_elbow), 1),
            "dom_knee":           round(_safe_avg(imp_knee),  1) if imp_knee    else None,
            "dom_hip":            round(_safe_avg(imp_hip),   1) if imp_hip     else None,
            "shoulder_alignment": round(_safe_avg(imp_shoulder), 2) if imp_shoulder else None,
            "n_frames":           len(imp_elbow),
            "low_confidence":     len(imp_elbow) < 3,
        }
        phases_computed.append("impacto")

        # ── FOLLOW-THROUGH ────────────────────────────────────────────────────
        if ft_elbow:
            phase_angles["followthrough"] = {
                "dom_elbow":          round(_safe_avg(ft_elbow), 1),
                "dom_hip":            round(_safe_avg(ft_hip), 1)      if ft_hip      else None,
                "shoulder_alignment": round(_safe_avg(ft_shoulder), 2) if ft_shoulder else None,
                "dom_knee":           None,   # no guardado en stroke_phases
                "n_frames":           len(ft_elbow),
                "low_confidence":     len(ft_elbow) < 3,
            }
            phases_computed.append("followthrough")
        else:
            phases_insuf.append("followthrough")

        # aceleracion no tiene ángulos propios en esta fuente — siempre insuf
        phases_insuf.append("aceleracion")

        # ── Disponibilidad ────────────────────────────────────────────────────
        # Mínimo: impacto siempre disponible.
        # "phase_data_available" = True si tenemos prep + impacto (deltas útiles)
        has_prep   = "preparacion"  in phases_computed
        has_impact = "impacto"      in phases_computed
        phase_data_available = has_prep and has_impact

        # ── Deltas ────────────────────────────────────────────────────────────
        deltas = _compute_phase_deltas(phase_angles, phases_computed)

        # ── Fallback note ─────────────────────────────────────────────────────
        if not phase_data_available:
            fallback_note = (
                "⚠️  ANÁLISIS DE FASES LIMITADO: solo datos de impacto disponibles. "
                "No se encontraron stroke_phases en los impact_frames de este golpe. "
                "Evalúa biomecánica usando directamente los ángulos de impacto."
            )
        else:
            codo_range = None
            if has_prep and "followthrough" in phases_computed:
                codo_range = round(
                    phase_angles["followthrough"]["dom_elbow"]
                    - phase_angles["preparacion"]["dom_elbow"],
                    1,
                )
            range_str  = f" | ROM codo: {codo_range}°" if codo_range is not None else ""
            insuf_str  = f" (fases sin datos: {', '.join(phases_insuf)})" if phases_insuf else ""
            fallback_note = (
                f"Análisis de fases disponible desde impact_frames: "
                f"{', '.join(phases_computed)}{insuf_str}{range_str}. "
                f"Fuente: stroke_phases del vision pipeline (prep+ft solo codo; impacto completo)."
            )

        result[stroke] = {
            "phase_data_available": phase_data_available,
            "phases_computed":      phases_computed,
            "phases_insufficient":  phases_insuf,
            "source":               "impact_frames",
            "n_impacts_used":       n_impacts_used,
            "n_impacts_total":      n_impacts_total,
            "speed_threshold_used": speed_threshold_used,
            "angles":               phase_angles,
            "deltas":               deltas,
            "fallback_note":        fallback_note,
        }

    return result


def _compute_phase_deltas(phase_angles: dict, phases_computed: list) -> dict:
    """
    Calcula deltas entre fases para evaluar la calidad de la cadena cinética.

    delta_shoulder_rotation: preparacion_hombros - impacto_hombros
        Hombros más cerrados en prep (alto) y abiertos en impacto (bajo) = delta alto = bueno.
        Un valor alto confirma que los hombros rotaron durante el swing.

    delta_hip_rotation: preparacion_cadera - impacto_cadera
        Cadera más cerrada en prep (ángulo menor = cargada) y girada en impacto.
        Un delta positivo alto indica que la cadera lideró el movimiento.

    delta_elbow_extension: followthrough_codo - preparacion_codo
        Codo más extendido en follow-through que en preparación = aceleración real.
        Un delta bajo indica brazo "armado" que no aceleró.
    """
    EMPTY = {
        "delta_shoulder_rotation": None,
        "delta_hip_rotation":      None,
        "delta_elbow_extension":   None,
        "rotation_quality":        "sin_datos",
        "kinetic_chain_note":      "Datos insuficientes para calcular deltas de cadena cinética.",
    }

    prep   = phase_angles.get("preparacion",   {})
    impact = phase_angles.get("impacto",        {})
    ft     = phase_angles.get("followthrough",  {})

    # Necesitamos al menos prep + impact para los deltas principales
    if not prep or not impact:
        return EMPTY

    # ── Delta hombros (rotación de tronco) ──────────────────────────────────
    prep_sh   = prep.get("shoulder_alignment")
    imp_sh    = impact.get("shoulder_alignment")
    delta_sh  = round(prep_sh - imp_sh, 1) if (prep_sh is not None and imp_sh is not None) else None

    # ── Delta cadera (rotación de cadera) ────────────────────────────────────
    # Cadera se "cierra" (ángulo menor) en preparación y se abre en impacto.
    # delta positivo = la cadera giró (correcto).
    prep_hip  = prep.get("dom_hip")
    imp_hip   = impact.get("dom_hip")
    delta_hip = round(prep_hip - imp_hip, 1) if (prep_hip is not None and imp_hip is not None) else None

    # ── Delta codo (aceleración del brazo) ───────────────────────────────────
    prep_el = prep.get("dom_elbow")
    ft_el   = ft.get("dom_elbow")
    delta_el = round(ft_el - prep_el, 1) if (ft_el is not None and prep_el is not None) else None

    # ── Clasificar calidad de rotación ───────────────────────────────────────
    # Usamos delta_hip como indicador principal de cadena cinética.
    if delta_hip is None:
        rotation_quality = "sin_datos"
    elif delta_hip >= 25:
        rotation_quality = "buena"
    elif delta_hip >= 10:
        rotation_quality = "moderada"
    else:
        rotation_quality = "pobre"

    # ── Narrativa para el prompt del especialista ────────────────────────────
    parts = []

    if delta_sh is not None:
        if delta_sh >= 5:
            parts.append(
                f"rotación de hombros confirmada (Δ={delta_sh:.1f}° — "
                f"hombros más cerrados en preparación, abiertos en impacto)"
            )
        else:
            parts.append(
                f"rotación de hombros limitada (Δ={delta_sh:.1f}° — "
                f"hombros casi paralelos en ambas fases, sin rotación real)"
            )

    if delta_hip is not None:
        if delta_hip >= 25:
            parts.append(
                f"cadera lideró el movimiento (Δ={delta_hip:.1f}° — "
                f"cadena cinética correcta: cadera rotó antes que el brazo)"
            )
        elif delta_hip >= 10:
            parts.append(
                f"cadera participó moderadamente (Δ={delta_hip:.1f}° — "
                f"hay rotación pero no lidera la cadena cinética completamente)"
            )
        else:
            parts.append(
                f"cadera casi no rotó (Δ={delta_hip:.1f}° — "
                f"golpe de brazo: la cadera no entregó energía al swing)"
            )

    if delta_el is not None:
        if delta_el >= 30:
            parts.append(
                f"aceleración del brazo completa (Δcodo={delta_el:.1f}° — "
                f"swing completo de preparación a follow-through)"
            )
        elif delta_el >= 10:
            parts.append(
                f"aceleración del brazo moderada (Δcodo={delta_el:.1f}°)"
            )
        else:
            parts.append(
                f"brazo sin aceleración notable (Δcodo={delta_el:.1f}° — "
                f"swing corto o impacto anticipado)"
            )

    kinetic_chain_note = (
        "CADENA CINÉTICA: " + " | ".join(parts)
        if parts else
        "CADENA CINÉTICA: datos insuficientes para evaluación completa."
    )

    return {
        "delta_shoulder_rotation": delta_sh,
        "delta_hip_rotation":      delta_hip,
        "delta_elbow_extension":   delta_el,
        "rotation_quality":        rotation_quality,
        "kinetic_chain_note":      kinetic_chain_note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NUEVA FUNCIÓN: extract_stroke_phases_summary
# ─────────────────────────────────────────────────────────────────────────────

def _classify_rom_quality(avg_rom: float, std_rom: float) -> str:
    """
    Clasifica ROM promedio en categoría biomecánica.
    
    < 30°   → corto (swing comprimido)
    30–50°  → normal (topspin regular)
    > 50°   → amplio (máxima potencia)
    """
    if avg_rom < 30:
        return "corto (swing comprimido — brazo no extiende completamente)"
    elif avg_rom <= 50:
        return "normal (rango óptimo para topspin — 30-50°)"
    else:
        return "amplio (máxima potencia — >50°, monitorear integridad de hombro)"


def _classify_accel_quality(avg_accel: float) -> str:
    """
    Clasifica aceleración (frames entre prep e impacto) en categoría.
    
    A 30fps:
      < 5 frames (~166 ms)  → rápido (whip explosivo)
      5–10 frames (166–333 ms) → normal (timing coordinado)
      > 10 frames (>333 ms) → lento (timing laxo)
    """
    if avg_accel < 5:
        return "rápido (whip explosivo — potencia bruta, <166 ms)"
    elif avg_accel <= 10:
        return "normal (timing coordinado — 166-333 ms, cadena cinética activa)"
    else:
        return "lento (timing laxo o desaceleración — >333 ms, pérdida de potencia)"


def extract_stroke_phases_summary(
    impact_frames: list,
    dominant_hand: str = "right",
) -> dict:
    """
    Extrae resumen de stroke_phases (ROM + aceleración) desde impact_frames.
    
    Agrupa por stroke_type y calcula estadísticas biomecánicas
    para inyectar en el prompt de los especialistas.
    
    Args:
        impact_frames: list[dict] — cada elemento tiene stroke_type + stroke_phases (opcional)
        dominant_hand: str — "right" | "left"
    
    Returns:
        dict con estructura:
        {
          "forehand": { n_frames, availability_pct, rom, accel, analysis },
          "backhand": { ... },
          "saque": { ... },
          "summary": { total_impacts, impacts_with_phases, phases_overall_availability, recommendation },
        }
    """
    if not impact_frames:
        return {
            "forehand": {},
            "backhand": {},
            "saque": {},
            "summary": {
                "total_impacts": 0,
                "impacts_with_phases": 0,
                "phases_overall_availability": 0.0,
                "recommendation": (
                    "Sin impact_frames disponibles — análisis de ROM/aceleración no posible. "
                    "Dependemos de ángulos absolutos de impacto."
                ),
            },
        }

    # Agrupar por stroke_type y extraer stroke_phases
    strokes_data = {
        "forehand": {"frames_with_phases": [], "frames_total": 0},
        "backhand": {"frames_with_phases": [], "frames_total": 0},
        "saque":    {"frames_with_phases": [], "frames_total": 0},
    }

    for impact in impact_frames:
        stroke = impact.get("stroke_type", "unknown")
        
        # Normalizar nombre del stroke
        if "forehand" in stroke.lower():
            stroke = "forehand"
        elif "backhand" in stroke.lower():
            stroke = "backhand"
        elif "saque" in stroke.lower() or "smash" in stroke.lower():
            stroke = "saque"
        else:
            continue
        
        if stroke in strokes_data:
            strokes_data[stroke]["frames_total"] += 1
            
            # Extraer stroke_phases si existen
            sp = impact.get("stroke_phases")
            if sp and isinstance(sp, dict):
                rom = sp.get("rom_degrees")
                accel = sp.get("accel_frames")
                
                # Validar datos numéricos y razonables
                if (rom is not None and isinstance(rom, (int, float)) and -180 < rom < 180 and
                    accel is not None and isinstance(accel, (int, float)) and 0 <= accel <= 30):
                    
                    strokes_data[stroke]["frames_with_phases"].append({
                        "rom": float(rom),
                        "accel": float(accel),
                    })

    # Calcular estadísticas por golpe
    result = {}
    total_impacts = 0
    total_with_phases = 0

    for stroke_type in ("forehand", "backhand", "saque"):
        data = strokes_data[stroke_type]
        n_total = data["frames_total"]
        n_with_phases = len(data["frames_with_phases"])
        
        total_impacts += n_total
        total_with_phases += n_with_phases
        
        if n_with_phases == 0:
            result[stroke_type] = {
                "n_frames": 0,
                "availability_pct": 0.0,
                "rom": {},
                "accel": {},
                "analysis": {
                    "rom_quality": "sin_datos",
                    "accel_quality": "sin_datos",
                    "biomechanics_note": (
                        f"Sin stroke_phases disponible para {stroke_type} "
                        f"({n_total} impactos detectados pero ninguno con datos de swing). "
                        f"Análisis limitado a ángulos de impacto bruto."
                    ),
                },
            }
            continue
        
        # Extraer listas
        rom_values = [f["rom"] for f in data["frames_with_phases"]]
        accel_values = [f["accel"] for f in data["frames_with_phases"]]
        
        # Estadísticas
        avg_rom = _safe_avg(rom_values)
        std_rom = _safe_std(rom_values)
        min_rom = round(min(rom_values), 1) if rom_values else 0.0
        max_rom = round(max(rom_values), 1) if rom_values else 0.0
        
        avg_accel = _safe_avg(accel_values)
        min_accel = int(min(accel_values)) if accel_values else 0
        max_accel = int(max(accel_values)) if accel_values else 0
        
        availability_pct = round(100 * n_with_phases / n_total, 1) if n_total > 0 else 0.0
        
        # Clasificar calidad
        rom_quality = _classify_rom_quality(avg_rom, std_rom)
        accel_quality = _classify_accel_quality(avg_accel)
        
        # Narrativa biomecánica
        consistency_note = (
            f"consistencia excelente (σ<5°)"
            if std_rom < 5 else
            f"cierta variabilidad (σ={std_rom}°)"
            if std_rom < 12 else
            f"alta variabilidad (σ={std_rom}°) — revisar timing"
        )
        
        biomechanics_note = (
            f"{stroke_type.upper()}: ROM promedio {avg_rom}° (rango {min_rom}°–{max_rom}°, {consistency_note}). "
            f"Swing ROM: {rom_quality}. "
            f"Aceleración: {accel_quality} (promedio {avg_accel:.1f} frames ≈ {avg_accel*33:.0f}ms). "
            f"Disponibilidad de datos: {availability_pct}% ({n_with_phases}/{n_total} impactos)."
        )
        
        result[stroke_type] = {
            "n_frames": n_with_phases,
            "availability_pct": availability_pct,
            "rom": {
                "avg_degrees": avg_rom,
                "std_degrees": std_rom,
                "min_degrees": min_rom,
                "max_degrees": max_rom,
            },
            "accel": {
                "avg_frames": avg_accel,
                "avg_milliseconds": round(avg_accel * 33.3, 0),
                "min_frames": min_accel,
                "max_frames": max_accel,
            },
            "analysis": {
                "rom_quality": rom_quality,
                "accel_quality": accel_quality,
                "biomechanics_note": biomechanics_note,
            },
        }

    # Resumen global
    overall_availability = (
        round(100 * total_with_phases / total_impacts, 1)
        if total_impacts > 0 else 0.0
    )

    if overall_availability == 0:
        recommendation = (
            "Sin stroke_phases disponible en ningún golpe. "
            "Análisis limitado a métricas de impacto bruto (ángulos absolutos de codo/cadera). "
            "Para ROM y timing completo, verificar quality_score de vision pipeline."
        )
    elif overall_availability < 50:
        recommendation = (
            f"Disponibilidad parcial de stroke_phases ({overall_availability}%). "
            f"ROM y timing disponibles solo para ~{overall_availability}% de los impactos. "
            f"Estadísticas sesgadas hacia impactos de alta velocidad."
        )
    else:
        recommendation = (
            f"Stroke_phases disponibles en {overall_availability}% de impactos. "
            f"ROM y timing del swing son métricas primarias confiables. "
            f"Usar para evaluar consistencia de swing y energía de aceleración."
        )

    result["summary"] = {
        "total_impacts": total_impacts,
        "impacts_with_phases": total_with_phases,
        "phases_overall_availability": overall_availability,
        "recommendation": recommendation,
    }

    return result
