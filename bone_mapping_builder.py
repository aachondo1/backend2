"""
TennisAI — Bone Mapping Builder v2
════════════════════════════════════
Novedades v2:
  1. correction_hints      — vector de corrección por joint crítico/warning.
  2. ideal_pose_overlay    — pose ATP canónica escalada por antropometría usuario.
  3. segment_lengths       — longitudes de huesos del usuario en session_meta.
"""

import math
from typing import Optional


def _get_phase_detector():
    try:
        from helpers import detect_stroke_phases
        return detect_stroke_phases
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════
# REFERENCIAS ATP
# ══════════════════════════════════════════════════════════════

# Referencias base para golpes sin grip específico detectado.
ATP_REFERENCE = {
    "forehand": {
        "right_elbow": {"ideal": 110, "range": (90,  130), "label": "Codo der. (impacto)"},
        "left_elbow":  {"ideal": 100, "range": (80,  125), "label": "Codo izq. (guía)"},
        "right_knee":  {"ideal": 140, "range": (120, 160), "label": "Rodilla der."},
        "left_knee":   {"ideal": 140, "range": (120, 160), "label": "Rodilla izq."},
        "right_hip":   {"ideal": 155, "range": (135, 170), "label": "Cadera der."},
        "left_hip":    {"ideal": 155, "range": (135, 170), "label": "Cadera izq."},
    },
    "backhand": {
        "right_elbow": {"ideal": 140, "range": (120, 165), "label": "Codo der. (golpe)"},
        "left_elbow":  {"ideal": 130, "range": (110, 155), "label": "Codo izq. (golpe)"},
        "right_knee":  {"ideal": 138, "range": (118, 158), "label": "Rodilla der."},
        "left_knee":   {"ideal": 138, "range": (118, 158), "label": "Rodilla izq."},
        "right_hip":   {"ideal": 150, "range": (130, 170), "label": "Cadera der."},
        "left_hip":    {"ideal": 150, "range": (130, 170), "label": "Cadera izq."},
    },
    "saque": {
        "right_elbow": {"ideal": 165, "range": (145, 180), "label": "Codo der. (extensión)"},
        "left_elbow":  {"ideal": 120, "range": (100, 145), "label": "Codo izq. (toss)"},
        "right_knee":  {"ideal": 145, "range": (125, 165), "label": "Rodilla der."},
        "left_knee":   {"ideal": 145, "range": (125, 165), "label": "Rodilla izq."},
        "right_hip":   {"ideal": 160, "range": (140, 175), "label": "Cadera der."},
        "left_hip":    {"ideal": 160, "range": (140, 175), "label": "Cadera izq."},
    },
}

# Referencias de forehand por grip detectado.
# El codo dominante tiene rangos distintos según cómo rota la muñeca en el grip:
#   eastern:      golpe plano, codo semi-cerrado (~80-112°)
#   semi_western: topspin moderado, rango intermedio (~100-138°)
#   western:      windshield wiper, codo abierto (~130-165°) — Djokovic/Nadal/Alcaraz
# Rodilla y cadera no varían significativamente por grip → se mantienen igual.
_ELBOW_DOM  = {"right": "right_elbow", "left": "left_elbow"}
_ELBOW_GUID = {"right": "left_elbow",  "left": "right_elbow"}

ATP_REFERENCE_FH_BY_GRIP = {
    "eastern": {
        "right_elbow": {"ideal":  96, "range": ( 80, 112), "label": "Codo der. — Eastern"},
        "left_elbow":  {"ideal":  96, "range": ( 80, 112), "label": "Codo izq. — Eastern"},
    },
    "semi_western": {
        "right_elbow": {"ideal": 119, "range": (100, 138), "label": "Codo der. — Semi-western"},
        "left_elbow":  {"ideal": 119, "range": (100, 138), "label": "Codo izq. — Semi-western"},
    },
    "western": {
        "right_elbow": {"ideal": 148, "range": (130, 165), "label": "Codo der. — Western"},
        "left_elbow":  {"ideal": 148, "range": (130, 165), "label": "Codo izq. — Western"},
    },
}

# Referencias de backhand por variante técnica (topspin / slice) y grip (one/two-handed).
ATP_REFERENCE_BH_BY_VARIANT = {
    "two_handed": {
        "topspin": {
            "right_elbow": {"ideal": 107, "range": ( 90, 125), "label": "Codo der. — BH 2M topspin"},
            "left_elbow":  {"ideal": 107, "range": ( 90, 125), "label": "Codo izq. — BH 2M topspin"},
        },
        "slice": {
            "right_elbow": {"ideal": 137, "range": (120, 155), "label": "Codo der. — BH 2M slice"},
            "left_elbow":  {"ideal": 137, "range": (120, 155), "label": "Codo izq. — BH 2M slice"},
        },
    },
    "one_handed": {
        "topspin": {
            "right_elbow": {"ideal": 157, "range": (145, 170), "label": "Codo der. — BH 1M topspin"},
            "left_elbow":  {"ideal": 157, "range": (145, 170), "label": "Codo izq. — BH 1M topspin"},
        },
        "slice": {
            "right_elbow": {"ideal": 147, "range": (130, 165), "label": "Codo der. — BH 1M slice"},
            "left_elbow":  {"ideal": 147, "range": (130, 165), "label": "Codo izq. — BH 1M slice"},
        },
    },
}


def _get_reference_for_stroke(stroke: str, grip_type: str = "unknown",
                               bh_variant: str = "topspin") -> dict:
    """
    Retorna el dict de referencia articular correcto según el golpe y grip detectado.

    Para forehand: mezcla la referencia base (rodilla, cadera, codo guía)
    con los rangos de codo dominante ajustados al grip.

    Para backhand: mezcla la referencia base con los rangos de codo
    ajustados a la variante (topspin/slice) y grip (one/two-handed).

    Para saque: devuelve ATP_REFERENCE["saque"] sin modificaciones.

    Siempre hace fallback seguro — si el grip no se reconoce usa la base.
    """
    base = dict(ATP_REFERENCE.get(stroke, ATP_REFERENCE["forehand"]))

    if stroke == "forehand":
        grip_refs = ATP_REFERENCE_FH_BY_GRIP.get(grip_type)
        if grip_refs:
            # Sobreescribir solo las entradas de codo con los rangos del grip
            base = {**base, **grip_refs}
        return base

    if stroke == "backhand":
        grip_key    = grip_type if grip_type in ("one_handed", "two_handed") else None
        variant_key = bh_variant if bh_variant in ("topspin", "slice") else "topspin"
        if grip_key:
            bh_refs = (ATP_REFERENCE_BH_BY_VARIANT
                       .get(grip_key, {})
                       .get(variant_key))
            if bh_refs:
                base = {**base, **bh_refs}
        return base

    return base

_LEFT_SWAP = {
    "right_elbow": "left_elbow", "left_elbow": "right_elbow",
    "right_hip":   "left_hip",   "left_hip":   "right_hip",
}

_WARNING_PCT  = 15
_CRITICAL_PCT = 30

# ══════════════════════════════════════════════════════════════
# POSES ATP CANÓNICAS  (33 landmarks, coordenadas normalizadas)
# ══════════════════════════════════════════════════════════════

def _atp_pose_forehand():
    return [
        [0.50,0.07,0.0,1.0],[0.52,0.06,0.0,1.0],[0.54,0.06,0.0,1.0],[0.56,0.06,0.0,1.0],
        [0.48,0.06,0.0,1.0],[0.46,0.06,0.0,1.0],[0.44,0.06,0.0,1.0],[0.57,0.09,0.0,1.0],
        [0.43,0.09,0.0,1.0],[0.51,0.11,0.0,1.0],[0.49,0.11,0.0,1.0],
        [0.62,0.22,0.0,1.0],[0.38,0.22,0.0,1.0],  # 11 hombro_izq  12 hombro_der
        [0.70,0.36,0.0,1.0],[0.30,0.28,0.0,1.0],  # 13 codo_izq    14 codo_der (110°)
        [0.75,0.46,0.0,1.0],[0.24,0.38,0.0,1.0],  # 15 muñeca_izq  16 muñeca_der
        [0.76,0.48,0.0,1.0],[0.23,0.39,0.0,1.0],[0.76,0.47,0.0,1.0],[0.23,0.38,0.0,1.0],
        [0.77,0.46,0.0,1.0],[0.22,0.37,0.0,1.0],
        [0.57,0.52,0.0,1.0],[0.43,0.52,0.0,1.0],  # 23 cadera_izq  24 cadera_der
        [0.59,0.67,0.0,1.0],[0.41,0.67,0.0,1.0],  # 25 rodilla_izq 26 rodilla_der (140°)
        [0.60,0.82,0.0,1.0],[0.40,0.82,0.0,1.0],
        [0.61,0.85,0.0,1.0],[0.39,0.85,0.0,1.0],[0.62,0.87,0.0,1.0],[0.38,0.87,0.0,1.0],
    ]

def _atp_pose_backhand():
    return [
        [0.50,0.07,0.0,1.0],[0.52,0.06,0.0,1.0],[0.54,0.06,0.0,1.0],[0.56,0.06,0.0,1.0],
        [0.48,0.06,0.0,1.0],[0.46,0.06,0.0,1.0],[0.44,0.06,0.0,1.0],[0.57,0.09,0.0,1.0],
        [0.43,0.09,0.0,1.0],[0.51,0.11,0.0,1.0],[0.49,0.11,0.0,1.0],
        [0.38,0.22,0.0,1.0],[0.62,0.22,0.0,1.0],  # 11 hombro_izq  12 hombro_der (rotado)
        [0.28,0.33,0.0,1.0],[0.68,0.33,0.0,1.0],  # 13 codo_izq (130°) 14 codo_der (140°)
        [0.22,0.42,0.0,1.0],[0.74,0.43,0.0,1.0],
        [0.21,0.43,0.0,1.0],[0.75,0.44,0.0,1.0],[0.21,0.42,0.0,1.0],[0.75,0.43,0.0,1.0],
        [0.20,0.41,0.0,1.0],[0.76,0.42,0.0,1.0],
        [0.43,0.52,0.0,1.0],[0.57,0.52,0.0,1.0],  # 23 cadera_izq  24 cadera_der
        [0.41,0.67,0.0,1.0],[0.59,0.67,0.0,1.0],  # 25 rodilla_izq 26 rodilla_der (138°)
        [0.40,0.82,0.0,1.0],[0.60,0.82,0.0,1.0],
        [0.39,0.85,0.0,1.0],[0.61,0.85,0.0,1.0],[0.38,0.87,0.0,1.0],[0.62,0.87,0.0,1.0],
    ]

def _atp_pose_saque():
    return [
        [0.50,0.05,0.0,1.0],[0.52,0.04,0.0,1.0],[0.54,0.04,0.0,1.0],[0.56,0.04,0.0,1.0],
        [0.48,0.04,0.0,1.0],[0.46,0.04,0.0,1.0],[0.44,0.04,0.0,1.0],[0.57,0.07,0.0,1.0],
        [0.43,0.07,0.0,1.0],[0.51,0.09,0.0,1.0],[0.49,0.09,0.0,1.0],
        [0.65,0.20,0.0,1.0],[0.38,0.22,0.0,1.0],  # 11 hombro_izq (toss) 12 hombro_der
        [0.72,0.10,0.0,1.0],[0.28,0.08,0.0,1.0],  # 13 codo_izq (120°)  14 codo_der (165°)
        [0.74,0.04,0.0,1.0],[0.22,0.02,0.0,1.0],  # 15 muñeca_izq       16 muñeca_der (arriba)
        [0.75,0.03,0.0,1.0],[0.21,0.01,0.0,1.0],[0.75,0.03,0.0,1.0],[0.21,0.01,0.0,1.0],
        [0.76,0.02,0.0,1.0],[0.20,0.01,0.0,1.0],
        [0.57,0.52,0.0,1.0],[0.43,0.52,0.0,1.0],  # 23 cadera_izq  24 cadera_der
        [0.58,0.67,0.0,1.0],[0.42,0.67,0.0,1.0],  # 25 rodilla_izq 26 rodilla_der (145°)
        [0.59,0.82,0.0,1.0],[0.41,0.82,0.0,1.0],
        [0.60,0.85,0.0,1.0],[0.40,0.85,0.0,1.0],[0.61,0.87,0.0,1.0],[0.39,0.87,0.0,1.0],
    ]

ATP_POSES = {
    "forehand": _atp_pose_forehand(),
    "backhand": _atp_pose_backhand(),
    "saque":    _atp_pose_saque(),
}

# ══════════════════════════════════════════════════════════════
# HELPERS DE ANÁLISIS
# ══════════════════════════════════════════════════════════════

def _deviation_pct(user_angle, ideal, range_):
    lo, hi     = range_
    half_width = (hi - lo) / 2.0
    if half_width == 0:
        return 0.0
    center  = (lo + hi) / 2.0
    raw_dev = abs(user_angle - center) - half_width
    if raw_dev <= 0:
        return 0.0
    return round(min(raw_dev / half_width * 100, 100), 1)

def _status_from_dev(dev_pct):
    if dev_pct >= _CRITICAL_PCT: return "critical"
    if dev_pct >= _WARNING_PCT:  return "warning"
    return "normal"

def _build_analysis_delta(angles, stroke, dominant_hand,
                          grip_type="unknown", bh_variant="topspin"):
    ref      = _get_reference_for_stroke(stroke, grip_type, bh_variant)
    is_lefty = dominant_hand == "left"
    result   = []
    for joint_key, meta in ref.items():
        lookup_key = _LEFT_SWAP.get(joint_key, joint_key) if is_lefty else joint_key
        user_angle = angles.get(lookup_key)
        if user_angle is None:
            continue
        dev_pct = _deviation_pct(user_angle, meta["ideal"], meta["range"])
        result.append({
            "joint":         joint_key,
            "label":         meta["label"],
            "user_angle":    round(user_angle, 1),
            "ideal_angle":   meta["ideal"],
            "deviation_pct": dev_pct,
            "status":        _status_from_dev(dev_pct),
        })
    result.sort(key=lambda x: x["deviation_pct"], reverse=True)
    return result

def _score_from_delta(delta):
    if not delta:
        return 50
    weights = {"right_elbow":2.0,"left_elbow":2.0,"right_knee":1.5,
               "left_knee":1.5,"right_hip":1.2,"left_hip":1.2}
    total_w = total_pen = 0.0
    for e in delta:
        w          = weights.get(e["joint"], 1.0)
        total_w   += w
        pf         = {"normal":0.0,"warning":0.5,"critical":1.0}[e["status"]]
        total_pen += w * pf * (e["deviation_pct"] / 100.0)
    penalty = (total_pen / total_w) * 100 if total_w > 0 else 0
    return max(0, min(100, round(100 - penalty * 1.5)))

# ══════════════════════════════════════════════════════════════
# HELPERS DE LANDMARKS
# ══════════════════════════════════════════════════════════════

def _landmarks_to_pose(landmarks_3d):
    if not landmarks_3d or len(landmarks_3d) < 33:
        return []
    return [
        [round(lm.get("x",0.0),4), round(lm.get("y",0.0),4),
         round(lm.get("z",0.0),4), round(lm.get("visibility",0.0),3)]
        for lm in landmarks_3d[:33]
    ]

def _average_landmarks(frames_with_lm):
    valid = [f for f in frames_with_lm if f.get("landmarks_3d") and len(f["landmarks_3d"])==33]
    if not valid:
        return []
    n    = len(valid)
    pose = []
    for i in range(33):
        xs  = [f["landmarks_3d"][i]["x"]          for f in valid]
        ys  = [f["landmarks_3d"][i]["y"]          for f in valid]
        zs  = [f["landmarks_3d"][i]["z"]          for f in valid]
        vis = [f["landmarks_3d"][i]["visibility"] for f in valid]
        pose.append([round(sum(xs)/n,4), round(sum(ys)/n,4),
                     round(sum(zs)/n,4), round(sum(vis)/n,3)])
    return pose

def _average_angles(frames):
    keys   = ["right_elbow","left_elbow","right_knee","left_knee","right_hip","left_hip"]
    result = {}
    for k in keys:
        vals = [f["angles"].get(k) for f in frames if f.get("angles",{}).get(k) is not None]
        if vals:
            result[k] = round(sum(vals)/len(vals), 1)
    return result

def _quality_score(impacts):
    if not impacts:
        return 0.0
    vis_vals = [f.get("visibility",0) for f in impacts if f.get("visibility") is not None]
    avg_vis  = sum(vis_vals)/len(vis_vals) if vis_vals else 0.5
    has_lm   = sum(1 for f in impacts if f.get("landmarks_3d")) / max(len(impacts),1)
    count_ok = min(len(impacts)/10.0, 1.0)
    return round(avg_vis*0.5 + has_lm*0.3 + count_ok*0.2, 3)

# ══════════════════════════════════════════════════════════════
# NORMALIZACIÓN ANTROPOMÉTRICA
# ══════════════════════════════════════════════════════════════

_BONE_SEGMENTS = [
    (11,13,"upper_arm_left"),(12,14,"upper_arm_right"),
    (13,15,"forearm_left"),  (14,16,"forearm_right"),
    (11,23,"torso_left"),    (12,24,"torso_right"),
    (23,25,"thigh_left"),    (24,26,"thigh_right"),
    (25,27,"shin_left"),     (26,28,"shin_right"),
]

def _dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def _compute_segment_lengths(pose):
    if len(pose) < 33:
        return {}
    return {name: round(_dist(pose[a], pose[b]), 4) for a,b,name in _BONE_SEGMENTS}

def _scale_atp_pose_to_user(atp_pose, user_pose):
    """
    Escala la pose ATP a las proporciones del usuario:
    1. Calcula ratio length_user / length_atp por segmento.
    2. Promedia los ratios aplicables a cada landmark.
    3. Re-ancla al centro de cadera del usuario.
    """
    if len(user_pose) < 33 or len(atp_pose) < 33:
        return atp_pose

    user_seg = _compute_segment_lengths(user_pose)
    atp_seg  = _compute_segment_lengths(atp_pose)

    ratios = {}
    for _,_,name in _BONE_SEGMENTS:
        u = user_seg.get(name, 0)
        a = atp_seg.get(name, 0)
        if a > 0 and u > 0:
            ratios[name] = u / a

    if not ratios:
        return atp_pose

    global_ratio = sum(ratios.values()) / len(ratios)

    # Centro de cadera
    user_hip_x = (user_pose[23][0] + user_pose[24][0]) / 2
    user_hip_y = (user_pose[23][1] + user_pose[24][1]) / 2
    atp_hip_x  = (atp_pose[23][0]  + atp_pose[24][0])  / 2
    atp_hip_y  = (atp_pose[23][1]  + atp_pose[24][1])  / 2

    # Ratios por punto (promedio de segmentos adyacentes)
    point_ratios = {i: [] for i in range(33)}
    for a,b,name in _BONE_SEGMENTS:
        r = ratios.get(name, global_ratio)
        point_ratios[a].append(r)
        point_ratios[b].append(r)

    scaled = []
    for i, pt in enumerate(atp_pose):
        r_list = point_ratios[i]
        r      = sum(r_list)/len(r_list) if r_list else global_ratio
        dx     = (pt[0] - atp_hip_x) * r
        dy     = (pt[1] - atp_hip_y) * r
        scaled.append([round(user_hip_x+dx,4), round(user_hip_y+dy,4), round(pt[2],4), pt[3]])

    return scaled

# ══════════════════════════════════════════════════════════════
# CORRECTION HINTS
# ══════════════════════════════════════════════════════════════

def _build_correction_hints(delta, stroke="forehand", grip_type="unknown"):
    """
    Genera hints accionables para joints warning/critical.
    vector_correction > 0  → debe extender
    vector_correction < 0  → debe flexionar

    Vocabulario grip-aware (Punto 3):
    - Western / one-handed: un codo extendido es virtud, no error.
      Se usa "palanca larga" / "extensión moderna" en lugar de "rígido" / "flexiona".
    - Eastern / semi-western / two-handed: vocabulario técnico estándar.
    """
    # Joints de codo que reciben vocabulario especial en grips de extensión
    _EXTENSION_GRIPS  = {"western", "one_handed"}
    _ELBOW_JOINTS     = {"right_elbow", "left_elbow"}

    is_extension_grip = grip_type in _EXTENSION_GRIPS

    hints = []
    for e in delta:
        if e["status"] == "normal":
            continue

        vector    = round(e["ideal_angle"] - e["user_angle"], 1)
        direction = "extender" if vector > 0 else "flexionar"
        abs_vec   = abs(vector)
        magnitude = "ligeramente" if abs_vec < 5 else ("moderadamente" if abs_vec < 15 else "significativamente")

        # ── Vocabulario adaptado al grip ──────────────────────
        joint_is_elbow = e["joint"] in _ELBOW_JOINTS

        if joint_is_elbow and is_extension_grip and direction == "flexionar":
            # Codo extendido en western/one-handed = técnica correcta para este grip.
            # No decir "flexiona" — elogiar y explicar como virtud biomecánica.
            hint_text = (
                f"{e['label']}: {e['user_angle']:.0f}° — "
                f"palanca larga característica del {grip_type.replace('_',' ')}. "
                f"La extensión del brazo genera mayor aceleración de cabeza de raqueta. "
                f"Solo revisar si hay bloqueo de muñeca o rigidez muscular."
            )
        elif joint_is_elbow and is_extension_grip and direction == "extender":
            # Codo más cerrado de lo esperado para un grip de extensión
            action = "Extiende"
            hint_text = (
                f"{action} {magnitude} el {e['label']} — "
                f"el {grip_type.replace('_',' ')} requiere mayor extensión (~{abs_vec:.0f}° más) "
                f"para activar el windshield wiper y generar topspin efectivo."
            )
        else:
            # Vocabulario estándar para todos los demás casos
            action = "Extiende" if direction == "extender" else "Flexiona"
            hint_text = f"{action} {magnitude} el {e['label']} — faltan ~{abs_vec:.0f}° para el rango óptimo"

        hints.append({
            "joint":             e["joint"],
            "label":             e["label"],
            "status":            e["status"],
            "user_angle":        e["user_angle"],
            "ideal_angle":       e["ideal_angle"],
            "deviation_pct":     e["deviation_pct"],
            "vector_correction": vector,
            "direction":         direction,
            "hint_text":         hint_text,
        })
    hints.sort(key=lambda h: (0 if h["status"]=="critical" else 1, -h["deviation_pct"]))
    return hints

# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═
# PHASE-AWARE FRAME EXTRACTION
# ═

_WINDOW_BEFORE = 12
_WINDOW_AFTER  = 8


def _get_phase_aware_frame(impact, all_mp_frames, elbow_key):
    """Refina un impacto usando detect_stroke_phases.
    Devuelve el frame del pico de aceleracion angular (impacto real).
    Fallback transparente: si no hay frames suficientes devuelve impact sin cambios.
    """
    detect_fn = _get_phase_detector()
    if detect_fn is None or not all_mp_frames:
        return impact

    ref_frame_idx = impact.get("mediapipe_frame")
    if ref_frame_idx is None:
        return impact

    window = sorted(
        [f for f in all_mp_frames
         if ref_frame_idx - _WINDOW_BEFORE <= f["frame"] <= ref_frame_idx + _WINDOW_AFTER],
        key=lambda f: f["frame"],
    )
    if len(window) < 4:
        return impact

    phases = detect_fn(window, elbow_key=elbow_key)
    if phases is None:
        return impact

    real = phases["impacto"]
    return {
        **impact,
        "angles":              real["angles"],
        "landmarks_3d":        real.get("landmarks_3d"),
        "mediapipe_timestamp": real["timestamp"],
        "_phases": {
            "preparacion_ts":    phases["preparacion"]["timestamp"],
            "carga_ts":          phases["carga"]["timestamp"],
            "impacto_ts":        real["timestamp"],
            "follow_through_ts": phases["follow_through"]["timestamp"],
            "elbow_at_impact":   real["elbow_angle"],
            "elbow_at_carga":    phases["carga"]["elbow_angle"],
            "total_frames":      phases["total_frames"],
        },
    }


# ══════════════════════════════════════════════════════════════
# NORMALIZACIÓN DE STROKE TYPE
# ══════════════════════════════════════════════════════════════

_STROKE_NORM = {
    "forehand":"forehand","backhand":"backhand","saque":"saque",
    "saque_o_smash":"saque","forehand_o_backhand":"forehand",
}

def _normalize_stroke(raw):
    if not raw: return None
    return _STROKE_NORM.get(raw.lower(), raw.lower())

# ══════════════════════════════════════════════════════════════
# BUILDER DE MODO
# ══════════════════════════════════════════════════════════════

def _build_mode(pose, angles, stroke, dominant_hand, label, extra=None,
                grip_type="unknown", bh_variant="topspin"):
    """Construye un dict de modo completo con todos los campos v2."""
    delta   = _build_analysis_delta(angles, stroke, dominant_hand, grip_type, bh_variant)
    score   = _score_from_delta(delta)
    hints   = _build_correction_hints(delta, stroke=stroke, grip_type=grip_type)
    atp_raw = ATP_POSES.get(stroke, ATP_POSES["forehand"])
    overlay = _scale_atp_pose_to_user(atp_raw, pose) if pose else atp_raw

    mode = {
        "pose":               pose,
        "score":              score,
        "analysis_delta":     delta,
        "correction_hints":   hints,
        "ideal_pose_overlay": overlay,
        "label":              label,
    }
    if extra:
        mode.update(extra)
    return mode

# ══════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════

def generate_bone_mapping_input(
    impact_frames,
    mediapipe_result,
    dominant_hand    = "right",
    active_strokes   = None,
    forehand_grip    = None,
    backhand_grip    = None,
):
    """
    Genera el JSON completo de bone mapping v2 para todos los golpes activos.
    Retorna Dict { stroke_type: { session_meta, modes, timeline } }

    Args:
        impact_frames:   lista de impact_frames con stroke_type + angles + landmarks_3d
        mediapipe_result: dict con "frames" de MediaPipe
        dominant_hand:   "right" | "left"
        active_strokes:  lista de golpes a incluir (None = todos)
        forehand_grip:   dict de coordinator_data["forehand_grip"]
                         — {"grip": "western"|"semi_western"|"eastern"|"unknown", ...}
        backhand_grip:   dict de coordinator_data["backhand_grip"]
                         — {"grip": "two_handed"|"one_handed"|"unknown",
                            "bh_variant": "topspin"|"slice", ...}
    """
    # ── Extraer grip_type y bh_variant de los dicts del coordinador ──────────
    fh_grip_type = (forehand_grip or {}).get("grip", "unknown")
    bh_grip_type = (backhand_grip or {}).get("grip", "unknown")
    bh_variant   = (backhand_grip or {}).get("bh_variant", "topspin")

    by_stroke = {}
    for imp in impact_frames:
        stroke = _normalize_stroke(imp.get("stroke_type"))
        if stroke is None:
            continue
        by_stroke.setdefault(stroke, []).append(imp)

    if not by_stroke:
        all_as_forehand = [f for f in impact_frames if f.get("angles")]
        if all_as_forehand:
            by_stroke["forehand"] = all_as_forehand

    if active_strokes:
        normalized_active = [_normalize_stroke(s) for s in active_strokes]
        by_stroke = {k: v for k,v in by_stroke.items() if k in normalized_active}

    all_mp_frames = mediapipe_result.get("frames", []) if mediapipe_result else []

    result = {}

    for stroke, impacts in by_stroke.items():
        if not impacts:
            continue

        # Seleccionar grip correcto según el golpe
        grip_type  = fh_grip_type if stroke == "forehand" else bh_grip_type
        bh_var_use = bh_variant   if stroke == "backhand"  else "topspin"

        elbow_key = "left_elbow" if dominant_hand == "left" else "right_elbow"

        refined_impacts = [
            _get_phase_aware_frame(imp, all_mp_frames, elbow_key)
            for imp in impacts
        ]

        # Score individual por impacto (sobre angulos refinados)
        scored = []
        for imp in refined_impacts:
            angles = imp.get("angles", {})
            delta  = _build_analysis_delta(angles, stroke, dominant_hand, grip_type, bh_var_use)
            scored.append({**imp, "_delta": delta, "_score": _score_from_delta(delta)})

        scored_sorted   = sorted(scored, key=lambda x: x["_score"])
        sorted_by_speed = sorted(refined_impacts, key=lambda f: f.get("ball_speed_pixels",0), reverse=True)

        # Timeline
        timeline = sorted(
            [{"timestamp": round(s.get("impact_timestamp", s.get("mediapipe_timestamp",0)),2),
              "score": s["_score"]} for s in scored],
            key=lambda x: x["timestamp"],
        )

        # segment_lengths desde el mejor frame con landmarks
        ref_lm = next((f.get("landmarks_3d") for f in sorted_by_speed if f.get("landmarks_3d")), None)
        ref_pose = _landmarks_to_pose(ref_lm) if ref_lm else []
        segment_lengths = _compute_segment_lengths(ref_pose)

        # Modo REPRESENTATIVE
        rep_pool = sorted_by_speed[:5]
        mode_representative = _build_mode(
            pose=_average_landmarks(rep_pool),
            angles=_average_angles(rep_pool),
            stroke=stroke, dominant_hand=dominant_hand,
            label=f"Promedio top {len(rep_pool)} impactos",
            extra={"averaged_from": len(rep_pool)},
            grip_type=grip_type, bh_variant=bh_var_use,
        )

        # Modo BEST
        best = scored_sorted[-1]
        mode_best = _build_mode(
            pose=_landmarks_to_pose(best.get("landmarks_3d",[])),
            angles=best.get("angles",{}),
            stroke=stroke, dominant_hand=dominant_hand,
            label="Mejor golpe de la sesión",
            extra={"timestamp": round(best.get("impact_timestamp", best.get("mediapipe_timestamp",0)),2)},
            grip_type=grip_type, bh_variant=bh_var_use,
        )

        # Modo WORST
        worst = scored_sorted[0]
        mode_worst = _build_mode(
            pose=_landmarks_to_pose(worst.get("landmarks_3d",[])),
            angles=worst.get("angles",{}),
            stroke=stroke, dominant_hand=dominant_hand,
            label="Golpe con mayor desviación",
            extra={"timestamp": round(worst.get("impact_timestamp", worst.get("mediapipe_timestamp",0)),2)},
            grip_type=grip_type, bh_variant=bh_var_use,
        )

        result[stroke] = {
            "session_meta": {
                "stroke_type":     stroke,
                "dominant_hand":   dominant_hand,
                "total_impacts":     len(impacts),
                "impacts_with_pose": sum(1 for f in refined_impacts if f.get("landmarks_3d")),
                "quality_score":     _quality_score(impacts),
                "has_landmarks":     any(f.get("landmarks_3d") for f in refined_impacts),
                "segment_lengths": segment_lengths,
                "phase_refined":   any(f.get("_phases") for f in refined_impacts),
                "grip_type":       grip_type,       # ← nuevo: para el Digital Twin
                "bh_variant":      bh_var_use,      # ← nuevo: para el Digital Twin
            },
            "modes": {
                "representative": mode_representative,
                "best":           mode_best,
                "worst":          mode_worst,
            },
            "timeline": timeline,
        }

    return result


# ══════════════════════════════════════════════════════════════
# TESTING LOCAL
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json, sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test_vision_row.json"
    try:
        with open(path) as f:
            row = json.load(f)
    except FileNotFoundError:
        print(f"❌ No se encontró {path}"); sys.exit(1)

    impact_frames    = row.get("impact_frames", row.get("impacts", []))
    mediapipe_result = row.get("mediapipe_result", row.get("mediapipe", {}))
    dominant_hand    = row.get("dominant_hand", "right")
    if isinstance(impact_frames, str):    impact_frames    = json.loads(impact_frames)
    if isinstance(mediapipe_result, str): mediapipe_result = json.loads(mediapipe_result)

    bm = generate_bone_mapping_input(impact_frames, mediapipe_result, dominant_hand)

    print(f"\n✅ Bone mapping v2 — {len(bm)} golpe(s): {list(bm.keys())}\n")
    for stroke, data in bm.items():
        meta  = data["session_meta"]
        modes = data["modes"]
        print(f"  [{stroke.upper()}]")
        print(f"    impactos:        {meta['total_impacts']}")
        print(f"    quality:         {meta['quality_score']}")
        print(f"    segment_lengths: {meta['segment_lengths']}")
        for mode_name, mode in modes.items():
            n_pose    = len(mode.get("pose") or [])
            n_overlay = len(mode.get("ideal_pose_overlay") or [])
            n_hints   = len(mode.get("correction_hints") or [])
            n_crit    = sum(1 for d in mode["analysis_delta"] if d["status"]=="critical")
            print(f"    [{mode_name}] score={mode['score']} | pose={n_pose}pts | "
                  f"overlay={n_overlay}pts | hints={n_hints} | critical={n_crit}")
            for hint in mode.get("correction_hints",[]):
                icon = "🔴" if hint["status"]=="critical" else "🟡"
                print(f"      {icon} {hint['hint_text']}")
        print()

    with open("bone_mapping_output_v2.json","w") as f:
        json.dump(bm, f, indent=2, ensure_ascii=False)
    print("💾 Guardado en bone_mapping_output_v2.json")
