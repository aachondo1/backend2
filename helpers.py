"""
TennisAI — Helpers compartidos
──────────────────────────────
Funciones de utilidad usadas por todos los agentes y el orquestador.
Este módulo es Python puro — sin Modal, sin dependencias externas.
Testeable localmente con cualquier JSON de prueba.

Contenido:
  - Supabase: supabase_patch, supabase_post
  - Contexto: format_camera_context, format_equipment_context,
              format_session_context
  - Frames:   get_stroke_frames_or_fallback, extract_peak_frames
  - Parsing:  parse_json_response, assign_level_from_score
"""

import json
import re


# ══════════════════════════════════════════════════════════════
# HELPERS SUPABASE
# ══════════════════════════════════════════════════════════════

def supabase_patch(url: str, key: str, table: str, record_id: str, data: dict) -> bool:
    """PATCH parcial a un registro Supabase. Retorna True si exitoso."""
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
    """POST a Supabase. Retorna el registro creado o None si falla."""
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


# ══════════════════════════════════════════════════════════════
# HELPERS DE CONTEXTO
# ══════════════════════════════════════════════════════════════

def format_camera_context(camera_orientation: str | None) -> str:
    """
    Traduce camera_orientation (ej: "Fondo Trasero-Centro") a instrucciones
    biomecánicas concretas para los agentes.

    Orígenes posibles : Red | Fondo Trasero | Fondo Frontal | Lateral
    Posiciones posibles: Izquierda | Centro | Derecha del jugador
    """
    if not camera_orientation:
        return "ÁNGULO DE CÁMARA: No especificado — asumir ambigüedad lateral estándar."

    parts    = camera_orientation.split("-", 1)
    origin   = parts[0].strip() if len(parts) > 0 else ""
    position = parts[1].strip() if len(parts) > 1 else "Centro"

    origin_hints = {
        "Red": (
            "vista FRONTAL (desde la red) — el jugador se mueve hacia/lejos de cámara. "
            "Visibilidad alta de alineación de hombros y caderas. "
            "Ángulos de codo y rodilla pueden verse distorsionados por perspectiva frontal. "
            "Ideal para detectar apertura de stance y cruce de piernas."
        ),
        "Fondo Trasero": (
            "vista POSTERIOR (detrás del jugador) — el jugador golpea alejándose de cámara. "
            "Máxima visibilidad de la cadena cinética: tobillo→rodilla→cadera→hombro→codo→muñeca. "
            "Ángulo óptimo para evaluar rotación de tronco, carga de piernas y follow-through completo. "
            "Referencia estándar ATP para análisis biomecánico."
        ),
        "Fondo Frontal": (
            "vista FRONTAL LEJANA (frente al jugador) — el jugador golpea hacia la cámara. "
            "Alta visibilidad de preparación, posición de raqueta y extensión de brazo. "
            "Ángulos de rodilla y cadera pueden subestimarse por perspectiva frontal. "
            "Útil para evaluar punto de impacto relativo al cuerpo."
        ),
        "Lateral": (
            "vista LATERAL (costado del jugador) — cámara a la altura de la línea de saque (~1.5 m), "
            "ángulo perpendicular al plano de movimiento del jugador. "
            "Visibilidad óptima de ángulos sagitales: flexión de rodilla, inclinación de torso, "
            "arco completo del brazo en el swing y extensión en el follow-through. "
            "A esta altura la cámara captura bien la fase de carga de piernas y el punto de impacto. "
            "PRECAUCIÓN: la perspectiva puede comprimir ligeramente los ángulos de cadera en el eje vertical "
            "y subestimar la rotación de tronco. Los ángulos de hombro y codo del brazo más alejado "
            "pueden aparecer ocluidos por el torso. Ángulo preferido para análisis de saque y timing."
        ),
    }

    position_hints = {
        "Izquierda": (
            "cámara a la IZQUIERDA del jugador — visibilidad reducida del lado dominante derecho, "
            "mayor detalle del revés y el lado izquierdo del cuerpo."
        ),
        "Centro": (
            "cámara al CENTRO — visión simétrica del jugador, mínimo sesgo de perspectiva lateral."
        ),
        "Derecha": (
            "cámara a la DERECHA del jugador — visibilidad mejorada del lado dominante derecho, "
            "ideal para capturar forehand y saque de diestros."
        ),
    }

    origin_text   = origin_hints.get(origin, f"Ángulo '{origin}' — interpretar con cautela.")
    position_text = position_hints.get(position, f"Posición '{position}'.")

    return (
        f"ÁNGULO DE CÁMARA: {camera_orientation}\n"
        f"  • Origen  ({origin}): {origin_text}\n"
        f"  • Posición ({position}): {position_text}\n"
        f"  → Calibra tus estimaciones de ángulos considerando la perspectiva descrita."
    )


def format_equipment_context(equipment_used: dict | None, dominant_hand: str | None) -> str:
    """
    Construye el bloque de contexto de equipamiento y lateralidad del jugador.
    equipment_used : {brand, model, head_size, nickname}
    dominant_hand  : "right" | "left"
    """
    lines = []

    # ── Mano dominante ──
    if dominant_hand == "right":
        lines.append("MANO DOMINANTE: Derecha — brazo de golpe: DERECHO.")
        lines.append("  → Forehand/saque: priorizar métricas del lado derecho (right_elbow, right_hip).")
        lines.append("  → Backhand: el codo izquierdo es el brazo guía; right_elbow es el apoyo.")
    elif dominant_hand == "left":
        lines.append("MANO DOMINANTE: Izquierda (ZURDO) — brazo de golpe: IZQUIERDO.")
        lines.append("  → Forehand/saque: priorizar métricas del lado IZQUIERDO (left_elbow, left_hip).")
        lines.append("  → Backhand: el codo derecho es el brazo guía; left_elbow es el apoyo.")
        lines.append("  ⚠️  IMPORTANTE: invertir todas las referencias de lateralidad en la evaluación.")
    else:
        lines.append("MANO DOMINANTE: No especificada — asumir diestro como default.")

    # ── Equipamiento ──
    if equipment_used and isinstance(equipment_used, dict):
        brand     = equipment_used.get("brand", "")
        model     = equipment_used.get("model", "")
        head_size = equipment_used.get("head_size", "")
        nickname  = equipment_used.get("nickname", "")

        racket_str = " ".join(filter(None, [brand, model])) or "Raqueta sin identificar"
        if nickname:
            racket_str += f' ("{nickname}")'

        lines.append(f"\nEQUIPAMIENTO USADO: {racket_str}")

        if head_size:
            size_hint = ""
            try:
                size_num = float("".join(c for c in str(head_size) if c.isdigit() or c == "."))
                if size_num <= 95:
                    size_hint = "cabeza pequeña — control máximo, sweet spot reducido (perfil avanzado/experto)."
                elif size_num <= 100:
                    size_hint = "cabeza media — balance control/potencia (perfil intermedio-avanzado)."
                elif size_num <= 105:
                    size_hint = "cabeza media-grande — más potencia y margen de error (perfil intermedio)."
                else:
                    size_hint = "cabeza grande — máxima área de impacto, mayor margen de error (perfil principiante-intermedio)."
            except (ValueError, TypeError):
                size_hint = f"tamaño {head_size}."

            lines.append(f"  • Cabeza: {head_size} in² — {size_hint}")
            lines.append(
                "  → El tamaño de cabeza afecta la tolerancia al error en el punto de impacto: "
                "considera esto al evaluar consistencia y uso del sweet spot."
            )
    else:
        lines.append("\nEQUIPAMIENTO USADO: No especificado.")

    return "\n".join(lines)


def format_session_context(session_type: str) -> str:
    """
    Traduce el tipo de sesión (clase | paleteo | partido) a instrucciones
    concretas para que los agentes ajusten tono, expectativas y scoring.
    """
    contexts = {
        "clase": (
            "CONTEXTO DE SESIÓN: CLASE CON INSTRUCTOR\n"
            "  • El jugador estaba bajo supervisión directa — se esperan posiciones más controladas.\n"
            "  • Si hay errores técnicos, son patrones arraigados (el instructor ya los estaba corrigiendo).\n"
            "  → Sé exigente en el análisis: un error en clase es más relevante que en partido.\n"
            "  → Tono del reporte: pedagógico y constructivo, orientado al aprendizaje.\n"
            "  → Scoring: aplicar los umbrales estándar sin descuentos por presión situacional."
        ),
        "paleteo": (
            "CONTEXTO DE SESIÓN: PALETEO / PELOTEO LIBRE\n"
            "  • Práctica semi-controlada sin presión de resultado — la técnica observada\n"
            "    es representativa del nivel base del jugador.\n"
            "  • Los errores suelen ser de hábito o automatización incompleta.\n"
            "  → Tono del reporte: técnico y directo, orientado a consolidar automatismos.\n"
            "  → Scoring: aplicar los umbrales estándar — es la línea base de referencia."
        ),
        "partido": (
            "CONTEXTO DE SESIÓN: PARTIDO COMPETITIVO\n"
            "  • Alta presión situacional: la técnica naturalmente se degrada bajo estrés.\n"
            "  • Errores aislados son esperables — priorizar patrones que se repiten.\n"
            "  • Mantener una buena forma técnica bajo presión es mérito adicional.\n"
            "  → Sé más comprensivo con desviaciones técnicas puntuales.\n"
            "  → Tono del reporte: estratégico y realista, orientado a la competencia.\n"
            "  → Scoring: considera un margen de tolerancia de ~5 puntos por degradación táctica.\n"
            "  → Prioriza consistencia y control sobre perfección biomecánica."
        ),
    }
    return contexts.get(
        session_type,
        f"CONTEXTO DE SESIÓN: {session_type.upper()}\n  → Aplicar criterios estándar de evaluación."
    )


# ══════════════════════════════════════════════════════════════
# HELPERS DE FRAMES
# ══════════════════════════════════════════════════════════════

def get_stroke_frames_or_fallback(
    coordinator_data: dict,
    mediapipe_data: dict,
    stroke: str,
) -> tuple[list, bool]:
    """
    Devuelve (frames_filtrados, es_fallback).

    1. Intenta filtrar frames de MediaPipe usando los índices enteros del coordinador.
    2. Si el coordinador no clasificó frames para ese golpe (lista vacía o ausente),
       devuelve TODOS los frames de MediaPipe como fallback con es_fallback=True.

    El caller debe incluir una advertencia en el reporte si es_fallback=True.

    NOTA: usa índices enteros (frame["frame"]) — nunca timestamps float
    para evitar el silent-fail de comparación de floats.
    """
    indices = set(coordinator_data.get("frames_by_stroke", {}).get(stroke, []))
    if indices:
        filtered = [f for f in mediapipe_data.get("frames", []) if f["frame"] in indices]
        if filtered:
            return filtered, False
    # Fallback: usar todos los frames disponibles
    return mediapipe_data.get("frames", []), True


def extract_peak_frames(
    mediapipe_data: dict,
    coordinator_data: dict,
    dominant_hand: str = None,
) -> dict:
    """
    Identifica los keyframes biomecánicamente más relevantes por golpe.
    Devuelve un dict con frames de preparación (codo más cerrado) e impacto
    (codo más abierto) para cada golpe activo.

    Estos frames se guardan en raw_data['digital_twin_data'] para que
    Three.js pueda renderizar la silueta del jugador en el frontend (v2).

    NOTA: MediaPipe en video 2D entrega coordenadas x/y normalizadas
    y z estimada (no confiable). Los ángulos calculados son los que
    se usan para las comparaciones; z se incluye solo como referencia visual.
    """
    is_lefty       = dominant_hand == "left"
    dom_elbow_key  = "left_elbow"  if is_lefty else "right_elbow"   # forehand / saque
    guide_elbow_key = "right_elbow" if is_lefty else "left_elbow"   # backhand (brazo guía)
    all_frames = mediapipe_data.get("frames", [])
    active     = coordinator_data.get("active_agents", [])
    fbs        = coordinator_data.get("frames_by_stroke", {})

    result: dict = {}

    for stroke in ("forehand", "backhand", "saque"):
        if stroke not in active:
            continue

        # Bug 2 fix: backhand usa el codo guía (no-dominante); forehand y saque usan el dominante
        elbow_key = guide_elbow_key if stroke == "backhand" else dom_elbow_key

        indices = set(fbs.get(stroke, []))
        stroke_frames = [f for f in all_frames if f["frame"] in indices] if indices else all_frames
        if not stroke_frames:
            continue

        # Para saque: impacto = extensión máxima (ángulo mayor)
        # Para forehand/backhand: preparación = codo más cerrado, impacto = codo más abierto
        # Solo incluir frames con dato real para elbow_key (evita que default 0 sesga la selección)
        angles = [
            (f, f["angles"][elbow_key])
            for f in stroke_frames
            if elbow_key in f.get("angles", {}) and f["angles"][elbow_key] is not None
        ]
        if not angles:
            continue

        frame_prep   = min(angles, key=lambda x: x[1])[0]  # codo más cerrado = preparación
        frame_impact = max(angles, key=lambda x: x[1])[0]  # codo más abierto = impacto/extensión

        result[stroke] = {
            "preparacion": {
                "frame":              frame_prep["frame"],
                "timestamp":          frame_prep["timestamp"],
                "angles":             frame_prep["angles"],
                "shoulder_alignment": frame_prep["shoulder_alignment"],
                "landmarks_3d":       frame_prep.get("landmarks_3d"),
            },
            "impacto": {
                "frame":              frame_impact["frame"],
                "timestamp":          frame_impact["timestamp"],
                "angles":             frame_impact["angles"],
                "shoulder_alignment": frame_impact["shoulder_alignment"],
                "landmarks_3d":       frame_impact.get("landmarks_3d"),
            },
        }

    return result


def detect_stroke_phases(
    stroke_frames: list,
    elbow_key: str = "right_elbow",
) -> dict | None:
    """
    Detecta las fases biomecánicas de un golpe usando derivadas angulares.

    Estrategia:
      1. Serie temporal de ángulos de codo dominante.
      2. Velocidad angular   = diferencia finita de primer orden (delta_angle/frame).
      3. Aceleración angular = diferencia finita de segundo orden (delta_vel/frame).
      4. Fases:
           preparacion    -> primer frame de la secuencia
           carga          -> frame con ángulo MÍNIMO (codo más cerrado = máxima carga)
           impacto        -> frame con MÁXIMA aceleración positiva post-carga
                            (pico de despliegue = transferencia real de energía)
           follow_through -> frame con ángulo MÁXIMO post-impacto

    Por qué aceleración y no ángulo máximo para el impacto:
      El codo sigue extendiéndose DESPUÉS del contacto. El pico de aceleración
      angular coincide con la máxima transferencia cinética — el impacto real,
      no la extensión final del follow-through.

    Args:
        stroke_frames : lista de frames MediaPipe ordenados temporalmente,
                        cada uno con {frame, timestamp, angles, ...}.
        elbow_key     : key del ángulo de codo dominante a analizar.

    Returns:
        Dict con las 4 fases (cada una con el frame completo + métricas),
        la serie de ángulos/velocidad, y metadatos.
        None si hay menos de 4 frames (insuficiente para derivadas).
    """
    if len(stroke_frames) < 4:
        return None

    # Ordenar por frame index para garantizar orden temporal
    frames = sorted(stroke_frames, key=lambda f: f["frame"])
    angles = [f["angles"].get(elbow_key, 0.0) for f in frames]
    n      = len(angles)

    # Velocidad angular (diferencia finita centrada donde sea posible)
    velocity    = [0.0] * n
    for i in range(1, n - 1):
        velocity[i] = (angles[i + 1] - angles[i - 1]) / 2.0
    velocity[0]     = angles[1]  - angles[0]
    velocity[n - 1] = angles[-1] - angles[-2]

    # Aceleración angular
    accel    = [0.0] * n
    for i in range(1, n - 1):
        accel[i] = velocity[i + 1] - velocity[i - 1]
    accel[0]     = velocity[1]  - velocity[0]
    accel[n - 1] = velocity[-1] - velocity[-2]

    # CARGA: ángulo mínimo
    carga_idx = angles.index(min(angles))

    # IMPACTO: pico de aceleración positiva ESTRICTAMENTE después de la carga.
    # Empezamos en carga_idx+1 para excluir el frame de carga mismo:
    # la diferencia finita centrada le asigna aceleración alta al valle porque
    # los vecinos tienen velocidades muy distintas — no es el impacto real.
    search_start = carga_idx + 1
    if search_start >= n:
        # Carga es el último frame → impacto = carga (secuencia incompleta)
        impact_idx = carga_idx
    else:
        post_carga_accels = accel[search_start:]
        if max(post_carga_accels) > 0:
            impact_idx = search_start + post_carga_accels.index(max(post_carga_accels))
        else:
            # Sin aceleración positiva post-carga → fallback al ángulo máximo
            post_carga_angles = angles[search_start:]
            impact_idx = search_start + post_carga_angles.index(max(post_carga_angles))

    # FOLLOW-THROUGH: ángulo máximo después del impacto
    post_impact_angles = angles[impact_idx:]
    follow_idx = impact_idx + post_impact_angles.index(max(post_impact_angles))

    # PREPARACION: primer frame
    prep_idx = 0

    def _frame_entry(idx):
        f = frames[idx]
        return {
            "frame":              f["frame"],
            "timestamp":          f["timestamp"],
            "angles":             f["angles"],
            "shoulder_alignment": f.get("shoulder_alignment"),
            "landmarks_3d":       f.get("landmarks_3d"),
            "phase_index":        idx,
            "elbow_angle":        round(angles[idx], 1),
            "elbow_velocity":     round(velocity[idx], 2),
            "elbow_accel":        round(accel[idx], 2),
        }

    return {
        "preparacion":    _frame_entry(prep_idx),
        "carga":          _frame_entry(carga_idx),
        "impacto":        _frame_entry(impact_idx),
        "follow_through": _frame_entry(follow_idx),
        "total_frames":   n,
        "elbow_key":      elbow_key,
        "angle_series":   [round(a, 1) for a in angles],
        "velocity_series":[round(v, 2) for v in velocity],
    }


# ══════════════════════════════════════════════════════════════
# HELPERS DE PARSING Y SCORING
# ══════════════════════════════════════════════════════════════

def parse_json_response(text: str) -> dict:
    """
    Extrae JSON de la respuesta de un agente LLM.
    Maneja: bloques ```json, JSON truncado, respuestas parciales.
    """
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Intentar cerrar JSON truncado
        for closing in ["}", "}}", "}}}}"]:
            try:
                return json.loads(raw + closing)
            except Exception:
                continue

        # Extracción parcial de campos clave
        partial = {}
        for key, cast in [
            ("global_score", float),
            ("nivel_general", str),
            ("diagnostico_global", str),
            ("session_type", str),
        ]:
            pattern = '"' + key + '"' + r'\s*:\s*"?([^,"\}\n]+)"?'
            m = re.search(pattern, raw)
            if m:
                try:
                    partial[key] = cast(m.group(1).strip().strip('"'))
                except Exception:
                    partial[key] = m.group(1).strip().strip('"')

        if "reporte_narrativo_completo" in raw:
            idx = raw.find('"reporte_narrativo_completo"')
            partial["reporte_narrativo_completo"] = raw[idx + 30 : idx + 5000]

        if partial:
            return partial

        # Tercer intento: extraer JSON embebido en texto narrativo del LLM
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass

        return {"error": "JSON truncado", "raw": raw[:500]}


def assign_level_from_score(score: float) -> str:
    """
    Convierte score numérico (0-100) a etiqueta de nivel.
    Tabla de producto:
      0-50  → principiante
      51-75 → intermedio
      76-90 → avanzado
      91+   → experto
    """
    if score <= 50:
        return "principiante"
    elif score <= 75:
        return "intermedio"
    elif score <= 90:
        return "avanzado"
    else:
        return "experto"


# ══════════════════════════════════════════════════════════════
# OPENROUTER CONFIGURATION
# ══════════════════════════════════════════════════════════════

MODEL_MAPPING = {
    "coordinator": "mistralai/mistral-large-2407",
    "specialist":   "deepseek/deepseek-r1",
    "synthesizer":  "z-ai/glm-5",
    "prescription": "openai/gpt-4o-mini",
    "quality":      "google/gemini-flash-1.5"
}


def get_openrouter_client(api_key: str = None):
    """
    Crea un cliente OpenAI compatible con OpenRouter.
    OpenRouter proporciona un endpoint OpenAI-compatible en https://openrouter.io/api/v1

    Args:
        api_key: OpenRouter API key. Si no se proporciona, se lee de OPENROUTER_API_KEY

    Returns:
        Cliente OpenAI configurado para usar OpenRouter
    """
    from openai import OpenAI
    import os

    _api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    return OpenAI(
        api_key=_api_key,
        base_url="https://openrouter.io/api/v1"
    )


def get_model_for_agent(agent_type: str) -> str:
    """
    Retorna el modelo OpenRouter para un tipo de agente específico.

    Args:
        agent_type: "coordinator", "specialist", "synthesizer", "prescription", "quality"

    Returns:
        Model ID en OpenRouter (ej: "mistralai/mistral-large-2407")
    """
    return MODEL_MAPPING.get(agent_type, "mistralai/mistral-large-2407")
