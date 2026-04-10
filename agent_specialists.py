"""
TennisAI — Agentes Especialistas (Forehand & Backhand)
═══════════════════════════════════════════════════════
Responsabilidades:
  - run_agent_forehand : análisis biomecánico del golpe de derecha
  - run_agent_backhand : análisis biomecánico del golpe de revés

Cada agente:
  1. Lee stroke_stats desde coordinator_data (avg + std_dev por golpe)
  2. Filtra impact_frames del golpe correspondiente
  3. Calcula métricas en punto de impacto
  4. Genera JSON estructurado con scoring en 6 dimensiones
  5. Genera narrativa en prosa (200-300 palabras)

El bloque de CONSISTENCIA (std_dev) es nuevo respecto a la versión
anterior inlinada en agents_pipeline.py. Permite al LLM distinguir
entre un jugador con ángulos correctos pero inconsistentes vs uno
con ángulos subóptimos pero reproducibles.

Imports esperados en agents_pipeline.py:
    from agent_specialists import run_agent_forehand, run_agent_backhand

Los decoradores @app.function viven en agents_pipeline.py.
Este módulo es Python puro, testeable localmente sin Modal.

Test local desde Colab:
    from agent_specialists import run_agent_forehand
    result = run_agent_forehand(
        coordinator_data=coordinator_result,
        mediapipe_data=mediapipe_slim,
        ball_data=ball_slim,
        camera_orientation="Lateral-Derecha",
        equipment_used={"brand": "Wilson", "model": "Pro Staff", "head_size": "97"},
        dominant_hand="right",
        session_type="paleteo",
        api_key="sk-ant-...",
    )
"""

import json


# ─────────────────────────────────────────────────────────────────────────────
# AGENTE FOREHAND
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_forehand(
    coordinator_data:   dict,
    mediapipe_data:     dict,
    ball_data:          dict,
    camera_orientation: str  = None,
    equipment_used:     dict = None,
    dominant_hand:      str  = None,
    session_type:       str  = "paleteo",
    api_key:            str  = None,
) -> dict:
    """
    Analiza el forehand biomecánicamente.

    Inputs clave desde coordinator_data:
      - stroke_stats["forehand"]  : avg + std_dev pre-calculados
      - impact_frames             : impactos clasificados por golpe
      - frames_by_stroke          : índices de frame por fase biomecánica

    Returns:
        dict con keys:
          stroke, scores, total_score, nivel,
          analisis_tecnico, metricas_clave,
          observaciones_detalladas, narrativa_seccion,
          datos_insuficientes
    """
    import os

    from helpers import (
        format_camera_context,
        format_equipment_context,
        format_session_context,
        parse_json_response,
        get_stroke_frames_or_fallback,
        get_openrouter_client,
        get_model_for_agent,
    )

    _api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    client   = get_openrouter_client(_api_key)

    camera_ctx    = format_camera_context(camera_orientation)
    equipment_ctx = format_equipment_context(equipment_used, dominant_hand)
    session_ctx   = format_session_context(session_type)

    # ── Lateralidad ──────────────────────────────────────────
    is_lefty   = dominant_hand == "left"
    dom_side   = "izquierdo" if is_lefty else "derecho"
    dom_abbrev = "izq"       if is_lefty else "der"
    elbow_key  = "left_elbow"  if is_lefty else "right_elbow"
    knee_key   = "left_knee"   if is_lefty else "right_knee"
    hip_key    = "left_hip"    if is_lefty else "right_hip"

    # ── Frames filtrados por golpe ────────────────────────────
    fh_frames, fh_fallback = get_stroke_frames_or_fallback(
        coordinator_data, mediapipe_data, "forehand"
    )
    impact_frames = coordinator_data.get("impact_frames", [])
    fh_impacts    = [
        f for f in impact_frames
        if f.get("stroke_type") in ("forehand",)
    ]

    fallback_warning = (
        "\n⚠️  ADVERTENCIA: El coordinador no identificó frames específicos de forehand. "
        "Se están usando promedios globales del video. "
        "El análisis puede ser menos preciso — incluir esta nota en observaciones_detalladas."
    ) if fh_fallback else ""

    # ── Promedios globales (MediaPipe summary) ────────────────
    summary      = mediapipe_data.get("summary", {})
    sum_elbow    = summary.get(f"avg_{elbow_key}", 0)
    sum_knee     = summary.get(f"avg_{knee_key}",  0)
    sum_hip      = summary.get(f"avg_{hip_key}",   0)

    # ── Ángulos en punto de impacto ───────────────────────────
    if fh_impacts:
        imp_elbow    = round(sum(f["angles"].get(elbow_key, 0) for f in fh_impacts) / len(fh_impacts), 1)
        imp_knee     = round(sum(f["angles"].get(knee_key,  0) for f in fh_impacts) / len(fh_impacts), 1)
        imp_hip      = round(sum(f["angles"].get(hip_key,   0) for f in fh_impacts) / len(fh_impacts), 1)
        imp_shoulder = round(sum(f["shoulder_alignment"] for f in fh_impacts) / len(fh_impacts), 2)
        impact_note  = f"ÁNGULOS EN PUNTO DE IMPACTO ({len(fh_impacts)} impactos detectados):"
    elif fh_frames:
        imp_elbow    = round(sum(f["angles"][elbow_key] for f in fh_frames) / len(fh_frames), 1)
        imp_knee     = round(sum(f["angles"][knee_key]  for f in fh_frames) / len(fh_frames), 1)
        imp_hip      = round(sum(f["angles"][hip_key]   for f in fh_frames) / len(fh_frames), 1)
        imp_shoulder = round(sum(f["shoulder_alignment"] for f in fh_frames) / len(fh_frames), 2)
        impact_note  = f"ÁNGULOS EN FRAMES DE FOREHAND ({len(fh_frames)} frames filtrados):"
    else:
        imp_elbow, imp_knee, imp_hip = sum_elbow, sum_knee, sum_hip
        imp_shoulder = summary.get("avg_shoulder_alignment", 0)
        impact_note  = "ÁNGULOS GLOBALES (sin frames de forehand aislados):"

    # ── Ball speed filtrada por forehand ─────────────────────
    # Si no hay impactos clasificados de forehand, reportar 0 en lugar de
    # promediar toda la sesión (que incluiría saques u otros golpes).
    if fh_impacts:
        fh_speeds = [f.get("ball_speed_pixels", 0) for f in fh_impacts if f.get("ball_speed_pixels")]
    else:
        fh_speeds = []
    max_ball_speed = max(fh_speeds) if fh_speeds else 0
    avg_ball_speed = round(sum(fh_speeds) / len(fh_speeds), 1) if fh_speeds else 0

    # ── Validación de sincronización pelota-pose ──────────────
    # ball_validated=False → ball tracker no confirmó el impacto
    # → potencia_pelota debe ser 0 en el scoring.
    impact_validation  = coordinator_data.get("data_quality", {}).get("impact_validation", {})
    fh_ball_validated  = impact_validation.get("forehand", {}).get("ball_validated", True)
    ball_validation_note = (
        "\n⚠️  VALIDACIÓN DE PELOTA: El ball tracker NO confirmó el impacto de forehand "
        "(diff_ms > 100ms o sin detección). "
        "El score de potencia_pelota DEBE ser 0 — la medición no es confiable. "
        "Indica esto en la justificación de potencia_pelota."
    ) if not fh_ball_validated else ""

    # ── Consistencia (std_dev), fatiga y posición ────────────
    fh_stats     = coordinator_data.get("stroke_stats", {}).get("forehand", {})
    fh_fatigue   = coordinator_data.get("fatigue_by_stroke", {}).get("forehand", {})
    player_pos   = coordinator_data.get("player_position", {})
    std_elbow    = fh_stats.get("std_dom_elbow",           0)
    std_knee     = fh_stats.get("std_dom_knee",            0)
    std_hip      = fh_stats.get("std_dom_hip",             0)
    std_shoulder = fh_stats.get("std_shoulder_alignment",  0)
    std_ball     = fh_stats.get("std_ball_speed",          0)
    n_impacts    = fh_stats.get("n_impacts",               len(fh_impacts))

    consistency_block = _format_consistency_block(
        stroke="forehand",
        dom_side=dom_side,
        std_elbow=std_elbow,
        std_knee=std_knee,
        std_hip=std_hip,
        std_shoulder=std_shoulder,
        std_ball=std_ball,
        n_impacts=n_impacts,
    )

    # ── Fatiga, posición, grip y fases listos para prompt ──────────
    fatigue_note   = fh_fatigue.get("narrative", "")
    position_note  = player_pos.get("position_note", "")
    fh_grip        = coordinator_data.get("forehand_grip", {})
    fh_grip_type   = fh_grip.get("grip", "unknown")
    fh_grip_note   = fh_grip.get("biomechanical_note", "")
    fh_grip_cam    = fh_grip.get("camera_compression_note", "")

    # ── Rangos dinámicos según grip detectado ────────────────
    # Debe ir ANTES de _format_phase_block que usa fh_grip_label
    fh_elbow_range, fh_grip_label = _get_forehand_elbow_range(fh_grip_type)
    fh_elbow_opt = f"{fh_elbow_range[0]}-{fh_elbow_range[1]}°"

    # ── Fases biomecánicas ────────────────────────────────────
    phase_angles_all = coordinator_data.get("phase_angles", {})
    fh_phase_data    = phase_angles_all.get("forehand", {})
    phase_block      = _format_phase_block(fh_phase_data, fh_grip_label)

    # Bloque de grip para el prompt: solo se inyecta si hay info útil
    if fh_grip_type != "unknown" and fh_grip_note:
        grip_block = (
            f"\nGRIP DE FOREHAND DETECTADO: {fh_grip_note}"
            + (f"\n{fh_grip_cam}" if fh_grip_cam else "")
            + f"\n  → Usa los rangos de {fh_grip_label} para evaluar punto_impacto."
            + f"\n  → NO uses los rangos genéricos ATP (90-120°) — no aplican para este grip."
        )
    elif fh_grip_note:  # unknown pero con nota
        grip_block = (
            f"\nGRIP DE FOREHAND: {fh_grip_note}"
            + (f"\n{fh_grip_cam}" if fh_grip_cam else "")
        )
    else:
        grip_block = ""

    # ── Prompt optimizado para Gemini 2.0 Flash con análisis profundo ──
    prompt = f"""Eres experto biomecánico en forehand de tenis. Analiza PROFUNDAMENTE estos datos reales.
Responde SOLO con JSON válido, sin markdown, sin backticks.

═══ CONTEXTO DE SESIÓN ═══
{session_ctx}

{camera_ctx}

{equipment_ctx}

═══ DATOS BIOMECÁNICOS COMPLETOS ═══

PROMEDIOS GLOBALES (sesión completa):
  Codo {dom_side} promedio: {sum_elbow}° | Rango óptimo ({fh_grip_label}): {fh_elbow_opt}
  Rodilla {dom_side} promedio: {sum_knee}° | Rango óptimo: 130-150°
  Cadera {dom_side} promedio: {sum_hip}° | Rango óptimo: 140-160°
  Alineación hombros promedio: {summary.get('avg_shoulder_alignment', 0)}° | Óptimo: < 5°
  Velocidad pelota máx: {max_ball_speed} px/frame | Promedio: {avg_ball_speed} px/frame

PUNTO DE IMPACTO (instantánea crítica):
{impact_note}
  Codo {dom_side}: {imp_elbow}° vs óptimo {fh_grip_label} {fh_elbow_opt} (delta: {abs(imp_elbow - fh_elbow_range[0]) if imp_elbow < fh_elbow_range[0] else abs(imp_elbow - fh_elbow_range[1]) if imp_elbow > fh_elbow_range[1] else 0}°)
  Rodilla {dom_side}: {imp_knee}° vs óptimo 130-150° (delta: {abs(imp_knee - 140) if imp_knee not in range(130, 151) else 0}°)
  Cadera {dom_side}: {imp_hip}° vs óptimo 140-160° (delta: {abs(imp_hip - 150) if imp_hip not in range(140, 161) else 0}°)
  Alineación hombros: {imp_shoulder}° vs óptimo < 5° (delta: {abs(imp_shoulder)}°)

CONSISTENCIA (Desviación Estándar - indicador de repetibilidad):
  Std Codo {dom_side}: {std_elbow}° ({std_elbow > 20 and "⚠️ INCONSISTENTE" or std_elbow > 10 and "⚠️ Moderado" or "✓ Consistente"})
  Std Rodilla {dom_side}: {std_knee}°
  Std Cadera {dom_side}: {std_hip}°
  Std Alineación hombros: {std_shoulder}° ({std_shoulder > 8 and "⚠️ Rotación irregular" or "✓ Alineación consistente"})
  Std Velocidad pelota: {std_ball}° (variación de potencia)
  Número de impactos analizados: {n_impacts}

{grip_block}

{phase_block}

{consistency_block}

═══ CONTEXTO BIOMECÁNICO ADICIONAL ═══
FATIGA DETECTADA: {fatigue_note}
  → Aplicar penalización en últimas repeticiones si hay degradación

POSICIÓN EN CANCHA: {position_note}
  → Ajustar expectativas según posición (cerca/atrás/lado)

═══ FRAMEWORK DE SCORING ═══

DIMENSIONES (máx 100 total):
  1. Preparacion (0-20): rotación inicial de caderas y hombros, posición base
     - Evaluar: hombros alineados, separación hombro-cadera, altura codo
     - Usar ÁNGULOS DE FASE PREPARACIÓN si disponibles

  2. Punto_impacto (0-20): ángulos en el contacto con la pelota
     - Evaluar: codo contra rango {fh_grip_label} ({fh_elbow_opt}), flexión rodilla, rotación cadera
     - CRÍTICO: comparar contra benchmarks ATP para el nivel detectado
     - Delta vs óptimo: 0° = 20pts, +/- 5° = 15pts, +/- 15° = 5pts, > 20° = 0pts

  3. Follow_through (0-20): extensión y deceleración post-impacto
     - Evaluar: brazo extendido, hombro completado rotación, pie base firme
     - Usar ÁNGULOS DE FASE FOLLOW-THROUGH si disponibles

  4. Posicion_pies (0-20): estabilidad base y movimiento de pies
     - Indicador: consistencia postural, ausencia de torsión, equilibrio final

  5. Ritmo_cadencia (0-10): fluidez de cadena cinética
     - Indicador: transición suave caderas→hombros→codo, sin pausas
     - Usar std_dev para validar: std > 20° = fragmentado (< 5pts)

  6. Potencia_pelota (0-10): velocidad resultante de golpe
     - Indicador: ball_speed máx y variación
     - {fh_ball_validated and "✓ Ball validated" or "⚠️ NO validado - score = 0"}

═══ INSTRUCCIONES CRÍTICAS ═══

VALIDACIÓN DE FASES:
  • Si "NO DISPONIBLE": ignorar completamente, usar solo ángulos de impacto
  • Si disponibles: evaluar CADA articulación contra su RANGO DE FASE
  • NO penalizar por falta de fases disponibles
  • Cadera 155° en PREPARACIÓN puede ser CORRECTA (rango prep: 140-170°)
  • NO inferir preparación si solo tienes datos de impacto

CONSISTENCIA EN SCORING:
  • std_elbow > 20°: swing muy variable → penalizar punto_impacto(-5pts) y ritmo_cadencia(-3pts)
  • std_shoulder > 8°: rotación irregular → penalizar preparacion(-5pts) y potencia_pelota(-2pts)
  • Mencionar explícitamente en justificaciones

NIVEL (basado en total_score):
  0-40: principiante | 41-60: intermedio | 61-80: avanzado | 81-100: experto

ANÁLISIS TÉCNICO:
  • Fortalezas: listar 2-3 aspectos donde el jugador supera benchmarks
  • Debilidades: listar 2-3 limitaciones más críticas
  • Patron_error_principal: UN error que se repite y causa degradación
  • Comparacion_optimo: breve párrafo comparando contra jugador ATP nivel promedio

{fallback_warning}{ball_validation_note}

JSON EXACTO (sin backticks, valores numéricos reales):
{{"stroke":"forehand","scores":{{"preparacion":{{"score":0,"max":20,"justificacion":"Detallar rotación y alineación inicial"}},"punto_impacto":{{"score":0,"max":20,"justificacion":"Incluir delta vs óptimo {fh_grip_label} y contexto ATP"}},"follow_through":{{"score":0,"max":20,"justificacion":"Descripción de extensión y desaceleración"}},"posicion_pies":{{"score":0,"max":20,"justificacion":"Estabilidad y base"}},"ritmo_cadencia":{{"score":0,"max":10,"justificacion":"Fluidez de cadena cinética, mencionar std"}},"potencia_pelota":{{"score":0,"max":10,"justificacion":"Velocidad resultante y consistencia"}}}},"total_score":0,"nivel":"principiante|intermedio|avanzado|experto","analisis_tecnico":{{"fortalezas":["",""],"debilidades":["",""],"patron_error_principal":"","comparacion_optimo":""}},"metricas_clave":{{"angulo_codo_{dom_abbrev}_impacto":{imp_elbow},"angulo_codo_{dom_abbrev}_promedio":{sum_elbow},"std_codo_{dom_abbrev}":{std_elbow},"flexion_rodilla_{dom_abbrev}_impacto":{imp_knee},"std_rodilla_{dom_abbrev}":{std_knee},"alineacion_hombros":{imp_shoulder},"std_hombros":{std_shoulder},"velocidad_pelota_max":{max_ball_speed},"std_velocidad_pelota":{std_ball},"impactos_analizados":{n_impacts}}},"observaciones_detalladas":"","datos_insuficientes":{str(fh_fallback).lower()}}}"""

    msg_json = client.chat.completions.create(
        model=get_model_for_agent("specialist"),
        max_tokens=5000,
        messages=[{"role": "user", "content": prompt}],
    )
    result = parse_json_response(msg_json.choices[0].message.content)

    # ── Narrativa (200-300 palabras) ──────────────────────────
    angle_trust  = _angle_trust_hint(camera_orientation)
    fallback_note = (
        " NOTA: Los datos provienen de promedios globales del video "
        "(el coordinador no identificó frames específicos de forehand), "
        "menciona esta limitación brevemente." if fh_fallback else ""
    )

    msg_narrative = client.chat.completions.create(
        model=get_model_for_agent("specialist"),
        max_tokens=2000,
        messages=[{"role": "user", "content": f"""Eres biomecánico especialista en forehand de tenis.
Escribe el análisis narrativo del forehand en 200-300 palabras en español. Prosa técnica, fluida, sin listas.
{session_ctx}
{angle_trust}{fallback_note}
Score: {result.get('total_score', 0)}/100 | Nivel: {result.get('nivel', '')}
Fortalezas: {result.get('analisis_tecnico', {}).get('fortalezas', [])}
Debilidades: {result.get('analisis_tecnico', {}).get('debilidades', [])}
Error principal: {result.get('analisis_tecnico', {}).get('patron_error_principal', '')}
Métricas: codo_impacto={imp_elbow}° (óptimo {fh_grip_label}: {fh_elbow_opt}) (±{std_elbow}°) | rodilla={imp_knee}° (±{std_knee}°) | hombros={imp_shoulder}° (±{std_shoulder}°) | pelota_max={max_ball_speed}px (±{std_ball}px)
Grip detectado: {fh_grip_label} — evalúa el codo contra los rangos de este grip, no contra el estándar ATP genérico.
Interpreta la consistencia: std_codo={std_elbow}° y std_hombros={std_shoulder}° — menciona si el patrón es reproducible o errático."""}],
    )
    result["narrativa_seccion"]  = msg_narrative.choices[0].message.content.strip()
    result["datos_insuficientes"] = fh_fallback
    return result


# ─────────────────────────────────────────────────────────────────────────────
# AGENTE BACKHAND
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_backhand(
    coordinator_data:   dict,
    mediapipe_data:     dict,
    ball_data:          dict,
    camera_orientation: str  = None,
    equipment_used:     dict = None,
    dominant_hand:      str  = None,
    session_type:       str  = "paleteo",
    api_key:            str  = None,
) -> dict:
    """
    Analiza el backhand biomecánicamente.

    En backhand el brazo GUÍA es el opuesto al dominante:
      diestro → brazo guía = izquierdo (left_elbow)
      zurdo   → brazo guía = derecho   (right_elbow)

    Inputs clave desde coordinator_data:
      - stroke_stats["backhand"]  : avg + std_dev pre-calculados
      - impact_frames             : impactos clasificados por golpe
      - frames_by_stroke          : índices de frame por fase biomecánica

    Returns:
        dict con keys:
          stroke, scores, total_score, nivel,
          analisis_tecnico, metricas_clave,
          observaciones_detalladas, narrativa_seccion,
          datos_insuficientes
    """
    import os

    from helpers import (
        format_camera_context,
        format_equipment_context,
        format_session_context,
        parse_json_response,
        get_stroke_frames_or_fallback,
        get_openrouter_client,
        get_model_for_agent,
    )

    _api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    client   = get_openrouter_client(_api_key)

    camera_ctx    = format_camera_context(camera_orientation)
    equipment_ctx = format_equipment_context(equipment_used, dominant_hand)
    session_ctx   = format_session_context(session_type)

    # ── Lateralidad: en backhand el brazo guía es el CONTRALATERAL ──
    is_lefty        = dominant_hand == "left"
    guide_side      = "derecho"      if is_lefty else "izquierdo"
    guide_abbrev    = "der"          if is_lefty else "izq"
    guide_elbow_key = "right_elbow"  if is_lefty else "left_elbow"
    guide_hip_key   = "right_hip"    if is_lefty else "left_hip"
    knee_key        = "left_knee"    if is_lefty else "right_knee"
    knee_side       = "izquierda"    if is_lefty else "derecha"

    # ── Frames filtrados por golpe ────────────────────────────
    bh_frames, bh_fallback = get_stroke_frames_or_fallback(
        coordinator_data, mediapipe_data, "backhand"
    )
    impact_frames = coordinator_data.get("impact_frames", [])
    bh_impacts    = [
        f for f in impact_frames
        if f.get("stroke_type") in ("backhand",)
    ]

    fallback_warning = (
        "\n⚠️  ADVERTENCIA: El coordinador no identificó frames específicos de backhand. "
        "Se están usando promedios globales del video. "
        "El análisis puede ser menos preciso — incluir esta nota en observaciones_detalladas."
    ) if bh_fallback else ""

    # ── Promedios globales (MediaPipe summary) ────────────────
    summary          = mediapipe_data.get("summary", {})
    sum_guide_elbow  = summary.get(f"avg_{guide_elbow_key}", 0)
    sum_knee         = summary.get(f"avg_{knee_key}",        0)
    sum_guide_hip    = summary.get(f"avg_{guide_hip_key}",   0)

    # ── Ángulos en punto de impacto ───────────────────────────
    if bh_impacts:
        imp_guide_elbow = round(sum(f["angles"].get(guide_elbow_key, 0) for f in bh_impacts) / len(bh_impacts), 1)
        imp_knee        = round(sum(f["angles"].get(knee_key,        0) for f in bh_impacts) / len(bh_impacts), 1)
        imp_guide_hip   = round(sum(f["angles"].get(guide_hip_key,   0) for f in bh_impacts) / len(bh_impacts), 1)
        imp_shoulder    = round(sum(f["shoulder_alignment"]  for f in bh_impacts) / len(bh_impacts), 2)
        impact_note     = f"ÁNGULOS EN PUNTO DE IMPACTO ({len(bh_impacts)} impactos detectados):"
    elif bh_frames:
        imp_guide_elbow = round(sum(f["angles"][guide_elbow_key] for f in bh_frames) / len(bh_frames), 1)
        imp_knee        = round(sum(f["angles"][knee_key]        for f in bh_frames) / len(bh_frames), 1)
        imp_guide_hip   = round(sum(f["angles"][guide_hip_key]   for f in bh_frames) / len(bh_frames), 1)
        imp_shoulder    = round(sum(f["shoulder_alignment"]      for f in bh_frames) / len(bh_frames), 2)
        impact_note     = f"ÁNGULOS EN FRAMES DE BACKHAND ({len(bh_frames)} frames filtrados):"
    else:
        imp_guide_elbow, imp_knee, imp_guide_hip = sum_guide_elbow, sum_knee, sum_guide_hip
        imp_shoulder = summary.get("avg_shoulder_alignment", 0)
        impact_note  = "ÁNGULOS GLOBALES (sin frames de backhand aislados):"

    # ── Ball speed filtrada por backhand ──────────────────────
    # Si no hay impactos clasificados de backhand, reportar 0.
    if bh_impacts:
        bh_speeds = [f.get("ball_speed_pixels", 0) for f in bh_impacts if f.get("ball_speed_pixels")]
    else:
        bh_speeds = []
    max_ball_speed = max(bh_speeds) if bh_speeds else 0
    avg_ball_speed = round(sum(bh_speeds) / len(bh_speeds), 1) if bh_speeds else 0

    # ── Validación de sincronización pelota-pose ──────────────
    impact_validation  = coordinator_data.get("data_quality", {}).get("impact_validation", {})
    bh_ball_validated  = impact_validation.get("backhand", {}).get("ball_validated", True)
    ball_validation_note = (
        "\n⚠️  VALIDACIÓN DE PELOTA: El ball tracker NO confirmó el impacto de backhand "
        "(diff_ms > 100ms o sin detección). "
        "El score de potencia_pelota DEBE ser 0 — la medición no es confiable. "
        "Indica esto en la justificación de potencia_pelota."
    ) if not bh_ball_validated else ""

    # ── Consistencia (std_dev), fatiga, posición y grip ─────
    bh_stats     = coordinator_data.get("stroke_stats", {}).get("backhand", {})
    bh_fatigue   = coordinator_data.get("fatigue_by_stroke", {}).get("backhand", {})
    player_pos   = coordinator_data.get("player_position", {})
    bh_grip      = coordinator_data.get("backhand_grip", {})
    std_elbow    = bh_stats.get("std_guide_elbow",         0)   # brazo guía en backhand
    std_knee     = bh_stats.get("std_dom_knee",            0)
    std_hip      = bh_stats.get("std_dom_hip",             0)
    std_shoulder = bh_stats.get("std_shoulder_alignment",  0)
    std_ball     = bh_stats.get("std_ball_speed",          0)
    n_impacts    = bh_stats.get("n_impacts",               len(bh_impacts))

    consistency_block = _format_consistency_block(
        stroke="backhand",
        dom_side=guide_side,
        std_elbow=std_elbow,
        std_knee=std_knee,
        std_hip=std_hip,
        std_shoulder=std_shoulder,
        std_ball=std_ball,
        n_impacts=n_impacts,
        is_guide_arm=True,
    )

    # ── Fatiga, posición, grip y fases listos para prompt ───────────
    fatigue_note  = bh_fatigue.get("narrative", "")
    position_note = player_pos.get("position_note", "")
    grip_note     = bh_grip.get("biomechanical_note", "")
    grip_type     = bh_grip.get("grip", "unknown")

    # ── Detección heurística de slice vs topspin ─────────────
    global_max_speed = mediapipe_data.get("summary", {}).get("max_ball_speed", 0) or max_ball_speed or 1
    slice_indicators = 0
    if imp_guide_elbow > 140:
        slice_indicators += 1
    if max_ball_speed < global_max_speed * 0.6:
        slice_indicators += 1
    if imp_shoulder > 15:
        slice_indicators += 1
    bh_stroke_variant = "slice" if slice_indicators >= 2 else "topspin"
    bh_variant_note = (
        f"⚠️  Indicadores de SLICE detectados: codo_guía={imp_guide_elbow}°>140°, "
        f"vel_pelota={max_ball_speed} vs global_max={global_max_speed:.0f}. "
        f"Refs slice: codo 130-160°, trayectoria descendente, contacto adelante, menor RPM."
    ) if bh_stroke_variant == "slice" else (
        "Backhand TOPSPIN estándar."
    )

    # ── Rangos dinámicos según grip y variante ────────────────
    # Debe ir ANTES de _format_phase_block que usa bh_range_label
    bh_elbow_range, bh_range_label = _get_backhand_elbow_range(grip_type, bh_stroke_variant)
    bh_elbow_opt = f"{bh_elbow_range[0]}-{bh_elbow_range[1]}°"

    # ── Fases biomecánicas ────────────────────────────────────────
    phase_angles_all = coordinator_data.get("phase_angles", {})
    bh_phase_data    = phase_angles_all.get("backhand", {})
    bh_phase_block   = _format_phase_block(bh_phase_data, bh_range_label)

    # ── Prompt optimizado para Gemini 2.0 Flash con análisis profundo ──
    bh_stroke_variant_upper = bh_stroke_variant.upper()
    prompt = f"""Eres experto biomecánico en backhand de tenis. Analiza PROFUNDAMENTE estos datos reales.
Responde SOLO con JSON válido, sin markdown, sin backticks.

═══ CONTEXTO DE SESIÓN ═══
{session_ctx}

{camera_ctx}

{equipment_ctx}

═══ DATOS BIOMECÁNICOS COMPLETOS ═══

PROMEDIOS GLOBALES (sesión completa):
  Codo {guide_side} promedio (brazo GUÍA): {sum_guide_elbow}° | Rango óptimo ({bh_range_label}): {bh_elbow_opt}
  Rodilla {knee_side} promedio: {sum_knee}° | Rango óptimo: 130-150°
  Cadera {guide_side} promedio: {sum_guide_hip}° | Rango óptimo: 140-160°
  Alineación hombros promedio: {summary.get('avg_shoulder_alignment', 0)}° | Óptimo: < 5°
  Velocidad pelota máx: {max_ball_speed} px/frame | Promedio: {avg_ball_speed} px/frame

PUNTO DE IMPACTO (instantánea crítica):
{impact_note}
  Codo {guide_side} (brazo guía): {imp_guide_elbow}° vs óptimo {bh_range_label} {bh_elbow_opt} (delta: {abs(imp_guide_elbow - bh_elbow_range[0]) if imp_guide_elbow < bh_elbow_range[0] else abs(imp_guide_elbow - bh_elbow_range[1]) if imp_guide_elbow > bh_elbow_range[1] else 0}°)
  Rodilla {knee_side}: {imp_knee}° vs óptimo 130-150° (delta: {abs(imp_knee - 140) if imp_knee not in range(130, 151) else 0}°)
  Cadera {guide_side}: {imp_guide_hip}° vs óptimo 140-160° (delta: {abs(imp_guide_hip - 150) if imp_guide_hip not in range(140, 161) else 0}°)
  Alineación hombros: {imp_shoulder}° vs óptimo < 5° (delta: {abs(imp_shoulder)}°)

CONSISTENCIA (Desviación Estándar - indicador de repetibilidad):
  Std Codo {guide_side} (brazo guía): {std_elbow}° ({std_elbow > 20 and "⚠️ MUY INCONSISTENTE" or std_elbow > 10 and "⚠️ Moderado" or "✓ Consistente"})
  Std Rodilla {knee_side}: {std_knee}°
  Std Cadera {guide_side}: {std_hip}°
  Std Alineación hombros: {std_shoulder}° ({std_shoulder > 8 and "⚠️ Rotación irregular" or "✓ Alineación consistente"})
  Std Velocidad pelota: {std_ball}° (variación de potencia)
  Número de impactos analizados: {n_impacts}

{consistency_block}

═══ ANÁLISIS DE VARIANTE TÉCNICA ═══
TIPO DETECTADO: {bh_stroke_variant_upper}
{bh_variant_note}

  • TOPSPIN: trayectoria ascendente, hombros CERRADOS (< 10°), aceleración vertical
  • SLICE/APPROACH: trayectoria DESCENDENTE, hombros MÁS ABIERTOS, velocidad menor es NORMAL
  • Rango óptimo codo guía para {bh_range_label}: {bh_elbow_opt}

TIPO DE GRIP DETECTADO: {grip_note}
  • TWO-HANDED: evaluar tracción bilateral, extensión codo guía adelante
  • ONE-HANDED: evaluar rotación completa hombro dominante, extensión brazo único

═══ CONTEXTO BIOMECÁNICO ADICIONAL ═══
FATIGA DETECTADA: {fatigue_note}
  → Aplicar penalización en últimas repeticiones si hay degradación

POSICIÓN EN CANCHA: {position_note}
  → Ajustar expectativas según posición (cerca/atrás/lado)
  → En SLICE APPROACH: pie delantero adelantado es CORRECTO

{bh_phase_block}

═══ FRAMEWORK DE SCORING ═══

DIMENSIONES (máx 100 total):
  1. Preparacion (0-20): rotación INVERSA de hombros + posición temprana
     - Evaluar: hombros separados, codo guía posicionado, altura correcta
     - Usar ÁNGULOS DE FASE PREPARACIÓN si disponibles

  2. Punto_impacto (0-20): ángulos en contacto con pelota
     - Evaluar: codo {guide_side} contra rango {bh_range_label} ({bh_elbow_opt}), flexión rodilla, rotación cadera
     - CRÍTICO: ajustar por VARIANTE detectada (slice vs topspin tienen métricas diferentes)
     - Delta vs óptimo: 0° = 20pts, +/- 5° = 15pts, +/- 15° = 5pts, > 20° = 0pts

  3. Follow_through (0-20): extensión y deceleración post-impacto
     - Evaluar: brazo completado extensión, hombro terminó rotación, rotación cadera finalizada
     - Usar ÁNGULOS DE FASE FOLLOW-THROUGH si disponibles
     - VARIANTE SLICE: follow short es CORRECTO, no es error

  4. Posicion_pies (0-20): estabilidad base y giro
     - Indicador: movimiento de pies, balance final, apertura de cadera
     - SLICE APPROACH: pie delantero adelantado NO es error

  5. Ritmo_cadencia (0-10): fluidez de cadena cinética
     - Indicador: transición suave cadera→hombro→codo, sin pausas
     - Usar std_dev para validar: std > 20° = fragmentado (< 5pts)

  6. Potencia_pelota (0-10): velocidad resultante relativa a variante
     - Indicador: ball_speed máx y variación
     - SLICE: velocidad menor es NORMAL — NO penalizar
     - TOPSPIN: mayor velocidad esperada
     - {bh_ball_validated and "✓ Ball validated" or "⚠️ NO validado - score = 0"}

═══ INSTRUCCIONES CRÍTICAS ═══

VALIDACIÓN DE FASES:
  • Si "NO DISPONIBLE": ignorar completamente, usar solo ángulos de impacto
  • Si disponibles: evaluar CADA articulación contra su RANGO DE FASE
  • delta_hip_rotation ALTO confirma que cadera lideró → no penalizar cadera en impacto
  • delta_shoulder_rotation BAJO indica golpe "armado" → penalizar preparacion y ritmo

VARIANTE-SPECIFIC SCORING:
  • SLICE: priorizar contacto adelante del cuerpo, extensión del brazo, control
  • TOPSPIN: evaluar rotación completa, velocidad de pelota mayor

CONSISTENCIA EN SCORING:
  • std_elbow (brazo guía) > 20°: preparación inconsistente → penalizar preparacion(-5pts) y punto_impacto(-5pts)
  • std_shoulder > 8°: rotación irregular → penalizar follow_through(-5pts) y potencia_pelota(-2pts)
  • Mencionar explícitamente en justificaciones

NIVEL (basado en total_score):
  0-40: principiante | 41-60: intermedio | 61-80: avanzado | 81-100: experto

{fallback_warning}{ball_validation_note}

JSON EXACTO (sin backticks, valores numéricos reales):
{{"stroke":"backhand","scores":{{"preparacion":{{"score":0,"max":20,"justificacion":"Rotación inversa y posición inicial"}},"punto_impacto":{{"score":0,"max":20,"justificacion":"Incluir delta vs óptimo {bh_range_label}, ajuste por variante"}},"follow_through":{{"score":0,"max":20,"justificacion":"Extensión post-impacto, contexto de variante"}},"posicion_pies":{{"score":0,"max":20,"justificacion":"Estabilidad y giro, contexto de slice approach"}},"ritmo_cadencia":{{"score":0,"max":10,"justificacion":"Fluidez de cadena cinética, mencionar std"}},"potencia_pelota":{{"score":0,"max":10,"justificacion":"Velocidad y consistencia, contexto de variante"}}}},"total_score":0,"nivel":"principiante|intermedio|avanzado|experto","analisis_tecnico":{{"fortalezas":["",""],"debilidades":["",""],"patron_error_principal":"","comparacion_optimo":""}},"metricas_clave":{{"angulo_codo_{guide_abbrev}_impacto":{imp_guide_elbow},"angulo_codo_{guide_abbrev}_promedio":{sum_guide_elbow},"std_codo_{guide_abbrev}":{std_elbow},"flexion_rodilla_{knee_side}_impacto":{imp_knee},"std_rodilla_{knee_side}":{std_knee},"alineacion_hombros":{imp_shoulder},"std_hombros":{std_shoulder},"velocidad_pelota_max":{max_ball_speed},"std_velocidad_pelota":{std_ball},"impactos_analizados":{n_impacts},"variante":"{bh_stroke_variant_upper}"}},"observaciones_detalladas":"","datos_insuficientes":{str(bh_fallback).lower()}}}"""

    msg_json = client.chat.completions.create(
        model=get_model_for_agent("specialist"),
        max_tokens=5000,
        messages=[{"role": "user", "content": prompt}],
    )
    result = parse_json_response(msg_json.choices[0].message.content)

    # ── Narrativa (200-300 palabras) ──────────────────────────
    angle_trust   = _angle_trust_hint(camera_orientation)
    fallback_note = (
        " NOTA: Los datos provienen de promedios globales del video "
        "(el coordinador no identificó frames específicos de backhand), "
        "menciona esta limitación brevemente." if bh_fallback else ""
    )

    msg_narrative = client.chat.completions.create(
        model=get_model_for_agent("specialist"),
        max_tokens=2000,
        messages=[{"role": "user", "content": f"""Eres biomecánico especialista en backhand de tenis.
Escribe el análisis narrativo del backhand en 200-300 palabras en español. Prosa técnica, fluida, sin listas.
VARIANTE ANALIZADA: {bh_stroke_variant_upper} — usa terminología y criterios específicos de esta variante.
{session_ctx}
{angle_trust}{fallback_note}
Score: {result.get('total_score', 0)}/100 | Nivel: {result.get('nivel', '')}
Fortalezas: {result.get('analisis_tecnico', {}).get('fortalezas', [])}
Debilidades: {result.get('analisis_tecnico', {}).get('debilidades', [])}
Error principal: {result.get('analisis_tecnico', {}).get('patron_error_principal', '')}
Métricas: codo_guía_impacto={imp_guide_elbow}° (óptimo {bh_range_label}: {bh_elbow_opt}) (±{std_elbow}°) | rodilla={imp_knee}° (±{std_knee}°) | hombros={imp_shoulder}° (±{std_shoulder}°) | pelota_max={max_ball_speed}px (±{std_ball}px)
Interpreta la consistencia: std_codo_guía={std_elbow}° y std_hombros={std_shoulder}° — menciona si el patrón es reproducible o errático.
Si es SLICE: comenta la eficacia táctica del approach, no solo la biomecánica."""}],
    )
    result["narrativa_seccion"]   = msg_narrative.choices[0].message.content.strip()
    result["datos_insuficientes"] = bh_fallback
    return result



# ─────────────────────────────────────────────────────────────────────────────
# AGENTE SAQUE
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_saque(
    coordinator_data:   dict,
    mediapipe_data:     dict,
    ball_data:          dict,
    camera_orientation: str  = None,
    equipment_used:     dict = None,
    dominant_hand:      str  = None,
    session_type:       str  = "paleteo",
    api_key:            str  = None,
) -> dict:
    """
    Analiza el saque biomecánicamente.

    El saque usa el brazo dominante como golpe, igual que el forehand,
    pero los rangos óptimos son distintos:
      - Codo dominante en impacto: 150-170° (extensión máxima, no 90-120°)
      - Rodilla dominante en trophy: 120-140° (más flexionada que en groundstrokes)

    Inputs clave desde coordinator_data:
      - stroke_stats["saque"]     : avg + std_dev pre-calculados
      - impact_frames             : impactos clasificados por golpe
      - fatigue_by_stroke["saque"]: degradación técnica por fatiga
      - player_position           : contexto táctico de posición en cancha
      - data_quality.impact_validation["saque"]: ball_validated flag
    """
    import os

    from helpers import (
        format_camera_context,
        format_equipment_context,
        format_session_context,
        parse_json_response,
        get_stroke_frames_or_fallback,
        get_openrouter_client,
        get_model_for_agent,
    )

    _api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    client   = get_openrouter_client(_api_key)

    camera_ctx    = format_camera_context(camera_orientation)
    equipment_ctx = format_equipment_context(equipment_used, dominant_hand)
    session_ctx   = format_session_context(session_type)

    # ── Lateralidad ──────────────────────────────────────────
    is_lefty   = dominant_hand == "left"
    dom_side   = "izquierdo" if is_lefty else "derecho"
    dom_abbrev = "izq"       if is_lefty else "der"
    elbow_key  = "left_elbow"  if is_lefty else "right_elbow"
    knee_key   = "left_knee"   if is_lefty else "right_knee"
    hip_key    = "left_hip"    if is_lefty else "right_hip"

    # ── Frames filtrados por golpe ────────────────────────────
    sq_frames, sq_fallback = get_stroke_frames_or_fallback(
        coordinator_data, mediapipe_data, "saque"
    )
    impact_frames = coordinator_data.get("impact_frames", [])
    sq_impacts    = [
        f for f in impact_frames
        if f.get("stroke_type") in ("saque", "saque_o_smash")
    ]

    fallback_warning = (
        "\n⚠️  ADVERTENCIA: El coordinador no identificó frames específicos de saque. "
        "Se están usando promedios globales del video. "
        "El análisis puede ser menos preciso — incluir esta nota en observaciones_detalladas."
    ) if sq_fallback else ""

    # ── Promedios globales ────────────────────────────────────
    summary   = mediapipe_data.get("summary", {})
    sum_elbow = summary.get(f"avg_{elbow_key}", 0)
    sum_knee  = summary.get(f"avg_{knee_key}",  0)
    sum_hip   = summary.get(f"avg_{hip_key}",   0)

    # ── Ángulos en punto de impacto ───────────────────────────
    if sq_impacts:
        imp_elbow    = round(sum(f["angles"].get(elbow_key, 0) for f in sq_impacts) / len(sq_impacts), 1)
        imp_knee     = round(sum(f["angles"].get(knee_key,  0) for f in sq_impacts) / len(sq_impacts), 1)
        imp_hip      = round(sum(f["angles"].get(hip_key,   0) for f in sq_impacts) / len(sq_impacts), 1)
        imp_shoulder = round(sum(f["shoulder_alignment"] for f in sq_impacts) / len(sq_impacts), 2)
        impact_note  = f"ÁNGULOS EN PUNTO DE IMPACTO DEL SAQUE ({len(sq_impacts)} impactos detectados):"
    elif sq_frames:
        imp_elbow    = round(sum(f["angles"][elbow_key] for f in sq_frames) / len(sq_frames), 1)
        imp_knee     = round(sum(f["angles"][knee_key]  for f in sq_frames) / len(sq_frames), 1)
        imp_hip      = round(sum(f["angles"][hip_key]   for f in sq_frames) / len(sq_frames), 1)
        imp_shoulder = round(sum(f["shoulder_alignment"] for f in sq_frames) / len(sq_frames), 2)
        impact_note  = f"ÁNGULOS EN FRAMES DE SAQUE ({len(sq_frames)} frames filtrados):"
    else:
        imp_elbow, imp_knee, imp_hip = sum_elbow, sum_knee, sum_hip
        imp_shoulder = summary.get("avg_shoulder_alignment", 0)
        impact_note  = "ÁNGULOS GLOBALES (sin frames de saque aislados):"

    # ── Ball speed filtrada por saque ─────────────────────────
    # Solo impactos de saque — no mezclar con groundstrokes.
    if sq_impacts:
        sq_speeds = [f.get("ball_speed_pixels", 0) for f in sq_impacts if f.get("ball_speed_pixels")]
    else:
        sq_speeds = []
    max_ball_speed = max(sq_speeds) if sq_speeds else 0
    avg_ball_speed = round(sum(sq_speeds) / len(sq_speeds), 1) if sq_speeds else 0

    # ── Validación de sincronización pelota-pose ──────────────
    impact_validation  = coordinator_data.get("data_quality", {}).get("impact_validation", {})
    sq_ball_validated  = impact_validation.get("saque", {}).get("ball_validated", True)
    ball_validation_note = (
        "\n⚠️  VALIDACIÓN DE PELOTA: El ball tracker NO confirmó el impacto de saque "
        "(diff_ms > 100ms o sin detección). "
        "El score de potencia_pelota DEBE ser 0 — la medición no es confiable. "
        "Indica esto en la justificación de potencia_pelota."
    ) if not sq_ball_validated else ""

    # ── Consistencia (std_dev), fatiga y posición ─────────────
    sq_stats     = coordinator_data.get("stroke_stats", {}).get("saque", {})
    sq_fatigue   = coordinator_data.get("fatigue_by_stroke", {}).get("saque", {})
    player_pos   = coordinator_data.get("player_position", {})

    std_elbow    = sq_stats.get("std_dom_elbow",          0)
    std_knee     = sq_stats.get("std_dom_knee",           0)
    std_hip      = sq_stats.get("std_dom_hip",            0)
    std_shoulder = sq_stats.get("std_shoulder_alignment", 0)
    std_ball     = sq_stats.get("std_ball_speed",         0)
    n_impacts    = sq_stats.get("n_impacts",              len(sq_impacts))

    fatigue_note  = sq_fatigue.get("narrative", "")
    position_note = player_pos.get("position_note", "")

    consistency_block = _format_consistency_block(
        stroke="saque",
        dom_side=dom_side,
        std_elbow=std_elbow,
        std_knee=std_knee,
        std_hip=std_hip,
        std_shoulder=std_shoulder,
        std_ball=std_ball,
        n_impacts=n_impacts,
    )

    # ── Prompt optimizado para Gemini 2.0 Flash con análisis profundo ──
    prompt = f"""Eres experto biomecánico en saque de tenis. Analiza PROFUNDAMENTE estos datos reales.
Responde SOLO con JSON válido, sin markdown, sin backticks.

═══ CONTEXTO DE SESIÓN ═══
{session_ctx}

{camera_ctx}

{equipment_ctx}

═══ DATOS BIOMECÁNICOS COMPLETOS ═══

PROMEDIOS GLOBALES (sesión completa):
  Codo {dom_side} promedio: {sum_elbow}° | Rango óptimo (extensión máxima): 150-170°
  Rodilla {dom_side} promedio: {sum_knee}° | Rango óptimo (carga trophy): 120-140°
  Cadera {dom_side} promedio: {sum_hip}° | Rango óptimo: 140-160°
  Alineación hombros promedio: {summary.get('avg_shoulder_alignment', 0)}° | Óptimo: < 5°
  Velocidad pelota máx: {max_ball_speed} px/frame | Promedio: {avg_ball_speed} px/frame

PUNTO DE IMPACTO (extensión máxima):
{impact_note}
  Codo {dom_side} en impacto: {imp_elbow}° vs óptimo 150-170° (delta: {abs(imp_elbow - 160) if imp_elbow not in range(150, 171) else 0}°)
  Rodilla {dom_side} en trophy: {imp_knee}° vs óptimo 120-140° (delta: {abs(imp_knee - 130) if imp_knee not in range(120, 141) else 0}°)
  Cadera {dom_side}: {imp_hip}° vs óptimo 140-160° (delta: {abs(imp_hip - 150) if imp_hip not in range(140, 161) else 0}°)
  Alineación hombros: {imp_shoulder}° vs óptimo < 5° (delta: {abs(imp_shoulder)}°)

CONSISTENCIA (Desviación Estándar - indicador de repetibilidad):
  Std Codo {dom_side}: {std_elbow}° ({std_elbow > 15 and "⚠️ EXTENSIÓN IRREGULAR" or std_elbow > 8 and "⚠️ Moderado" or "✓ Consistente"})
  Std Rodilla {dom_side}: {std_knee}° ({std_knee > 15 and "⚠️ TROPHY INCONSISTENTE" or std_knee > 8 and "⚠️ Moderado" or "✓ Consistente"})
  Std Cadera {dom_side}: {std_hip}°
  Std Alineación hombros: {std_shoulder}° ({std_shoulder > 8 and "⚠️ Rotación irregular" or "✓ Alineación consistente"})
  Std Velocidad pelota: {std_ball}° (variación de potencia)
  Número de impactos analizados: {n_impacts}

{consistency_block}

═══ CONTEXTO BIOMECÁNICO ADICIONAL ═══
FATIGA DETECTADA: {fatigue_note}
  → Aplicar penalización en últimas repeticiones si hay degradación

POSICIÓN EN CANCHA: {position_note}

═══ FRAMEWORK DE SCORING ═══

DIMENSIONES (máx 100 total):
  1. Preparacion_toss (0-20): lanzamiento de pelota y posición inicial
     - Evaluar: altura de toss, consistencia de ubicación, sincronización
     - Usar datos de posición de codo en fase preparación si disponibles

  2. Carga_trophy (0-20): "trophy position" con rodillas flexionadas
     - Evaluar: profundidad de rodillas (rango: 120-140°), posición de codo
     - CRÍTICO: rodillas más flexionadas = mejor acumulación de energía
     - Std_knee > 15° = muy inconsistente (< 5pts)

  3. Punto_impacto (0-20): extensión máxima en contacto con pelota
     - Evaluar: codo {dom_side} contra rango 150-170°, altura de lanzamiento, altura de brazo
     - CRÍTICO: extensión máxima es característica del saque
     - Delta vs óptimo: 0° = 20pts, +/- 5° = 15pts, +/- 15° = 5pts, > 20° = 0pts
     - Std_elbow > 15° = muy inconsistente (restar 5-10pts)

  4. Follow_through (0-20): continuidad del movimiento post-impacto
     - Evaluar: descarga del brazo, rotación de cadera completada
     - Evaluar: uso de pesos corporales, traslación hacia adelante

  5. Ritmo_cadencia (0-10): fluidez del servicio completo
     - Indicador: transición suave entre toss, carga, extensión, follow
     - Std_shoulder > 8° = rotación irregular (< 5pts)
     - Sincronización trophy-codo-pelota

  6. Potencia_pelota (0-10): velocidad resultante del servicio
     - Indicador: ball_speed máx y variación
     - Contexto de nivel: principiante < 5px/f, intermedio 5-8, avanzado > 8
     - {sq_ball_validated and "✓ Ball validated" or "⚠️ NO validado - score = 0"}

═══ INSTRUCCIONES CRÍTICAS ═══

SAQUE-SPECIFIC METRICS:
  • Extensión de codo: 150-170° es el rango correcto (no 90-120° como groundstrokes)
  • Trophy depth: rodillas 120-140° indica carga correcta
  • No penalizar por angles si std es alto en saque — es golpe más variable

CONSISTENCIA EN SCORING:
  • std_elbow > 15°: extensión muy variable → penalizar punto_impacto(-5 a -10pts)
  • std_knee > 15°: trophy inconsistente → penalizar carga_trophy(-5pts) y ritmo_cadencia(-3pts)
  • std_shoulder > 8°: rotación irregular → penalizar ritmo_cadencia(-3 a -5pts)
  • Mencionar explícitamente en justificaciones

NIVEL (basado en total_score):
  0-40: principiante | 41-60: intermedio | 61-80: avanzado | 81-100: experto

{fallback_warning}{ball_validation_note}

JSON EXACTO (sin backticks, valores numéricos reales):
{{"stroke":"saque","scores":{{"preparacion_toss":{{"score":0,"max":20,"justificacion":"Lanzamiento y sincronización inicial"}},"carga_trophy":{{"score":0,"max":20,"justificacion":"Profundidad de rodillas, flexión, energía acumulada"}},"punto_impacto":{{"score":0,"max":20,"justificacion":"Incluir delta vs óptimo 150-170°, mencionar std_elbow"}},"follow_through":{{"score":0,"max":20,"justificacion":"Descarga y continuidad post-impacto"}},"ritmo_cadencia":{{"score":0,"max":10,"justificacion":"Fluidez y sincronización del servicio completo, mencionar std_shoulder"}},"potencia_pelota":{{"score":0,"max":10,"justificacion":"Velocidad y contexto de nivel"}}}},"total_score":0,"nivel":"principiante|intermedio|avanzado|experto","analisis_tecnico":{{"fortalezas":["",""],"debilidades":["",""],"patron_error_principal":"","comparacion_optimo":""}},"metricas_clave":{{"extension_codo_{dom_abbrev}_impacto":{imp_elbow},"extension_codo_{dom_abbrev}_promedio":{sum_elbow},"std_codo_{dom_abbrev}":{std_elbow},"flexion_rodilla_{dom_abbrev}_carga":{imp_knee},"std_rodilla_{dom_abbrev}":{std_knee},"alineacion_hombros":{imp_shoulder},"std_hombros":{std_shoulder},"velocidad_pelota_max":{max_ball_speed},"std_velocidad_pelota":{std_ball},"impactos_analizados":{n_impacts}}},"observaciones_detalladas":"","datos_insuficientes":{str(sq_fallback).lower()}}}"""

    msg_json = client.chat.completions.create(
        model=get_model_for_agent("specialist"),
        max_tokens=5000,
        messages=[{"role": "user", "content": prompt}],
    )
    result = parse_json_response(msg_json.choices[0].message.content)

    # ── Narrativa (200-300 palabras) ──────────────────────────
    angle_trust   = _angle_trust_hint(camera_orientation, stroke="saque")
    fallback_note = (
        " NOTA: Los datos provienen de promedios globales del video, "
        "menciona esta limitación brevemente." if sq_fallback else ""
    )

    msg_narrative = client.chat.completions.create(
        model=get_model_for_agent("specialist"),
        max_tokens=2000,
        messages=[{"role": "user", "content": f"""Eres biomecánico especialista en saque de tenis.
Escribe el análisis narrativo del saque en 200-300 palabras en español. Prosa técnica, fluida, sin listas.
{session_ctx}
{angle_trust}{fallback_note}
Score: {result.get('total_score', 0)}/100 | Nivel: {result.get('nivel', '')}
Fortalezas: {result.get('analisis_tecnico', {}).get('fortalezas', [])}
Debilidades: {result.get('analisis_tecnico', {}).get('debilidades', [])}
Error principal: {result.get('analisis_tecnico', {}).get('patron_error_principal', '')}
Métricas: codo_impacto={imp_elbow}° (±{std_elbow}°) | rodilla_trophy={imp_knee}° (±{std_knee}°) | hombros={imp_shoulder}° (±{std_shoulder}°) | pelota_max={max_ball_speed}px (±{std_ball}px)
Fatiga: {fatigue_note}"""}],
    )
    result["narrativa_seccion"]   = msg_narrative.choices[0].message.content.strip()
    result["datos_insuficientes"] = sq_fallback
    return result

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS INTERNOS
# ─────────────────────────────────────────────────────────────────────────────

# Rangos óptimos de codo en impacto por grip de forehand.
# Basados en biomecánica real de cada técnica — no en rangos ATP genéricos.
_FOREHAND_GRIP_RANGES = {
    "eastern":      {"elbow": (80,  112), "label": "Eastern"},
    "semi_western": {"elbow": (100, 138), "label": "Semi-western"},
    "western":      {"elbow": (130, 165), "label": "Western/Full-western"},
    "unknown":      {"elbow": (90,  130), "label": "grip no determinado — referencia general"},
}

# Rangos óptimos de codo guía en impacto para backhand, por variante técnica.
# En topspin el codo se mantiene cerrado; en slice se extiende.
_BACKHAND_VARIANT_RANGES = {
    "topspin": {
        "two_handed": {"elbow": (90,  125), "label": "BH a dos manos topspin"},
        "one_handed": {"elbow": (145, 170), "label": "BH a una mano topspin"},
        "unknown":    {"elbow": (90,  130), "label": "BH topspin — referencia general"},
    },
    "slice": {
        "two_handed": {"elbow": (120, 155), "label": "BH a dos manos slice"},
        "one_handed": {"elbow": (130, 165), "label": "BH a una mano slice"},
        "unknown":    {"elbow": (120, 160), "label": "BH slice — referencia general"},
    },
}


def _get_forehand_elbow_range(grip_type: str) -> tuple[tuple[int, int], str]:
    """
    Retorna (rango_optimo, label_descriptivo) para el codo en impacto según grip.
    Siempre retorna un valor válido — fallback a 'unknown' si el grip no se reconoce.
    """
    data = _FOREHAND_GRIP_RANGES.get(grip_type, _FOREHAND_GRIP_RANGES["unknown"])
    return data["elbow"], data["label"]


def _get_backhand_elbow_range(grip_type: str, variant: str) -> tuple[tuple[int, int], str]:
    """
    Retorna (rango_optimo, label_descriptivo) para el codo guía según grip y variante.
    variant: "topspin" | "slice"
    grip_type: "two_handed" | "one_handed" | "unknown"
    """
    variant_map = _BACKHAND_VARIANT_RANGES.get(variant, _BACKHAND_VARIANT_RANGES["topspin"])
    data        = variant_map.get(grip_type, variant_map["unknown"])
    return data["elbow"], data["label"]


def _format_phase_block(phase_data: dict, grip_label: str = "") -> str:
    """
    Formatea el análisis de fases para inyectar en el prompt del especialista.

    Si phase_data_available=False, devuelve solo el fallback_note para que
    el especialista sepa que debe operar en modo de impacto único.

    Si phase_data_available=True, construye un bloque estructurado con:
      - Ángulos por fase con interpretación cualitativa
      - Deltas de cadena cinética (rotación de hombros, cadera, aceleración)
      - Nota sobre calidad de datos (low_confidence por fase)

    El bloque es intencionalmente explícito: el LLM debe ver los números
    con contexto de fase para no comparar la cadera en preparación (158°)
    contra el rango de impacto (140-160°) y marcarla incorrectamente.
    """
    if not phase_data:
        return ""

    if not phase_data.get("phase_data_available", False):
        return (
            "\nANÁLISIS DE FASES: NO DISPONIBLE\n"
            f"  {phase_data.get('fallback_note', '')}\n"
            "  → INSTRUCCIÓN: Evaluar exactamente igual que si no hubiera análisis de fases.\n"
            "  → NO usar rangos por fase. NO inferir ángulos de preparación desde promedios globales.\n"
            "  → Usar solo los ángulos de impacto reportados en la sección anterior."
        )

    angles   = phase_data.get("angles", {})
    deltas   = phase_data.get("deltas", {})
    computed = phase_data.get("phases_computed", [])
    insuf    = phase_data.get("phases_insufficient", [])

    lines = ["\nANÁLISIS POR FASE BIOMECÁNICA:"]
    lines.append(
        "  ⚠️  CRÍTICO: Evaluar cada articulación contra el rango de SU FASE, "
        "no contra un rango único. La cadera a 158° en PREPARACIÓN es correcta "
        "(carga máxima); la misma cadera a 102° en IMPACTO significa que ya completó "
        "la rotación — también puede ser correcto."
    )

    # Referencia de rangos óptimos por fase (independiente del grip)
    _PHASE_REFS = {
        "preparacion": {
            "dom_hip":            (140, 170, "carga de cadera — más alto = mejor carga"),
            "dom_knee":           (125, 155, "flexión de rodilla — carga para propulsión"),
            "dom_elbow":          (70,  120, "brazo preparado — codo cerrado para generar velocidad"),
            "shoulder_alignment": (None, 10, "hombros cerrados — mayor el ángulo, mejor rotación pendiente"),
        },
        "aceleracion": {
            "dom_hip":            (120, 165, "cadera en transición"),
            "dom_knee":           (128, 158, "extensión de piernas"),
            "dom_elbow":          (90,  145, "aceleración del brazo"),
            "shoulder_alignment": (None,  8, "hombros abriéndose"),
        },
        "impacto": {
            "dom_knee":           (128, 158, "base estable en contacto"),
            "dom_hip":            (100, 155, "cadera rotada o en rotación — amplio rango aceptable"),
            "shoulder_alignment": (None,  5, "hombros alineados con la red"),
            # dom_elbow se omite aquí — depende del grip, ya está en el prompt principal
        },
        "followthrough": {
            "dom_hip":            ( 90, 155, "completar rotación"),
            "dom_elbow":          (130, 175, "extensión post-impacto"),
            "shoulder_alignment": (None, 10, "apertura de hombros"),
        },
    }

    _PHASE_LABELS = {
        "preparacion":   "PREPARACIÓN (backswing/carga)",
        "aceleracion":   "ACELERACIÓN (forward swing)",
        "impacto":       "IMPACTO (contacto con pelota)",
        "followthrough": "FOLLOW-THROUGH (post-impacto)",
    }

    for phase in ("preparacion", "aceleracion", "impacto", "followthrough"):
        pdata = angles.get(phase)
        if not pdata:
            continue

        is_low_conf = pdata.get("low_confidence", False)
        conf_tag    = " ⚠️ datos parciales" if is_low_conf else ""
        n           = pdata.get("n_frames", 0)
        rejected    = pdata.get("frames_rejected", 0)

        lines.append(f"\n  [{_PHASE_LABELS[phase]} — {n} frames{conf_tag}]")
        if rejected:
            lines.append(f"    ({rejected} frames rechazados por baja visibilidad)")

        refs = _PHASE_REFS.get(phase, {})

        for key, label in [
            ("dom_hip",           "Cadera dom."),
            ("dom_knee",          "Rodilla dom."),
            ("dom_elbow",         "Codo dom."),
            ("shoulder_alignment","Hombros alin."),
        ]:
            val = pdata.get(key)
            if val is None or val == 0:
                continue

            ref = refs.get(key)
            if ref:
                lo, hi, desc = ref
                if lo is not None:
                    in_range = lo <= val <= hi
                    status   = "✓" if in_range else ("⚠" if abs(val - (lo+hi)/2) < (hi-lo) else "✗")
                    range_str = f"(rango fase: {lo}-{hi}° — {desc})"
                else:
                    # Solo límite superior (shoulder_alignment)
                    in_range = val <= hi
                    status   = "✓" if in_range else "⚠"
                    range_str = f"(óptimo: <{hi}° — {desc})"
                lines.append(f"    {label}: {val}° {status}  {range_str}")
            else:
                lines.append(f"    {label}: {val}°")

        # En impacto, añadir nota sobre el codo con referencia al grip
        if phase == "impacto" and grip_label:
            imp_elbow = pdata.get("dom_elbow")
            if imp_elbow:
                lines.append(
                    f"    Codo dom.: {imp_elbow}°  "
                    f"→ ver rango de {grip_label} en sección de ángulos de impacto arriba"
                )

    # ── Deltas de cadena cinética ──────────────────────────────────────────
    if deltas and deltas.get("rotation_quality") != "sin_datos":
        lines.append(f"\n  DELTAS DE CADENA CINÉTICA:")
        d_sh  = deltas.get("delta_shoulder_rotation")
        d_hip = deltas.get("delta_hip_rotation")
        d_el  = deltas.get("delta_elbow_extension")

        if d_sh  is not None: lines.append(f"    Δ rotación hombros (prep→impacto): {d_sh:+.1f}°")
        if d_hip is not None: lines.append(f"    Δ rotación cadera  (prep→impacto): {d_hip:+.1f}°")
        if d_el  is not None: lines.append(f"    Δ extensión codo   (prep→follow):  {d_el:+.1f}°")
        lines.append(f"    → {deltas.get('kinetic_chain_note', '')}")

    if insuf:
        lines.append(f"\n  (Fases con datos insuficientes: {', '.join(insuf)})")

    return "\n".join(lines)


def _format_consistency_block(
    stroke:       str,
    dom_side:     str,
    std_elbow:    float,
    std_knee:     float,
    std_hip:      float,
    std_shoulder: float,
    std_ball:     float,
    n_impacts:    int,
    is_guide_arm: bool = False,
) -> str:
    """
    Genera el bloque de CONSISTENCIA para inyectar en el prompt.
    Incluye interpretación automática de los umbrales para guiar al LLM.
    """
    arm_label = "brazo guía" if is_guide_arm else "brazo de golpe"

    def _interpret_elbow(std: float) -> str:
        if std == 0:    return "sin datos suficientes"
        if std < 10:    return "swing muy consistente ✓"
        if std < 20:    return "consistencia aceptable"
        return          "swing errático — inconsistencia alta ⚠️"

    def _interpret_shoulder(std: float) -> str:
        if std == 0:    return "sin datos suficientes"
        if std < 5:     return "rotación de tronco muy estable ✓"
        if std < 8:     return "rotación aceptable"
        return          "rotación inconsistente — pérdida de cadena cinética ⚠️"

    def _interpret_knee(std: float) -> str:
        if std == 0:    return "sin datos suficientes"
        if std < 8:     return "base estable ✓"
        if std < 15:    return "base aceptable"
        return          "inestabilidad en piernas ⚠️"

    # Aviso de muestra pequeña: con n < 8 el std_dev es orientativo, no diagnóstico
    sample_warning = ""
    if n_impacts < 8:
        elbow_thresh = 25 if n_impacts < 8 else 20
        sample_warning = (
            f"\n  ⚠️  MUESTRA PEQUEÑA (n={n_impacts}): con menos de 8 impactos el std_dev "
            f"refleja varianza muestral tanto como inconsistencia real. "
            f"Solo penalizar consistencia si std_elbow > {elbow_thresh}° o std_shoulder > 10°. "
            f"No penalizar por std moderados (10-{elbow_thresh - 1}°) con n pequeño."
        )

    lines = [
        f"CONSISTENCIA BIOMECÁNICA ({n_impacts} impactos de {stroke} analizados):",
        f"  Codo {dom_side} ({arm_label}): ±{std_elbow}° — {_interpret_elbow(std_elbow)}",
        f"    → referencia: <10°=consistente | 10-20°=aceptable | >20°=errático",
        f"  Rodilla {dom_side}: ±{std_knee}° — {_interpret_knee(std_knee)}",
        f"  Cadera {dom_side}: ±{std_hip}°",
        f"  Hombros (alineación): ±{std_shoulder}° — {_interpret_shoulder(std_shoulder)}",
        f"    → referencia: <5°=estable | 5-8°=aceptable | >8°=inconsistente",
        f"  Velocidad pelota: ±{std_ball} px/frame",
        f"    → referencia: std_ball alto con avg_ball bajo = potencia inconsistente",
    ]
    if sample_warning:
        lines.append(sample_warning)
    return "\n".join(lines)


def _angle_trust_hint(camera_orientation: str | None, stroke: str = "groundstroke") -> str:
    """Hint de confianza angular según ángulo de cámara y tipo de golpe, para la narrativa."""
    if stroke == "saque":
        if camera_orientation and "Lateral" in camera_orientation:
            return (
                "Vista lateral — ángulo ideal para el saque. "
                "Confía en extensión de codo dominante, carga de rodillas (trophy) y arco completo del swing. "
                "Precaución: la rotación de tronco puede aparecer comprimida y el brazo de lanzamiento ocluirse en impacto. "
            )
        if camera_orientation and "Fondo" in camera_orientation:
            return "Confía en la alineación de hombros y simetría de la cadena cinética. "
        return "Interpreta los ángulos con cautela — el ángulo de cámara puede distorsionar la perspectiva del saque. "

    # Groundstrokes (forehand / backhand)
    if camera_orientation and "Lateral" in camera_orientation:
        return (
            "Confía en ángulos sagitales (codo, rodilla, torso) — cámara lateral. "
            "Ten en cuenta posible oclusión del brazo trasero y compresión de rotación de cadera. "
        )
    if camera_orientation and "Fondo" in camera_orientation:
        return "Confía principalmente en la alineación de hombros y simetría corporal. "
    return "Interpreta los ángulos con cautela dado que el ángulo de cámara puede distorsionar la perspectiva. "
