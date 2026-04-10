"""
TennisAI — Agente Coordinador
══════════════════════════════
Responsabilidades:
  1. Recibir datos slim de MediaPipe, YOLO y Ball Tracker
  2. Recibir pre-cómputo Python (stroke_stats, tactical_context, data_quality)
  3. Clasificar frames por tipo de golpe Y por fase biomecánica
  4. Decidir qué agentes activar, con nivel de confianza por golpe
  5. Retornar coordinator_result enriquecido para los agentes especializados

Imports esperados en agents_pipeline.py:
    from agent_coordinator import run_agent_coordinator

El decorador @app.function vive en agents_pipeline.py.
Este módulo es Python puro, testeable localmente sin Modal.

Test local desde Colab:
    from agent_coordinator import run_agent_coordinator
    result = run_agent_coordinator(
        mediapipe_data=mediapipe_slim,
        yolo_data=yolo_slim,
        ball_data=ball_slim,
        session_type="paleteo",
        camera_orientation="Lateral-Derecha",
        equipment_used={"brand": "Wilson", "model": "Pro Staff", "head_size": "97"},
        dominant_hand="right",
        stroke_stats=stroke_stats,
        tactical_context=tactical_context,
        data_quality=data_quality,
        api_key="sk-ant-...",
    )
"""

import json


# ─── FUNCIÓN PRINCIPAL (lógica pura, sin Modal) ──────────────

def run_agent_coordinator(
    mediapipe_data:   dict,
    yolo_data:        dict,
    ball_data:        dict,
    session_type:     str,
    camera_orientation: str  = None,
    equipment_used:   dict   = None,
    dominant_hand:    str    = None,
    # Pre-cómputo Python (coordinator_precompute.py)
    stroke_stats:     dict   = None,
    tactical_context: dict   = None,
    data_quality:     dict   = None,
    # Solo para test local; en Modal se lee de os.environ
    api_key:          str    = None,
) -> dict:
    """
    Lógica del agente coordinador.
    Retorna coordinator_result con:
      - active_agents          : lista de golpes a analizar
      - agent_confidence       : confianza + evidencia por golpe
      - impact_frames          : impactos clasificados con ángulos
      - frames_by_stroke       : frames por golpe Y por fase biomecánica
      - stroke_stats           : pass-through del pre-cómputo
      - tactical_context       : pass-through del pre-cómputo
      - data_quality_notes     : observaciones libres del LLM
      - general_observations   : resumen general del LLM
      - camera_quality         : "buena" | "regular" | "mala"
      - camera_angle_detected  : "lateral" | "detras" | "frontal" | "desconocido"
    """
    import os

    # ── Importar helpers (disponibles en el mismo directorio en Modal) ──
    from helpers import (
        format_camera_context,
        format_equipment_context,
        format_session_context,
        parse_json_response,
        get_openrouter_client,
        get_model_for_agent,
    )

    _api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    client   = get_openrouter_client(_api_key)

    camera_ctx    = format_camera_context(camera_orientation)
    equipment_ctx = format_equipment_context(equipment_used, dominant_hand)
    session_ctx   = format_session_context(session_type)

    # ── Enriquecer session_ctx con implicación táctica ───────
    # La implicación táctica se integra directamente al contexto de sesión
    # para que el LLM la aplique al decidir qué agentes activar y con
    # qué leniencia, no solo como dato informativo separado.
    if tactical_context and tactical_context.get("implication"):
        session_ctx += (
            f"\n\nCONTEXTO TÁCTICO ADICIONAL: {tactical_context['implication']}"
            f"\n  → Aplica esta información al decidir qué agentes activar"
            f" y al calibrar los umbrales de evidencia mínima por tipo de golpe."
        )

    # ── Bloques de pre-cómputo para el prompt ────────────────
    stroke_stats_block    = _format_stroke_stats_block(stroke_stats, dominant_hand)
    tactical_block        = _format_tactical_context_block(tactical_context)
    data_quality_block    = _format_data_quality_block(data_quality)

    # ── Frames con pelota (solo los detectados, máx 50) ──────
    ball_frames_detected  = [
        f for f in ball_data.get("frames", []) if f.get("ball_detected")
    ][:50]

    # ── Prompt optimizado para DeepSeek V3.2 con máximo contexto ──
    prompt = f"""Eres el coordinador de análisis biomecánico de tenis.
Tarea: Clasificar la sesión con razonamiento profundo basado en contexto completo.

═══ CONTEXTO DE SESIÓN ═══
{session_ctx}
{camera_ctx}
{equipment_ctx}

═══ DATOS BIOMECÁNICOS MEDIAPIPE (completo) ═══
Duración: {mediapipe_data['duration_seconds']}s | Frames analizados: {mediapipe_data['frames_analyzed']}

Estadísticas globales por articulación:
  CODO (dominante): promedio {mediapipe_data['summary']['avg_right_elbow']}° | rango: mín-máx
  CODO (no dominante): promedio {mediapipe_data['summary']['avg_left_elbow']}°
  RODILLA (dominante): promedio {mediapipe_data['summary']['avg_right_knee']}°
  RODILLA (no dominante): promedio {mediapipe_data['summary']['avg_left_knee']}°
  CADERA (dominante): promedio {mediapipe_data['summary']['avg_right_hip']}°
  CADERA (no dominante): promedio {mediapipe_data['summary']['avg_left_hip']}°
  HOMBROS (alineación): promedio {mediapipe_data['summary']['avg_shoulder_alignment']}°

Muestra de frames clave (15 frames distribuidos):
{json.dumps(mediapipe_data.get('frames', [])[:15], indent=2)}

═══ DETECCIÓN YOLO (visión) ═══
Tasa de detección jugador: {yolo_data['detection_rate_percent']}%
Hints de golpes por tipo:
{json.dumps(yolo_data['stroke_hints_summary'], indent=2)}

═══ SEGUIMIENTO PELOTA ═══
Tasa de detección: {ball_data['ball_detection_rate_percent']}%
Velocidad máxima: {ball_data['max_ball_speed_pixels']} px/frame
Eventos detectados: {len(ball_frames_detected)}

Detalle de eventos de pelota (primeros 50):
{json.dumps(ball_frames_detected[:50], indent=2)}

═══ ANÁLISIS STATS PRE-COMPUTADO ═══
{stroke_stats_block}

═══ CONTEXTO TÁCTICO ═══
{tactical_block}

═══ EVALUACIÓN CALIDAD DE DATOS ═══
{data_quality_block}

═══ LÓGICA DE RAZONAMIENTO ═══

PASO 1 - ACTIVACIÓN DE AGENTES:
Analiza evidencia CONVERGENTE de 3 fuentes:
  • YOLO: hints de golpes por tipo (count ≥ 3 = señal fuerte)
  • Ball tracking: impactos asignables por velocidad y timing (≥ 2 = señal fuerte)
  • MediaPipe ángulos: patrones de codo consistentes con tipo de golpe
Activa golpe SI: (YOLO ≥ 3) O (ball impacts ≥ 2) O (ambos presentes aunque menores)
Documenta TODAS las fuentes de evidencia en agent_confidence.

PASO 2 - CLASIFICACIÓN DE FASES (para cada golpe activo):
Usa CODO DOMINANTE como señal principal de movimiento.

RANGOS DE REFERENCIA POR FASE (codo dominante):
  preparacion  : 30-70°   → ángulo cerrado (armado), mínimo local
  aceleracion  : 70-120°  → ángulo creciente (despliegue hacia impacto)
  impacto      : 120-160° → extensión máxima local O ball_speed pico
  followthrough: 80-140°  → ángulo descendente (deceleración post-impacto)
  [SAQUE] impacto: 150-170° → extensión mayor que groundstrokes, NO aplicar rango 120-160°

Proporciona ÍNDICES enteros secuenciales de frames (campo "frame" en MediaPipe).
Las listas pueden estar vacías si no hay frames claros para esa fase.
Asegúrate que fases no se superpongan y sean progresivas en el tiempo.

PASO 3 - IMPACTOS CON PELOTA:
Cruza ball_frames (ball_speed > 0) con frames MediaPipe cercanos (diff_ms < 100ms).
Para cada impacto válido:
  - stroke_type: asigna basado en YOLO hints + ángulos MediaPipe contextuales
  - Incluye ángulos completos de articulaciones en punto de impacto
  - Captura alineación de hombros

═══ SALIDA JSON (estructura exacta) ═══
{{
  "session_type": "{session_type}",
  "camera_quality": "buena|regular|mala",
  "camera_angle_detected": "lateral|detras|frontal|desconocido",

  "active_agents": ["forehand", "backhand"],

  "agent_confidence": {{
    "forehand": {{"activate": true, "confidence": 0.85, "evidence": "YOLO: 7 hints | Ball impacts: 2 eventos | Ángulos codo 45-95° consistentes con derecha"}},
    "backhand": {{"activate": true, "confidence": 0.65, "evidence": "YOLO: 4 hints | Ball impacts: 1 evento | Ángulos codo 50-100° presentes"}},
    "saque": {{"activate": false, "confidence": 0.2, "evidence": "YOLO: <3 hints | Ball impacts: 0 | Altura de codo inconsistente con saque"}}
  }},

  "impact_frames": [
    {{"timestamp": 1.23, "frame": 30, "stroke_type": "forehand", "ball_speed": 8.5, "diff_ms": 15, "right_elbow": 85.2, "left_elbow": 120.5, "right_knee": 95.0, "left_knee": 105.0, "right_hip": 110.0, "left_hip": 115.0, "shoulder_alignment": 5.2}}
  ],

  "frames_by_stroke": {{
    "forehand": {{"preparacion": [8, 9, 10, 11], "aceleracion": [12, 13, 14, 15, 16], "impacto": [17, 18], "followthrough": [19, 20, 21, 22]}},
    "backhand": {{"preparacion": [40, 41], "aceleracion": [42, 43, 44], "impacto": [45], "followthrough": [46, 47]}},
    "saque": {{"preparacion": [], "aceleracion": [], "impacto": [], "followthrough": []}}
  }},

  "data_quality_notes": "Observaciones sobre confiabilidad: cobertura de poses, oclusiones, detección de pelota",
  "general_observations": "Patrones generales: lateralidad, postura, movimientos repetitivos"
}}"""

    # ── Llamada al LLM ────────────────────────────────────────
    message = client.chat.completions.create(
        model=get_model_for_agent("coordinator"),
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )

    result = parse_json_response(message.choices[0].message.content)

    # ── Validación post-LLM: reconciliar active_agents con agent_confidence ──
    # El LLM puede ser inconsistente: marcar activate=true en agent_confidence
    # pero no incluir el golpe en active_agents, o viceversa.
    # Python es la fuente de verdad: si confidence < 0.5, el golpe se desactiva
    # independientemente de lo que el LLM haya puesto en active_agents.
    result = _reconcile_active_agents(result, impact_frames=result.get("impact_frames", []))

    # ── Pass-through de pre-cómputo al resultado ─────────────
    # Los agentes especializados leen estos campos directamente
    # desde coordinator_result, no necesitan recalcularlos.
    if stroke_stats:
        result["stroke_stats"] = stroke_stats
    if tactical_context:
        result["tactical_context"] = tactical_context

    # data_quality se enriquece después con sync_impact_quality_check
    # en run_agents_pipeline; aquí solo lo dejamos disponible si ya existe.
    if data_quality:
        result.setdefault("data_quality", {}).update(data_quality)

    return result


# ─── HELPERS DE FORMATEO PARA EL PROMPT ──────────────────────

def _format_stroke_stats_block(stroke_stats: dict | None, dominant_hand: str | None) -> str:
    """
    Convierte el output de compute_stroke_stats en texto legible para el prompt.
    Destaca std_shoulder_alignment como indicador de consistencia de rotación.
    """
    if not stroke_stats:
        return "══ ESTADÍSTICAS POR GOLPE ══\nNo disponibles (pre-cómputo no ejecutado)."

    hand_label = "zurdo" if dominant_hand == "left" else "diestro"
    lines = [f"══ ESTADÍSTICAS POR GOLPE (pre-calculadas, mano {hand_label}) ══"]

    for stroke, stats in stroke_stats.items():
        n        = stats.get("n_frames", 0)
        rejected = stats.get("low_quality_frames_rejected", 0)
        lines.append(f"\n{stroke.upper()} — {n} frames válidos ({rejected} rechazados por baja visibilidad)")
        lines.append(
            f"  Codo dominante : avg={stats.get('avg_dom_elbow', 0)}°  "
            f"std={stats.get('std_dom_elbow', 0)}°  "
            f"[std alto = inconsistencia en el swing]"
        )
        lines.append(
            f"  Rodilla dom.   : avg={stats.get('avg_dom_knee', 0)}°  "
            f"std={stats.get('std_dom_knee', 0)}°"
        )
        lines.append(
            f"  Cadera dom.    : avg={stats.get('avg_dom_hip', 0)}°  "
            f"std={stats.get('std_dom_hip', 0)}°"
        )
        lines.append(
            f"  Hombros alin.  : avg={stats.get('avg_shoulder_alignment', 0)}°  "
            f"std={stats.get('std_shoulder_alignment', 0)}°  "
            f"[std alto = rotación inconsistente = pérdida de potencia]"
        )
        if stroke == "backhand":
            lines.append(
                f"  Codo guía      : avg={stats.get('avg_guide_elbow', 0)}°  "
                f"std={stats.get('std_guide_elbow', 0)}°"
            )

    return "\n".join(lines)


def _format_tactical_context_block(tactical_context: dict | None) -> str:
    """
    Convierte el output de compute_tactical_context en texto para el prompt.
    """
    if not tactical_context:
        return "══ CONTEXTO TÁCTICO ══\nNo disponible."

    dist = tactical_context.get("stroke_distribution", {})
    lines = [
        "══ CONTEXTO TÁCTICO (pre-calculado desde YOLO) ══",
        f"Posición dominante : {tactical_context.get('dominant_position', 'desconocida')}",
        f"Aproximaciones red : {tactical_context.get('net_approaches', 0)}",
        f"Distribución golpes: "
        f"forehand {round(dist.get('forehand', 0)*100)}% | "
        f"backhand {round(dist.get('backhand', 0)*100)}% | "
        f"saque {round(dist.get('saque', 0)*100)}%",
        f"Implicación táctica: {tactical_context.get('implication', '')}",
    ]
    return "\n".join(lines)


def _format_data_quality_block(data_quality: dict | None) -> str:
    """
    Convierte el output de build_data_quality_report en texto para el prompt.
    El coordinador usa esto para saber qué fuentes son confiables antes de clasificar.
    """
    if not data_quality:
        return "══ CALIDAD DE DATOS ══\nNo disponible."

    sync_pct  = round(data_quality.get("ball_sync_rate", 0) * 100)
    mp_pct    = round(data_quality.get("mediapipe_coverage", 0) * 100)
    lowvis    = round(data_quality.get("low_visibility_frames_pct", 0) * 100)
    score     = data_quality.get("overall_quality_score", 0)

    lines = [
        "══ CALIDAD DE DATOS (pre-calculada) ══",
        f"MediaPipe coverage  : {mp_pct}%  ({lowvis}% frames con visibilidad baja)",
        f"Ball sync rate      : {sync_pct}%  (impactos pelota+pose dentro de 100ms)",
        f"Impactos totales    : {data_quality.get('impacts_total', 0)}  "
        f"({data_quality.get('impacts_with_ball_sync', 0)} bien sincronizados, "
        f"{data_quality.get('impacts_high_quality', 0)} alta calidad)",
        f"Score calidad global: {score}  (0-1)",
        f"Recomendación       : {data_quality.get('recommendation', '')}",
    ]
    return "\n".join(lines)


# ─── VALIDACIÓN POST-LLM ─────────────────────────────────────

def _reconcile_active_agents(result: dict, impact_frames: list = None) -> dict:
    """
    Reconcilia active_agents con agent_confidence después del parse del LLM.

    Reglas (Python es la fuente de verdad):
      1. Si hay ≥2 impactos reales de ese stroke_type en impact_frames
         → activar siempre, independiente de la confianza del LLM.
      2. Si confidence < 0.5 Y no hay impactos reales → desactivar.
      3. Si activate=False en agent_confidence Y no hay impactos reales → desactivar.
      4. Si agent_confidence está ausente o malformado → conservar active_agents
         tal como vino del LLM (fallback seguro).

    También agrega "deactivation_reason" en agent_confidence para trazabilidad.

    Modifica result in-place y lo retorna.
    """
    CONFIDENCE_THRESHOLD = 0.5

    # Contar impactos reales por stroke_type — override duro sobre el LLM
    impact_counts: dict[str, int] = {"forehand": 0, "backhand": 0, "saque": 0}
    if impact_frames:
        for f in impact_frames:
            st = f.get("stroke_type")
            if st in ("forehand", "backhand"):
                impact_counts[st] += 1
            elif st in ("saque", "saque_o_smash"):
                impact_counts["saque"] += 1
    IMPACT_OVERRIDE_MIN = 2  # con ≥2 impactos reales, activar siempre

    agent_confidence = result.get("agent_confidence", {})
    active_agents    = result.get("active_agents", [])

    # Si el LLM no devolvió agent_confidence, no tocar active_agents
    if not agent_confidence or not isinstance(agent_confidence, dict):
        print("⚠️  agent_confidence ausente o malformado — conservando active_agents del LLM")
        return result

    reconciled_active = []

    for stroke in ("forehand", "backhand", "saque"):
        conf_entry = agent_confidence.get(stroke, {})

        if not isinstance(conf_entry, dict):
            # Entrada malformada: conservar si estaba en active_agents
            if stroke in active_agents:
                reconciled_active.append(stroke)
            continue

        activate   = conf_entry.get("activate", False)
        confidence = conf_entry.get("confidence", 0.0)

        n_impacts = impact_counts.get(stroke, 0)

        if n_impacts >= IMPACT_OVERRIDE_MIN:
            # Override duro: impactos reales detectados → activar siempre
            conf_entry["activate"] = True
            reconciled_active.append(stroke)
            print(f"  ✅ {stroke}: activado por impactos reales ({n_impacts} impactos, confidence={confidence:.2f})")

        elif not activate:
            # LLM decidió no activar y no hay impactos suficientes → respetar
            conf_entry["deactivation_reason"] = "LLM marcó activate=false (sin impactos reales)"
            print(f"  ⏭  {stroke}: desactivado por LLM (activate=false)")

        elif confidence < CONFIDENCE_THRESHOLD:
            # LLM quiso activar pero confianza insuficiente y sin impactos → desactivar
            conf_entry["activate"]            = False
            conf_entry["deactivation_reason"] = (
                f"Confianza {confidence:.2f} < umbral {CONFIDENCE_THRESHOLD} — "
                f"desactivado por validación Python"
            )
            print(
                f"  ⚠️  {stroke}: desactivado por baja confianza "
                f"({confidence:.2f} < {CONFIDENCE_THRESHOLD})"
            )

        else:
            # Activar: confianza suficiente y LLM lo marcó como activate=true
            reconciled_active.append(stroke)
            print(f"  ✅ {stroke}: activado (confidence={confidence:.2f})")

    result["active_agents"]    = reconciled_active
    result["agent_confidence"] = agent_confidence

    print(f"  → active_agents reconciliados: {reconciled_active}")
    return result
