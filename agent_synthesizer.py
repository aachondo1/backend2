"""
TennisAI — Agent Synthesizer v2
════════════════════════════════
Módulo externo extraído de agents_pipeline_v7.
Testeable desde Colab sin Modal:

    from agent_synthesizer import run_agent_synthesizer
    result = run_agent_synthesizer(
        coordinator_data   = coordinator_result,
        forehand_data      = forehand_result,
        backhand_data      = backhand_result,
        saque_data         = saque_result,
        mediapipe_data     = mediapipe_result,
        session_type       = "paleteo",
        previous_session   = None,
        camera_orientation = "Lateral-Centro",
        equipment_used     = None,
        dominant_hand      = "right",
        api_key            = "sk-ant-...",
    )

Cambios respecto al sintetizador inline de v7:
  1. System prompt noise-aware: tono del LLM según quality_score
       < 0.55 → cauteloso ("parece que", "hay indicios de"), sin biomecánica directa
       0.55–0.80 → moderado, lenguaje tentativo en métricas numéricas
       > 0.80 → asertivo y quirúrgico, instrucciones técnicas directas
  2. Root cause analysis: prompt1 busca hilo conductor cruzado entre golpes
       Si FH y BH fallan por lo mismo → root_cause es ese patrón, no los síntomas
  3. Comparison delta: calcula delta por golpe vs sesión anterior
       Genera delta_headline condicional (alerta si retroceso > 5 pts)
  4. Narrativa Hecho → Impacto → Acción en recomendaciones prioritarias
  5. Prompt2 recibe JSON de prompt1 como "verdad absoluta":
       no puede contradecir scores ni root_cause ni insights ya calculados
  6. Global score ponderado por session_type (partido → saque pesa más)
  7. top_3_insights y fatigue_analysis como campos estructurados en el output
"""

import json


# ─── PESOS DE GOLPES POR SESSION_TYPE ────────────────────────────────────────
STROKE_WEIGHTS = {
    "partido": {"forehand": 0.35, "backhand": 0.30, "saque": 0.35},
    "clase":   {"forehand": 0.40, "backhand": 0.40, "saque": 0.20},
    "paleteo": {"forehand": 0.40, "backhand": 0.40, "saque": 0.20},
}
STROKE_WEIGHTS_DEFAULT = {"forehand": 0.40, "backhand": 0.40, "saque": 0.20}


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _compute_weighted_score(scores_summary: dict, session_type: str) -> float:
    """
    Promedio ponderado de golpes activos.
    Renormaliza si no todos los golpes están presentes.
    """
    weights      = STROKE_WEIGHTS.get(session_type, STROKE_WEIGHTS_DEFAULT)
    weighted_sum = 0.0
    total_weight = 0.0

    for stroke, data in scores_summary.items():
        score = data.get("total", 0)
        if score:
            w             = weights.get(stroke, 0.20)
            weighted_sum += score * w
            total_weight += w

    if total_weight == 0:
        return 0.0
    return round(weighted_sum / total_weight, 1)


def _build_noise_context(coordinator_data: dict) -> tuple[float, str, str]:
    """
    Extrae quality_score y construye:
      - context_note:  texto breve para el user prompt (qué pasó con los datos)
      - system_tone:   instrucción de tono para el system prompt (cómo hablar)

    Tres niveles:
      < 0.55 → CAUTELOSO: sin biomecánica directa, priorizar táctica y grabación
      0.55–0.80 → MODERADO: lenguaje tentativo en métricas numéricas
      > 0.80 → ASERTIVO: instrucciones técnicas directas con grados y tiempos
    """
    data_quality  = coordinator_data.get("data_quality", {})
    quality_score = data_quality.get("overall_quality_score", 1.0)
    try:
        quality_score = float(quality_score)
    except (TypeError, ValueError):
        quality_score = 1.0

    noise_report   = data_quality.get("noise_report", {})
    anomalies      = noise_report.get("total_anomalies_found", 0)
    affected_parts = noise_report.get("affected_body_parts", [])
    parts_str      = ", ".join(affected_parts) if affected_parts else "no especificadas"

    if quality_score < 0.55:
        context_note = (
            f"⚠️  CALIDAD BAJA (score={quality_score:.2f}, anomalías={anomalies}, "
            f"partes afectadas: {parts_str})."
        )
        system_tone = (
            "NIVEL DE CONFIANZA BAJO — actúa como coach CAUTELOSO. "
            "Usa frases como 'Parece que...', 'Hay indicios de...', 'Se observa una tendencia a...'. "
            "NO des instrucciones biomecánicas directas sobre ángulos o grados específicos. "
            "Prioriza consejos de posicionamiento táctico y mejora de condiciones de grabación. "
            "Si mencionas una limitación técnica, ofrece siempre una alternativa táctica."
        )
    elif quality_score < 0.80:
        context_note = (
            f"CALIDAD MODERADA (score={quality_score:.2f}, anomalías={anomalies})."
        )
        system_tone = (
            "NIVEL DE CONFIANZA MODERADO — para ángulos y métricas numéricas específicas usa "
            "frases como 'se estima', 'aproximadamente', 'la tendencia indica'. "
            "Para patrones cualitativos claros (ritmo, preparación, footwork) puedes ser directo."
        )
    else:
        context_note = (
            f"CALIDAD ALTA (score={quality_score:.2f}). Datos biomecánicos confiables."
        )
        system_tone = (
            "NIVEL DE CONFIANZA ALTO — sé asertivo y quirúrgico. "
            "Los datos son sólidos: puedes dar instrucciones técnicas directas "
            "con grados específicos, tiempos y referencias biomecánicas precisas."
        )

    return quality_score, context_note, system_tone


def _build_fatigue_text(coordinator_data: dict) -> str:
    """Bloque de texto de fatiga para el prompt."""
    fatigue = coordinator_data.get("fatigue_by_stroke", {})
    if not fatigue:
        return ""

    lines = ["ANÁLISIS DE FATIGA POR GOLPE:"]
    for stroke, data in fatigue.items():
        trend  = data.get("trend", "")
        delta  = data.get("delta_percent", "")
        minute = data.get("degradation_start_minute")
        lines.append(
            f"  • {stroke}: tendencia={trend}"
            + (f", degradación={delta}%" if delta != "" else "")
            + (f", inicio_degradación=min {minute}" if minute else "")
        )
    return "\n".join(lines)


def _compute_delta(
    scores_summary: dict,
    previous_session: dict | None,
    global_score: float,
) -> tuple[str, str, dict]:
    """
    Calcula delta por golpe vs sesión anterior.
    Retorna (evolution_context_text, delta_headline, comparison_delta).

    comparison_delta: dict persistible con deltas numéricos por golpe + global.
      {
        "global":    {"prev": 63.0, "current": 56.0, "delta": -7.0},
        "forehand":  {"prev": 70.0, "current": 60.0, "delta": -10.0},
        ...
      }

    delta_headline: frase condicional para el prompt.
      - Si algún golpe retrocede > 5 pts → alerta específica
      - Si todo mejora → positivo
      - Sin historial → strings vacíos, comparison_delta vacío
    """
    if not previous_session:
        return "", "", {}

    prev_scores = previous_session.get("scores_detalle", previous_session.get("scores", {}))
    prev_global = previous_session.get("global_score", global_score)

    delta_lines      = []
    alerts           = []
    improvements     = []
    comparison_delta = {}

    for stroke, data in scores_summary.items():
        current = data.get("total", 0)
        prev    = 0
        if isinstance(prev_scores.get(stroke), dict):
            prev = prev_scores[stroke].get("total", 0)
        elif isinstance(prev_scores.get(stroke), (int, float)):
            prev = float(prev_scores[stroke])

        if prev and current:
            delta = round(current - prev, 1)
            sign  = "+" if delta >= 0 else ""
            delta_lines.append(f"  • {stroke}: {prev} → {current} ({sign}{delta})")
            comparison_delta[stroke] = {"prev": prev, "current": current, "delta": delta}
            if delta <= -5:
                alerts.append(f"retroceso crítico en {stroke} ({sign}{delta} pts)")
            elif delta >= 5:
                improvements.append(f"mejora en {stroke} (+{delta} pts)")

    global_delta = round(global_score - float(prev_global), 1)
    global_sign  = "+" if global_delta >= 0 else ""
    comparison_delta["global"] = {
        "prev":    float(prev_global),
        "current": global_score,
        "delta":   global_delta,
    }

    evolution_context = (
        f"SESIÓN ANTERIOR — score global: {prev_global} | "
        f"scores previos: {json.dumps(prev_scores, ensure_ascii=False)}\n"
        f"DELTA vs sesión anterior (global: {global_sign}{global_delta}):\n"
        + "\n".join(delta_lines)
    )

    if alerts and improvements:
        delta_headline = (
            f"Sesión mixta: {', '.join(improvements)}, pero ALERTA: {', '.join(alerts)}. "
            "El headline del diagnóstico debe reflejar esta tensión — no solo lo positivo."
        )
    elif alerts:
        delta_headline = (
            f"ALERTA DE REGRESIÓN: {', '.join(alerts)}. "
            "El diagnóstico global NO debe ser positivo — priorizar la regresión."
        )
    elif improvements:
        delta_headline = f"Progreso confirmado: {', '.join(improvements)}."
    else:
        delta_headline = "Sin cambios significativos respecto a la sesión anterior."

    return evolution_context, delta_headline, comparison_delta


def _angle_reliability_note(camera_orientation: str | None) -> str:
    if not camera_orientation:
        return "Ángulos 2D con menor confiabilidad — priorizar patrones cualitativos."
    if "Lateral" in camera_orientation:
        return "Vista lateral — ángulos sagitales confiables. Precaución con rotación de cadera y brazo trasero."
    if "Fondo" in camera_orientation:
        return "Ángulos de alineación de hombros y caderas son los más confiables con esta vista."
    return "Ángulos 2D con menor confiabilidad — priorizar patrones cualitativos."


# ─── FUNCIÓN PRINCIPAL ───────────────────────────────────────────────────────

def run_agent_synthesizer(
    coordinator_data:   dict,
    forehand_data:      dict | None,
    backhand_data:      dict | None,
    saque_data:         dict | None,
    mediapipe_data:     dict,
    session_type:       str,
    previous_session:   dict | None = None,
    camera_orientation: str  | None = None,
    equipment_used:     dict | None = None,
    dominant_hand:      str  | None = None,
    api_key:            str         = "",
) -> dict:
    """
    Sintetiza los resultados de los agentes especialistas en un reporte cohesivo.

    Dos llamadas LLM:
      1. JSON estructurado (system noise-aware) → scores, root_cause, top_3_insights,
         fatigue_analysis, delta de evolución, prioridades
      2. Reporte narrativo → recibe el JSON de llamada 1 como "verdad absoluta";
         estructura obligatoria Hecho → Impacto → Acción

    Output keys garantizados:
      global_score, nivel_general, diagnostico_global, root_cause,
      analisis_por_golpe, patrones_globales, top_3_insights,
      fatigue_analysis, evolucion, prioridades_mejora,
      scores_detalle, reporte_narrativo_completo
    """
    import anthropic
    from helpers import (
        format_camera_context,
        format_equipment_context,
        format_session_context,
        parse_json_response,
    )

    client        = anthropic.Anthropic(api_key=api_key)
    camera_ctx    = format_camera_context(camera_orientation)
    equipment_ctx = format_equipment_context(equipment_used, dominant_hand)
    session_ctx   = format_session_context(session_type)

    # ── 1. Scores summary ─────────────────────────────────────────────────────
    active_agents:  list = coordinator_data.get("active_agents", [])
    scores_summary: dict = {}

    for stroke, data in [
        ("forehand", forehand_data),
        ("backhand", backhand_data),
        ("saque",    saque_data),
    ]:
        if stroke in active_agents and data and not data.get("error"):
            scores_summary[stroke] = {
                "total":         data.get("total_score", 0),
                "nivel":         data.get("nivel", ""),
                "scores":        data.get("scores", {}),
                "analisis":      data.get("analisis_tecnico", {}),
                "observaciones": data.get("observaciones_detalladas", ""),
            }

    # ── 2. Global score ponderado ─────────────────────────────────────────────
    global_score = _compute_weighted_score(scores_summary, session_type)

    # ── 3. Contextos ──────────────────────────────────────────────────────────
    quality_score, noise_ctx, system_tone = _build_noise_context(coordinator_data)
    fatigue_text                          = _build_fatigue_text(coordinator_data)
    angle_note                            = _angle_reliability_note(camera_orientation)
    tactical_ctx                          = coordinator_data.get("tactical_context", {})
    evolution_context, delta_headline, comparison_delta = _compute_delta(scores_summary, previous_session, global_score)

    fh_score    = scores_summary.get("forehand", {}).get("total", 0)
    bh_score    = scores_summary.get("backhand", {}).get("total", 0)
    saque_score = scores_summary.get("saque",    {}).get("total", 0)

    data_warnings = []
    for stroke, data in [("forehand", forehand_data), ("backhand", backhand_data), ("saque", saque_data)]:
        if data and data.get("datos_insuficientes"):
            data_warnings.append(f"{stroke} analizado con promedios globales")
    warning_note = (
        f"⚠️  LIMITACIONES: {'; '.join(data_warnings)}. Mencionar brevemente."
        if data_warnings else ""
    )

    context = "\n".join(filter(None, [
        session_ctx,
        f"SCORE GLOBAL (ponderado por sesión): {global_score}/100",
        f"forehand: {fh_score or 'N/A'} | backhand: {bh_score or 'N/A'} | saque: {saque_score or 'N/A'}",
        f"duración: {mediapipe_data.get('duration_seconds', '?')}s | "
        f"cámara declarada: {camera_orientation or 'no especificada'} | "
        f"cámara detectada: {coordinator_data.get('camera_angle_detected', '?')}",
        camera_ctx,
        equipment_ctx,
        noise_ctx,
        fatigue_text                                                                    or None,
        f"CONTEXTO TÁCTICO: {json.dumps(tactical_ctx, ensure_ascii=False)}"
            if tactical_ctx else None,
        evolution_context                                                               or None,
        f"DELTA HEADLINE: {delta_headline}"                                             if delta_headline else None,
        warning_note                                                                    or None,
    ]))

    json_scores = json.dumps(scores_summary, ensure_ascii=False)

    # ════════════════════════════════════════════════════════════════════════════
    # LLAMADA 1 — JSON estructurado (system prompt noise-aware)
    # ════════════════════════════════════════════════════════════════════════════
    system1 = (
        "Eres analista jefe de tenis de alto rendimiento. "
        "Respondes SOLO con JSON válido, sin markdown ni texto adicional. "
        f"{system_tone}"
    )

    prompt1 = f"""{context}

ANÁLISIS COMPLETO DE ESPECIALISTAS:
{json_scores}

INSTRUCCIÓN — root_cause (campo obligatorio):
Identifica el problema raíz compartido entre golpes, si existe.
Si forehand Y backhand fallan por lo mismo (ej: rodillas estiradas, preparación tardía,
falta de rotación de caderas) → ese es el root_cause, no los síntomas individuales.
Si no hay patrón común claro → "No se detectó causa raíz común entre golpes".

INSTRUCCIÓN — top_3_insights:
Ordena por IMPACTO REAL en el juego. PRIORIZA insights cruzados (afectan 2+ golpes).
Un problema de footwork que afecta FH y BH es más importante que un error de muñeca aislado.
Si la fatiga afecta múltiples golpes → es un insight válido de tipo "fatiga".
Cada insight debe ser accionable en la próxima sesión.

INSTRUCCIÓN — evolucion:
{"Calcula delta por golpe. Si algún golpe retrocedió > 5 pts, refléjalo en tendencia y progreso_global." if previous_session else "Sin historial — tendencia: primera_sesion."}
{f"DELTA HEADLINE a respetar: {delta_headline}" if delta_headline else ""}

JSON exacto (sin texto adicional):
{{
  "global_score": {global_score},
  "session_type": "{session_type}",
  "nivel_general": "principiante|intermedio|avanzado|experto",
  "diagnostico_global": "máximo 3 oraciones",
  "root_cause": "",
  "analisis_por_golpe": {{
    "forehand": {{"score": {fh_score},    "patrones_detectados": [], "riesgo_lesion": ""}},
    "backhand": {{"score": {bh_score},    "patrones_detectados": [], "riesgo_lesion": ""}},
    "saque":    {{"score": {saque_score}, "patrones_detectados": [], "riesgo_lesion": ""}}
  }},
  "patrones_globales": [],
  "top_3_insights": [
    {{
      "area": "",
      "impacto": "alto|medio|bajo",
      "descripcion": "",
      "golpes_afectados": []
    }}
  ],
  "fatigue_analysis": {{
    "status": "sin_degradacion|degradacion_leve|degradacion_moderada|degradacion_severa|sin_datos",
    "minuto_inicio": null,
    "observacion": ""
  }},
  "evolucion": {{
    "tiene_historial": {"true" if previous_session else "false"},
    "progreso_global": "",
    "dimensiones_mejoradas": [],
    "dimensiones_regresadas": [],
    "tendencia": "primera_sesion|mejorando|estable|regresando"
  }},
  "prioridades_mejora": [
    {{
      "prioridad": 1,
      "golpe": "",
      "dimension": "",
      "score_actual": 0,
      "score_objetivo": 0,
      "impacto_estimado": "",
      "urgencia": "critica|alta|media|baja"
    }}
  ],
  "scores_detalle": {json_scores}
}}"""

    msg1       = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=5000,
        system=system1,
        messages=[{"role": "user", "content": prompt1}],
    )
    structured = parse_json_response(msg1.content[0].text)

    # ════════════════════════════════════════════════════════════════════════════
    # LLAMADA 2 — Reporte narrativo
    # El JSON de llamada 1 es la "verdad absoluta": no puede ser contradicho.
    # Estructura obligatoria en recomendaciones: Hecho → Impacto → Acción
    # ════════════════════════════════════════════════════════════════════════════
    seccion_fh = (forehand_data.get("narrativa_seccion", forehand_data.get("observaciones_detalladas", "")) if forehand_data else "")
    seccion_bh = (backhand_data.get("narrativa_seccion", backhand_data.get("observaciones_detalladas", "")) if backhand_data else "")
    seccion_sq = (saque_data.get("narrativa_seccion",    saque_data.get("observaciones_detalladas",    "")) if saque_data    else "")

    top_insights = structured.get("top_3_insights", [])
    root_cause   = structured.get("root_cause", "")

    insights_block = ""
    if top_insights:
        insights_block = (
            "PATRONES CRUZADOS DETECTADOS (anclar el diagnóstico en estos):\n"
            + "\n".join(
                f"  • [{i.get('impacto','').upper()}] {i.get('area','')}: {i.get('descripcion','')} "
                f"(golpes afectados: {', '.join(i.get('golpes_afectados', []))})"
                for i in top_insights
            )
        )

    system2 = (
        "Eres analista jefe de tenis de alto rendimiento. "
        "Escribes en español, prosa técnica y fluida, sin listas numeradas ni headers. "
        f"{system_tone} "
        "REGLA ABSOLUTA: No puedes contradecir los scores, el root_cause ni los insights "
        "calculados en el JSON estructurado. Ese análisis es la verdad absoluta de esta sesión."
    )

    prompt2 = f"""Ensambla el reporte narrativo final de esta sesión de tenis.

VERDAD ABSOLUTA — no contradecir bajo ninguna circunstancia:
  Score global: {structured.get('global_score', global_score)}/100
  Nivel: {structured.get('nivel_general', '')}
  Root cause: {root_cause if root_cause else 'No detectado'}
  Diagnóstico: {structured.get('diagnostico_global', '')}
{insights_block}
{f"DELTA HEADLINE: {delta_headline}" if delta_headline else ""}

NOTA SOBRE ÁNGULOS: {angle_note}
{warning_note}

SECCIONES PRE-ESCRITAS DE ESPECIALISTAS (usar casi literalmente, solo suavizar transiciones):
=== FOREHAND ===
{seccion_fh if seccion_fh else "[No aplica para esta sesión]"}

=== BACKHAND ===
{seccion_bh if seccion_bh else "[No aplica para esta sesión]"}

=== SAQUE ===
{seccion_sq if seccion_sq else "[No aplica para esta sesión]"}

DATOS DE SESIÓN:
{context}

ESTRUCTURA QUE DEBES ESCRIBIR (solo estas 4 secciones — los golpes ya están arriba):

1. DIAGNÓSTICO GENERAL (2-3 párrafos)
   Si existe root_cause, ábrelo con él como hilo conductor.
   Describe nivel actual, patrones globales y cadena cinética.
   Integra los patrones cruzados de los insights.

2. PATRONES CRÍTICOS Y RIESGO DE LESIÓN (1-2 párrafos)
   Errores que se repiten en múltiples golpes.
   Riesgos biomecánicos reales a medio plazo.

3. COMPARACIÓN CON REFERENCIA (1 párrafo)
   Dónde está el jugador vs el nivel técnico esperado para este score global.

4. RECOMENDACIONES PRIORITARIAS (1-2 párrafos)
   Los 2-3 cambios con mayor impacto real.
   OBLIGATORIO — cada recomendación debe seguir esta estructura exacta:
     Hecho (dato observado) → Impacto (consecuencia en el juego) → Acción (ejercicio o drill concreto).
   Ejemplo correcto: "Tu preparación de derecha promedia X grados de apertura (Hecho).
   Esto reduce la aceleración disponible en pelotas rápidas (Impacto).
   En la próxima sesión, practica el ejercicio de sombra con marcador de posición de raqueta (Acción)."

Formato: texto continuo sin headers, cada sección separada por doble salto de línea."""

    msg2 = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=6000,
        system=system2,
        messages=[{"role": "user", "content": prompt2}],
    )

    narrative   = msg2.content[0].text.strip()
    full_report = "\n\n".join(filter(None, [seccion_fh, seccion_bh, seccion_sq, narrative]))

    # ── Ensamblar output final ────────────────────────────────────────────────
    structured["reporte_narrativo_completo"] = full_report
    structured["global_score"]               = global_score
    structured["scores_detalle"]             = scores_summary
    structured["prioridades_mejora"]         = structured.get("prioridades_mejora", [])

    # Garantizar campos nuevos si el LLM los omitió
    structured.setdefault("top_3_insights",   [])
    structured.setdefault("root_cause",        "")
    structured.setdefault("fatigue_analysis", {
        "status":        "sin_datos",
        "minuto_inicio": None,
        "observacion":   "",
    })

    # Campos de comparativa — calculados en Python, no por LLM
    # Se sobreescriben siempre para garantizar consistencia con _compute_delta
    structured["delta_headline"]   = delta_headline
    structured["comparison_delta"] = comparison_delta

    return structured
