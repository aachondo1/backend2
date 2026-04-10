"""
TennisAI — Agents Pipeline v8 — FIXED
══════════════════════════════════════
FIX aplicado:
  ✅ extract_peak_frames llamada con 3 argumentos (sin impact_frames) — línea ~722

Cambios respecto a v7:
  - agent_synthesizer delega a agent_synthesizer.py (igual que especialistas)
    La lógica inline queda en _agent_synthesizer_legacy() como referencia.

Mejoras en agent_synthesizer.py:
  1. System prompt noise-aware: tono del LLM según quality_score
       < 0.55 → cauteloso, sin biomecánica directa
       0.55–0.80 → moderado, lenguaje tentativo en métricas
       > 0.80 → asertivo y quirúrgico
  2. Root cause analysis: detecta hilo conductor cruzado entre golpes
  3. Comparison delta: calcula delta por golpe vs sesión anterior
       Genera alerta si retroceso > 5 pts
  4. Narrativa Hecho → Impacto → Acción en recomendaciones
  5. Prompt2 recibe JSON de prompt1 como "verdad absoluta"
  6. Global score ponderado por session_type

Módulos requeridos en el mismo directorio del deploy:
  - coordinator_precompute.py
  - agent_coordinator.py
  - agent_specialists.py
  - agent_synthesizer.py   ← nuevo
  - helpers.py

Deploy:
  modal deploy agents_pipeline_v8.py

Secrets requeridos en Modal:
  - anthropic-key  →  ANTHROPIC_API_KEY
  - supabase-key   →  SUPABASE_URL, SUPABASE_SERVICE_KEY
"""

import modal
import json

app = modal.App("tennis-agents-pipeline")

# ── Imagen liviana: solo Anthropic, sin visión computacional ──
# add_local_python_source incluye los módulos auxiliares en el container (Modal 1.0+)
agents_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "openai==1.37.0",
        "httpx==0.27.0",
    )
    .add_local_python_source(
        "agent_coordinator",
        "agent_specialists",
        "agent_synthesizer",
        "coordinator_precompute",
        "helpers",
        "bone_mapping_builder",
    )
)


# ─── HELPERS SUPABASE ────────────────────────────────────────
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


# ─── AGENTE COORDINADOR ──────────────────────────────────────
@app.function(image=agents_image, timeout=120, memory=1024,
              secrets=[modal.Secret.from_name("anthropic-key")])
def agent_coordinator(
    mediapipe_data:     dict,
    yolo_data:          dict,
    ball_data:          dict,
    session_type:       str,
    camera_orientation: str  = None,
    equipment_used:     dict = None,
    dominant_hand:      str  = None,
    stroke_stats:       dict = None,
    tactical_context:   dict = None,
    data_quality:       dict = None,
) -> dict:
    from agent_coordinator import run_agent_coordinator
    import os
    return run_agent_coordinator(
        mediapipe_data     = mediapipe_data,
        yolo_data          = yolo_data,
        ball_data          = ball_data,
        session_type       = session_type,
        camera_orientation = camera_orientation,
        equipment_used     = equipment_used,
        dominant_hand      = dominant_hand,
        stroke_stats       = stroke_stats,
        tactical_context   = tactical_context,
        data_quality       = data_quality,
        api_key            = os.environ["ANTHROPIC_API_KEY"],
    )


# ─── AGENTE FOREHAND ─────────────────────────────────────────
@app.function(image=agents_image, timeout=120, memory=1024,
              secrets=[modal.Secret.from_name("anthropic-key")])
def agent_forehand(
    coordinator_data:   dict,
    mediapipe_data:     dict,
    ball_data:          dict,
    camera_orientation: str  = None,
    equipment_used:     dict = None,
    dominant_hand:      str  = None,
    session_type:       str  = "paleteo",
) -> dict:
    from agent_specialists import run_agent_forehand
    import os
    return run_agent_forehand(
        coordinator_data   = coordinator_data,
        mediapipe_data     = mediapipe_data,
        ball_data          = ball_data,
        camera_orientation = camera_orientation,
        equipment_used     = equipment_used,
        dominant_hand      = dominant_hand,
        session_type       = session_type,
        api_key            = os.environ["ANTHROPIC_API_KEY"],
    )


# ─── AGENTE BACKHAND ─────────────────────────────────────────
@app.function(image=agents_image, timeout=120, memory=1024,
              secrets=[modal.Secret.from_name("anthropic-key")])
def agent_backhand(
    coordinator_data:   dict,
    mediapipe_data:     dict,
    ball_data:          dict,
    camera_orientation: str  = None,
    equipment_used:     dict = None,
    dominant_hand:      str  = None,
    session_type:       str  = "paleteo",
) -> dict:
    from agent_specialists import run_agent_backhand
    import os
    return run_agent_backhand(
        coordinator_data   = coordinator_data,
        mediapipe_data     = mediapipe_data,
        ball_data          = ball_data,
        camera_orientation = camera_orientation,
        equipment_used     = equipment_used,
        dominant_hand      = dominant_hand,
        session_type       = session_type,
        api_key            = os.environ["ANTHROPIC_API_KEY"],
    )


# ─── AGENTE SAQUE ────────────────────────────────────────────
@app.function(image=agents_image, timeout=120, memory=1024,
              secrets=[modal.Secret.from_name("anthropic-key")])
def agent_saque(
    coordinator_data:   dict,
    mediapipe_data:     dict,
    ball_data:          dict,
    camera_orientation: str  = None,
    equipment_used:     dict = None,
    dominant_hand:      str  = None,
    session_type:       str  = "paleteo",
) -> dict:
    from agent_specialists import run_agent_saque
    import os
    return run_agent_saque(
        coordinator_data   = coordinator_data,
        mediapipe_data     = mediapipe_data,
        ball_data          = ball_data,
        camera_orientation = camera_orientation,
        equipment_used     = equipment_used,
        dominant_hand      = dominant_hand,
        session_type       = session_type,
        api_key            = os.environ["ANTHROPIC_API_KEY"],
    )


# ─── AGENTE SINTETIZADOR ─────────────────────────────────────
# Delega a agent_synthesizer.py (igual que los especialistas)
@app.function(image=agents_image, timeout=300, memory=1024,
              secrets=[modal.Secret.from_name("anthropic-key")])
def agent_synthesizer(
    coordinator_data:   dict,
    forehand_data:      dict,
    backhand_data:      dict,
    saque_data:         dict,
    mediapipe_data:     dict,
    session_type:       str,
    previous_session:   dict = None,
    camera_orientation: str  = None,
    equipment_used:     dict = None,
    dominant_hand:      str  = None,
) -> dict:
    from agent_synthesizer import run_agent_synthesizer
    import os
    return run_agent_synthesizer(
        coordinator_data   = coordinator_data,
        forehand_data      = forehand_data,
        backhand_data      = backhand_data,
        saque_data         = saque_data,
        mediapipe_data     = mediapipe_data,
        session_type       = session_type,
        previous_session   = previous_session,
        camera_orientation = camera_orientation,
        equipment_used     = equipment_used,
        dominant_hand      = dominant_hand,
        api_key            = os.environ["ANTHROPIC_API_KEY"],
    )


def _agent_synthesizer_legacy(
    coordinator_data:   dict,
    forehand_data:      dict,
    backhand_data:      dict,
    saque_data:         dict,
    mediapipe_data:     dict,
    session_type:       str,
    previous_session:   dict = None,
    camera_orientation: str  = None,
    equipment_used:     dict = None,
    dominant_hand:      str  = None,
) -> dict:
    """Lógica inline original — conservada como referencia, no se ejecuta."""
    import os, re
    from helpers import (
        format_camera_context, format_equipment_context,
        format_session_context, parse_json_response,
        get_openrouter_client, get_model_for_agent,
    )

    client        = get_openrouter_client(os.environ.get("OPENROUTER_API_KEY"))
    camera_ctx    = format_camera_context(camera_orientation)
    equipment_ctx = format_equipment_context(equipment_used, dominant_hand)
    session_ctx   = format_session_context(session_type)

    scores_summary = {}
    active_agents  = coordinator_data.get("active_agents", [])

    if "forehand" in active_agents and forehand_data and not forehand_data.get("error"):
        scores_summary["forehand"] = {
            "total":        forehand_data.get("total_score", 0),
            "nivel":        forehand_data.get("nivel", ""),
            "scores":       forehand_data.get("scores", {}),
            "analisis":     forehand_data.get("analisis_tecnico", {}),
            "observaciones":forehand_data.get("observaciones_detalladas", ""),
        }
    if "backhand" in active_agents and backhand_data and not backhand_data.get("error"):
        scores_summary["backhand"] = {
            "total":        backhand_data.get("total_score", 0),
            "nivel":        backhand_data.get("nivel", ""),
            "scores":       backhand_data.get("scores", {}),
            "analisis":     backhand_data.get("analisis_tecnico", {}),
            "observaciones":backhand_data.get("observaciones_detalladas", ""),
        }
    if "saque" in active_agents and saque_data and not saque_data.get("error"):
        scores_summary["saque"] = {
            "total":        saque_data.get("total_score", 0),
            "nivel":        saque_data.get("nivel", ""),
            "scores":       saque_data.get("scores", {}),
            "analisis":     saque_data.get("analisis_tecnico", {}),
            "observaciones":saque_data.get("observaciones_detalladas", ""),
        }

    total_scores = [v["total"] for v in scores_summary.values() if v.get("total")]
    global_score = round(sum(total_scores) / len(total_scores), 1) if total_scores else 0

    evolution_context = ""
    if previous_session:
        evolution_context = (
            f"SESIÓN ANTERIOR — score: {previous_session.get('global_score')} | "
            f"scores: {json.dumps(previous_session.get('scores', {}))}"
        )

    context = (
        f"{session_ctx}\n"
        f"SCORE GLOBAL: {global_score}/100\n"
        f"forehand: {scores_summary.get('forehand', {}).get('total', 'N/A')} | "
        f"backhand: {scores_summary.get('backhand', {}).get('total', 'N/A')} | "
        f"saque: {scores_summary.get('saque', {}).get('total', 'N/A')}\n"
        f"duración: {mediapipe_data['duration_seconds']}s | "
        f"cámara declarada: {camera_orientation or 'no especificada'} | "
        f"cámara detectada: {coordinator_data.get('camera_quality')} / {coordinator_data.get('camera_angle_detected')}\n"
        f"{camera_ctx}\n"
        f"{equipment_ctx}\n"
        f"{evolution_context}\n"
        f"ANÁLISIS COMPLETO: {json.dumps(scores_summary, ensure_ascii=False)}"
    )

    json_scores = json.dumps(scores_summary, ensure_ascii=False)
    fh_score    = scores_summary.get("forehand", {}).get("total", 0)
    bh_score    = scores_summary.get("backhand", {}).get("total", 0)
    saque_score = scores_summary.get("saque",    {}).get("total", 0)

    prompt1 = (
        "Eres analista jefe de tenis. Responde SOLO con JSON válido, sin markdown ni texto adicional.\n"
        + context + "\n"
        + '{"global_score":' + str(global_score)
        + ',"session_type":"' + session_type + '"'
        + ',"nivel_general":"principiante|intermedio|avanzado|experto"'
        + ',"diagnostico_global":"maximo 3 oraciones"'
        + ',"analisis_por_golpe":{"forehand":{"score":' + str(fh_score) + ',"patrones_detectados":[],"riesgo_lesion":""}'
        + ',"backhand":{"score":' + str(bh_score) + ',"patrones_detectados":[],"riesgo_lesion":""}'
        + ',"saque":{"score":' + str(saque_score) + ',"patrones_detectados":[],"riesgo_lesion":""}}'
        + ',"patrones_globales":[]'
        + ',"evolucion":{"tiene_historial":false,"progreso_global":"","dimensiones_mejoradas":[],"dimensiones_regresadas":[],"tendencia":"primera_sesion"}'
        + ',"prioridades_mejora":[{"prioridad":1,"golpe":"","dimension":"","score_actual":0,"score_objetivo":0,"impacto_estimado":"","urgencia":"critica|alta|media|baja"}]'
        + ',"scores_detalle":' + json_scores + "}"
    )
    msg1       = client.chat.completions.create(
        model=get_model_for_agent("synthesizer"), max_tokens=5000,
        messages=[{"role": "user", "content": prompt1}],
    )
    structured = parse_json_response(msg1.choices[0].message.content)

    seccion_fh = (forehand_data.get("narrativa_seccion", forehand_data.get("observaciones_detalladas", "")) if forehand_data else "")
    seccion_bh = (backhand_data.get("narrativa_seccion", backhand_data.get("observaciones_detalladas", "")) if backhand_data else "")
    seccion_sq = (saque_data.get("narrativa_seccion",    saque_data.get("observaciones_detalladas",    "")) if saque_data    else "")

    data_warnings = []
    if forehand_data and forehand_data.get("datos_insuficientes"):
        data_warnings.append("forehand analizado con promedios globales")
    if backhand_data and backhand_data.get("datos_insuficientes"):
        data_warnings.append("backhand analizado con promedios globales")
    if saque_data and saque_data.get("datos_insuficientes"):
        data_warnings.append("saque analizado con promedios globales")
    warning_note = (
        f"\n⚠️ LIMITACIONES DE DATOS: {'; '.join(data_warnings)}. Mencionar brevemente."
        if data_warnings else ""
    )

    angle_reliability = (
        "Vista lateral — ángulos sagitales confiables. Precaución con rotación de cadera y brazo trasero."
        if camera_orientation and "Lateral" in camera_orientation
        else "Ángulos de alineación de hombros y caderas son los más confiables con esta vista."
        if camera_orientation and "Fondo" in camera_orientation
        else "Ángulos 2D con menor confiabilidad — priorizar patrones cualitativos."
    )

    msg2 = client.chat.completions.create(
        model=get_model_for_agent("synthesizer"), max_tokens=6000,
        messages=[{"role": "user", "content": f"""Eres analista jefe de tenis de alto rendimiento.
Ensambla el reporte narrativo final combinando las secciones pre-escritas con tu análisis global.
Escribe en español, prosa técnica y fluida, sin listas numeradas.

NOTA SOBRE ÁNGULOS: {angle_reliability}{warning_note}

SECCIONES PRE-ESCRITAS (usar casi literalmente, puedes suavizar transiciones):
=== FOREHAND ===
{seccion_fh if seccion_fh else "[No aplica para esta sesión]"}

=== BACKHAND ===
{seccion_bh if seccion_bh else "[No aplica para esta sesión]"}

=== SAQUE ===
{seccion_sq if seccion_sq else "[No aplica para esta sesión]"}

DATOS GLOBALES PARA TUS SECCIONES:
{context}

ESTRUCTURA QUE DEBES ESCRIBIR (solo estas secciones, las de golpes ya están arriba):
1. DIAGNÓSTICO GENERAL (2-3 párrafos): nivel actual, patrones globales, cadena cinética general
2. PATRONES CRÍTICOS Y RIESGO DE LESIÓN (1-2 párrafos): errores que se repiten, riesgos biomecánicos
3. COMPARACIÓN ATP / REFERENCIA (1 párrafo): dónde está el jugador vs referencia profesional
4. RECOMENDACIONES PRIORITARIAS (1-2 párrafos): los 2-3 cambios con mayor impacto

Formato de entrega: texto continuo sin headers, cada sección separada por salto de línea doble."""}],
    )

    structured["reporte_narrativo_completo"] = (
        f"{seccion_fh}\n\n{seccion_bh}\n\n{seccion_sq}\n\n{msg2.choices[0].message.content.strip()}"
    ).strip()
    structured["global_score"]  = global_score
    structured["scores_detalle"] = scores_summary
    structured["prioridades_mejora"] = structured.get("prioridades_mejora", [])

    return structured


# ─── AGENTE COACH ────────────────────────────────────────────
# Sin cambios respecto a v6
@app.function(image=agents_image, timeout=180, memory=1024,
              secrets=[modal.Secret.from_name("openrouter-key")])
def agent_coach(
    synthesizer_result: dict,
    session_type:       str,
    equipment_used:     dict = None,
    dominant_hand:      str  = None,
) -> dict:
    import os
    from helpers import (
        format_equipment_context, format_session_context, parse_json_response,
        get_openrouter_client, get_model_for_agent,
    )

    client        = get_openrouter_client(os.environ.get("OPENROUTER_API_KEY"))
    equipment_ctx = format_equipment_context(equipment_used, dominant_hand)
    session_ctx   = format_session_context(session_type)

    prioridades  = synthesizer_result.get("prioridades_mejora", [])
    global_score = synthesizer_result.get("global_score", 0)
    nivel        = synthesizer_result.get("nivel_general", "intermedio")
    diagnostico  = synthesizer_result.get("diagnostico_global", "")
    root_cause   = synthesizer_result.get("root_cause", "")

    # Recolectar flags de riesgo de lesión desde analisis_por_golpe
    risk_flags = []
    for golpe_data in synthesizer_result.get("analisis_por_golpe", {}).values():
        riesgo = golpe_data.get("riesgo_lesion", "")
        if riesgo and riesgo.lower() not in ("ninguno", "", "bajo", "none", "no"):
            risk_flags.append(riesgo)
    risk_ctx = f"RIESGO DE LESIÓN DETECTADO: {'; '.join(risk_flags)}" if risk_flags else ""

    prompt = f"""Eres un coach de tenis experto especializado en prescripción de entrenamientos biomecánicamente fundamentados.
Tu tarea: crear un plan de entrenamiento práctico, testeable y progresivo basado en el análisis biomecánico.
Responde SOLO con JSON válido, sin markdown.

{session_ctx}
{equipment_ctx}

DIAGNÓSTICO: {diagnostico}
CAUSA RAÍZ: {root_cause}
NIVEL: {nivel} | SCORE GLOBAL: {global_score}/100
PRIORIDADES DE MEJORA: {json.dumps(prioridades, ensure_ascii=False)}
{risk_ctx}

═══════════════════════════════════════════════════════════════════

LÓGICA DE PRESCRIPCIÓN:

1. CONEXIÓN RAÍZ → DRILL
   Para cada prioridad de mejora, diseña drills que:
   - Atacan el PATRÓN ESPECÍFICO (ej: si la causa es "cadera lenta en rotación", drills sin pelota enfocados en rotación pélvica)
   - Son PROGRESIVOS: empiezan sin pelota → con pared → peloteo → juego real
   - Incluyen INDICADORES MEDIBLES: reps, series, métricas de ejecución

2. DRILL DESIGN
   Cada drill debe:
   - Tener 1 objetivo biomecánico claro (no multitarea)
   - Ser ejecutable en 15-30 minutos máximo
   - Incluir 3-4 series de trabajo + descanso
   - Progresión visible en 2-3 semanas de práctica consistente

3. MENTAL CUES
   Palabras clave que el jugador REPITE en cancha mientras juega:
   - Deben ser accionables en tiempo real
   - Cortas (1-2 palabras máximo cada una)
   - Directamente vinculadas a la causa raíz

4. INJURY PREVENTION
   Si hay riesgo detectado:
   - Incluir 1-2 ejercicios de fortalecimiento/movilidad específicos
   - Estos se hacen FUERA DE CANCHA, no durante drills
   - Incluir: series × reps, duración total

5. PLAN SEMANAL
   - LUNES-VIERNES: énfasis en drills específicos + días de trabajo técnico
   - SÁBADO: juego real con atención a las cues mentales
   - DOMINGO: recuperación o drills de mantenimiento

═══════════════════════════════════════════════════════════════════

REGLAS PARA EL JSON:

- "weekly_focus": UNA frase de máx 6 palabras que resuma el enfoque principal de la semana
  EJEMPLO: "Acelerar rotación de cadera en golpes"

- "drill_cards": 3-4 fichas de entrenamiento concretas basadas en la CAUSA RAÍZ
  ESTRUCTURA de cada drill:
    "title": nombre descriptivo del ejercicio (máx 6 palabras)
      EJEMPLO: "Rotación pélvica sin pelota"
    "type": uno de "Canasta" | "Pared" | "Peloteo" | "Sin pelota" | "En juego real"
    "reps": series y repeticiones como string
      EJEMPLO: "4 series × 15 reps, descanso 45s entre series"
    "instruction": instrucciones precisas paso a paso (2-3 oraciones máximo)
      EJEMPLO: "De pie, brazos cruzados. Rota cadera izquierda adelante, mantén 2s. Alterna lados. Énfasis en amplitud de rotación, no velocidad."
    "why": conexión explícita entre este drill y la causa raíz detectada
      EJEMPLO: "Ataca directamente la rotación pélvica lenta. Crea patrón motor de rotación amplia que se transfiere a golpes reales."

- "mental_cues": lista de 2-3 palabras clave CORTAS para repetir en cancha
  EJEMPLOS: ["Cadera primero", "Esperar rotación", "Acelerar"]
  CRITERIO: cada cue debe ser ejecutable en < 1 segundo

- "injury_prevention": si hay riesgo_lesion, incluir 1-2 ejercicios (lista de dicts)
  ESTRUCTURA si aplica:
    [{{"nombre": "nombre corto", "series_reps": "3×8", "duracion": "10 minutos", "descripcion": "instrucciones simples"}}]
  Si NO hay riesgo: lista vacía []

- "ejercicios_prioritarios": mantener formato anterior para compatibilidad frontend
  ESTRUCTURA: [{{"nombre":"", "objetivo":"", "duracion_minutos":0, "repeticiones":"", "descripcion":"", "indicador_progreso":""}}]

- "plan_semanal": distribución semanal de trabajo
  ESTRUCTURA:
  {{"lunes":"", "martes":"", "miercoles":"", "jueves":"", "viernes":"", "sabado":"", "domingo":""}}
  EJEMPLO lunes: "Drills sin pelota (rotación pélvica) 20 min + peloteo focus (rallies 5-10 golpes) 20 min"

- "mensaje_motivacional": 1-2 oraciones que conecten el plan con el progreso esperado
  TONO: realista, específico, orientado a la meta

- "proxima_sesion_foco": qué revisar en la próxima sesión (conectado con causa raíz)
  EJEMPLO: "Amplitud y velocidad de rotación pélvica en forehand"

- "notas_coach": observaciones generales sobre el plan, progresión esperada, tiempos
  EJEMPLO: "Esperar mejoras en 2-3 semanas. Si hay resistencia, reducir a 2 series."

═══════════════════════════════════════════════════════════════════

JSON exacto (mantener estructura):
{{"weekly_focus":"","drill_cards":[{{"title":"","type":"","reps":"","instruction":"","why":""}}],"mental_cues":[],"injury_prevention":[],"plan_semanal":{{"lunes":"","martes":"","miercoles":"","jueves":"","viernes":"","sabado":"","domingo":""}},"ejercicios_prioritarios":[{{"nombre":"","objetivo":"","duracion_minutos":0,"repeticiones":"","descripcion":"","indicador_progreso":""}}],"mensaje_motivacional":"","proxima_sesion_foco":"","notas_coach":""}}"""

    msg = client.chat.completions.create(
        model=get_model_for_agent("prescription"), max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    result = parse_json_response(msg.choices[0].message.content)

    if "raw" in result and len(result) == 1:
        result = {
            "weekly_focus": "",
            "drill_cards": [],
            "mental_cues": [],
            "injury_prevention": [],
            "plan_semanal": {},
            "ejercicios_prioritarios": [],
            "mensaje_motivacional": result.get("raw", "")[:500],
            "proxima_sesion_foco": "",
            "notas_coach": "Plan generado parcialmente — reintentar análisis.",
        }
    return result


# ─── FUNCIÓN PRINCIPAL ───────────────────────────────────────
@app.function(
    image=agents_image,
    timeout=900,
    memory=1024,
    secrets=[
        modal.Secret.from_name("openrouter-key"),
        modal.Secret.from_name("supabase-key"),
    ],
)
def run_agents_pipeline(
    vision_job_id:      str,
    session_type:       str  = "paleteo",
    user_id:            str  = None,
    previous_session:   dict = None,
    camera_orientation: str  = None,
    equipment_used:     dict = None,
    dominant_hand:      str  = None,
    session_date:       str  = None,   # ISO-8601 propagado desde el frontend
) -> dict:
    import concurrent.futures, os, httpx

    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_SERVICE_KEY"]

    print(f"🤖 Agents pipeline v8 — job: {vision_job_id} | sesión: {session_type} | user: {user_id}")

    # ── a. Leer vision_results desde Supabase ────────────────
    print("📥 Leyendo vision_results desde Supabase...")
    resp = httpx.get(
        f"{supabase_url}/rest/v1/vision_results"
        f"?id=eq.{vision_job_id}"
        f"&select=mediapipe_result,yolo_result,ball_result,impact_frames,"
        f"camera_orientation,equipment_used,dominant_hand",
        headers={"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}"},
        timeout=30,
    )
    if resp.status_code != 200 or not resp.json():
        error_msg = f"No se encontró vision_results para job: {vision_job_id}"
        print(f"❌ {error_msg}")
        supabase_patch(supabase_url, supabase_key, "vision_results", vision_job_id,
                       {"status": "error", "error_message": error_msg})
        return {"error": error_msg, "vision_job_id": vision_job_id}

    row = resp.json()[0]

    # ── b. Extraer datos ─────────────────────────────────────
    mediapipe_result   = row.get("mediapipe_result",   {})
    yolo_result        = row.get("yolo_result",        {})
    ball_result        = row.get("ball_result",        {})
    impact_frames_raw  = row.get("impact_frames",      [])
    camera_orientation = row.get("camera_orientation")
    equipment_used     = row.get("equipment_used")
    # Supabase JSONB llega como string vía REST API — deserializar si es necesario
    if isinstance(equipment_used, str):
        try:    equipment_used = json.loads(equipment_used)
        except: equipment_used = {}
    dominant_hand      = row.get("dominant_hand")
    # session_date: parámetro tiene prioridad; fallback al valor almacenado en vision_results
    session_date       = session_date or row.get("session_date")

    # ── FIX RATE LIMIT: top 20 impactos por tipo de golpe ────
    def _top20(stroke):
        hits = [f for f in impact_frames_raw if f.get("stroke_type") == stroke]
        return sorted(hits, key=lambda f: f.get("ball_speed_pixels") or 0, reverse=True)[:20]

    impact_frames = (
        _top20("forehand") +
        _top20("backhand") +
        _top20("saque_o_smash") + _top20("saque") +
        [f for f in impact_frames_raw if f.get("stroke_type") is None][:10]
    )
    print(f"📉 Impactos: {len(impact_frames_raw)} → {len(impact_frames)} (top 20 por tipo)")

    # ── PRE-CÓMPUTO: estadísticas deterministas por golpe ────────
    # Todas las funciones operan sobre impact_frames o yolo_result
    # (persistidos en Supabase). Se inyectan en coordinator_result
    # para que los especialistas los lean sin cambios de firma.
    try:
        from coordinator_precompute import (
            compute_stroke_stats_from_impacts,
            compute_fatigue_by_stroke,
            compute_player_position_context,
        )
        stroke_stats      = compute_stroke_stats_from_impacts(impact_frames, dominant_hand or "right")
        fatigue_by_stroke = compute_fatigue_by_stroke(impact_frames, dominant_hand or "right")
        player_position   = compute_player_position_context(yolo_result)
        from coordinator_precompute import infer_backhand_grip, infer_forehand_grip
        backhand_grip     = infer_backhand_grip(impact_frames, dominant_hand or "right")
        forehand_grip     = infer_forehand_grip(impact_frames, dominant_hand or "right")
        print(
            f"📊 Pre-cómputo listo — golpes: {list(stroke_stats.keys())} | "
            f"posición: {player_position.get('dominant_position')} | "
            f"backhand grip: {backhand_grip.get('grip')} | "
            f"forehand grip: {forehand_grip.get('grip')} (conf: {forehand_grip.get('confidence', 0):.0%})"
        )
    except Exception as e:
        stroke_stats      = {}
        fatigue_by_stroke = {}
        player_position   = {}
        backhand_grip     = {}
        forehand_grip     = {}
        print(f"⚠️  Pre-cómputo falló (no crítico): {e}")

    # ── Slims para el coordinador ────────────────────────────
    # v3 no guarda frames raw en DB — se trabaja solo con summary
    mediapipe_slim = mediapipe_result
    ball_slim      = ball_result
    yolo_slim      = yolo_result
    print(f"📊 Datos listos — mp frames: {mediapipe_result.get('frames_analyzed','?')} | impacts: {len(impact_frames)}")

    racket_label = (
        f"{equipment_used.get('brand','')} {equipment_used.get('model','')}".strip()
        if equipment_used else "no especificada"
    )
    print(
        f"✅ Datos cargados — "
        f"frames: {mediapipe_result.get('frames_analyzed', '?')} | "
        f"cámara: {camera_orientation or 'no especificada'} | "
        f"mano: {dominant_hand or 'no especificada'} | "
        f"raqueta: {racket_label}"
    )

    # ── c. Status → agents_processing ───────────────────────
    supabase_patch(supabase_url, supabase_key, "vision_results", vision_job_id,
                   {"status": "agents_processing"})
    print("🔄 Status actualizado → agents_processing")

    # ── d. Coordinador ───────────────────────────────────────
    print("🎯 Fase 1: Coordinador...")
    coordinator_result = agent_coordinator.remote(
        mediapipe_slim, yolo_slim, ball_slim, session_type,
        camera_orientation, equipment_used, dominant_hand,
        stroke_stats,
    )
    if impact_frames and not coordinator_result.get("impact_frames"):
        coordinator_result["impact_frames"] = impact_frames
    if stroke_stats and not coordinator_result.get("stroke_stats"):
        coordinator_result["stroke_stats"] = stroke_stats
    if fatigue_by_stroke:
        coordinator_result["fatigue_by_stroke"] = fatigue_by_stroke
    if player_position:
        coordinator_result["player_position"] = player_position
    if backhand_grip:
        coordinator_result["backhand_grip"] = backhand_grip
    if forehand_grip:
        coordinator_result["forehand_grip"] = forehand_grip

    # ── Fases biomecánicas desde impact_frames ───────────────────────────────
    # v2: compute_phase_angles ya NO usa mediapipe_data["frames"] (ausente en DB).
    # Opera directamente sobre impact_frames que sí están en Supabase:
    #   - fase impacto     → angles completo del frame de impacto
    #   - fase preparacion → stroke_phases.prep_angle_elbow
    #   - fase followthrough → stroke_phases.followthrough_angle_elbow
    # No requiere cambios en DB ni en vision pipeline.
    try:
        from coordinator_precompute import compute_phase_angles
        phase_angles = compute_phase_angles(
            impact_frames = impact_frames,
            dominant_hand = dominant_hand or "right",
        )
        coordinator_result["phase_angles"] = phase_angles
        available = [s for s, d in phase_angles.items() if d.get("phase_data_available")]
        limited   = [s for s, d in phase_angles.items() if not d.get("phase_data_available")]
        n_used    = {s: d.get("n_impacts_used", "?") for s, d in phase_angles.items()}
        print(
            f"📐 Fases biomecánicas (impact_frames) — "
            f"disponibles: {available or 'ninguno'} | "
            f"limitadas: {limited or 'ninguno'} | "
            f"impactos usados: {n_used}"
        )
    except Exception as e:
        coordinator_result["phase_angles"] = {}
        print(f"⚠️  compute_phase_angles falló (no crítico): {e}")

    # ── Validación post-LLM: ball_validated por golpe ────────
    # Enriquece coordinator_result["data_quality"]["impact_validation"]
    # con flags ball_validated=True/False que los especialistas usan
    # para penalizar potencia_pelota cuando el tracker no confirmó el impacto.
    try:
        from coordinator_precompute import sync_impact_quality_check, build_data_quality_report
        data_quality      = build_data_quality_report(
            mediapipe_result, yolo_result, ball_result, impact_frames
        )
        coordinator_result = sync_impact_quality_check(
            coordinator_result, impact_frames, data_quality
        )
        print("✅ sync_impact_quality_check completado")
    except Exception as e:
        print(f"⚠️  sync_impact_quality_check falló (no crítico): {e}")

    active_agents = coordinator_result.get("active_agents", [])
    print(f"✅ Coordinador completado — agentes activos: {active_agents}")

    # ── e. Especialistas en paralelo ─────────────────────────
    print("🤖 Fase 2: Agentes especializados en paralelo...")
    forehand_result = backhand_result = saque_result = None
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        if "forehand" in active_agents:
            futures["forehand"] = executor.submit(
                agent_forehand.remote,
                coordinator_result, mediapipe_slim, ball_slim,
                camera_orientation, equipment_used, dominant_hand, session_type,
            )
        if "backhand" in active_agents:
            futures["backhand"] = executor.submit(
                agent_backhand.remote,
                coordinator_result, mediapipe_slim, ball_slim,
                camera_orientation, equipment_used, dominant_hand, session_type,
            )
        if "saque" in active_agents:
            futures["saque"] = executor.submit(
                agent_saque.remote,
                coordinator_result, mediapipe_slim, ball_slim,
                camera_orientation, equipment_used, dominant_hand, session_type,
            )
        if "forehand" in futures: forehand_result = futures["forehand"].result()
        if "backhand" in futures: backhand_result = futures["backhand"].result()
        if "saque"    in futures: saque_result    = futures["saque"].result()
    print("✅ Agentes especializados completados")

    # ── f. Sintetizador ──────────────────────────────────────
    print("📝 Fase 3: Sintetizador...")
    synthesizer_result = agent_synthesizer.remote(
        coordinator_result, forehand_result, backhand_result, saque_result,
        mediapipe_result, session_type, previous_session,
        camera_orientation, equipment_used, dominant_hand,
    )
    print("✅ Sintetizador completado")

    # ── g. Coach ─────────────────────────────────────────────
    print("💪 Fase 4: Coach...")
    coach_result = agent_coach.remote(
        synthesizer_result, session_type, equipment_used, dominant_hand,
    )
    print("✅ Coach completado")

    # ── h. Bone mapping y digital twin ───────────────────────
    print("🦴 Generando bone mapping y digital twin...")

    # digital_twin_data: keyframes de preparación e impacto por golpe (backward compat)
    from helpers import extract_peak_frames
    digital_twin_data = extract_peak_frames(mediapipe_result, coordinator_result, dominant_hand)

    # bone_mapping_data: JSON completo para BoneMappingTab.tsx
    # Genera modos representative/best/worst con pose (33 landmarks) + analysis_delta ATP
    bone_mapping_data = {}
    try:
        from bone_mapping_builder import generate_bone_mapping_input

        bone_mapping_data = generate_bone_mapping_input(
            impact_frames    = impact_frames,
            mediapipe_result = mediapipe_result,
            dominant_hand    = dominant_hand or "right",
            active_strokes   = active_agents,
            forehand_grip    = forehand_grip,
            backhand_grip    = backhand_grip,
        )
        # Inyectar scores de bone mapping en coordinator_result para el sintetizador
        for stroke, bm in bone_mapping_data.items():
            rep_score = bm.get("modes", {}).get("representative", {}).get("score")
            if rep_score is not None:
                if "bone_mapping_scores" not in coordinator_result:
                    coordinator_result["bone_mapping_scores"] = {}
                coordinator_result["bone_mapping_scores"][stroke] = rep_score

        print(f"✅ Bone mapping generado — golpes: {list(bone_mapping_data.keys())}")
        for stroke, bm in bone_mapping_data.items():
            meta      = bm.get("session_meta", {})
            rep_score = bm.get("modes", {}).get("representative", {}).get("score", "?")
            print(
                f"  [{stroke}] score_rep={rep_score} | impactos={meta.get('total_impacts','?')} "
                f"| quality={meta.get('quality_score','?')} | landmarks={'✓' if meta.get('has_landmarks') else '✗'}"
            )
    except Exception as e:
        import traceback
        print(f"⚠️  Bone mapping falló (no crítico): {e}")
        print(traceback.format_exc())

    # ── i. Guardar sesión en Supabase ────────────────────────
    print("💾 Guardando sesión en Supabase...")
    session_data = {
        "user_id":            user_id,
        "session_type":       session_type,
        "global_score":       synthesizer_result.get("global_score", 0),
        "nivel_general":      synthesizer_result.get("nivel_general", ""),
        "diagnostico_global": synthesizer_result.get("diagnostico_global", ""),
        "reporte_narrativo":  synthesizer_result.get("reporte_narrativo_completo", ""),
        "scores_detalle":     synthesizer_result.get("scores_detalle", {}),
        "prioridades_mejora": synthesizer_result.get("prioridades_mejora", []),
        "plan_ejercicios":    coach_result,
        "camera_orientation": camera_orientation,
        "equipment_used":     equipment_used,
        # ── Campos nuevos Día 2 ──────────────────────────────
        "synthesizer_metadata": {
            "top_3_insights":   synthesizer_result.get("top_3_insights",   []),
            "root_cause":       synthesizer_result.get("root_cause",       ""),
            "delta_headline":   synthesizer_result.get("delta_headline",   ""),
            "fatigue_analysis": synthesizer_result.get("fatigue_analysis", {}),
            "comparison_delta": synthesizer_result.get("comparison_delta", {}),
        },
        "quality_score": {
            "mediapipe_coverage":      coordinator_result.get("data_quality", {}).get("mediapipe_coverage"),
            "ball_sync_rate":          coordinator_result.get("data_quality", {}).get("ball_sync_rate"),
            "overall_quality_score":   coordinator_result.get("data_quality", {}).get("overall_quality_score"),
            "processing_gaps_percent": mediapipe_result.get("processing_gaps_percent"),
        },
        **({"actual_session_date": session_date} if session_date else {}),
        # ── raw_data ─────────────────────────────────────────
        "raw_data": {
            "coordinator":       coordinator_result,
            "forehand":          forehand_result,
            "backhand":          backhand_result,
            "saque":             saque_result,
            "digital_twin_data": digital_twin_data,
            "bone_mapping":      bone_mapping_data,
        },
    }

    saved_session = supabase_post(supabase_url, supabase_key, "sessions", session_data)
    session_id    = saved_session.get("id") if saved_session else None
    print(f"{'✅' if session_id else '⚠️'} Sesión guardada — id: {session_id}")

    patch_data = {"status": "completed"}
    if session_id:
        patch_data["session_id"] = session_id
    ok = supabase_patch(supabase_url, supabase_key, "vision_results", vision_job_id, patch_data)
    print(f"{'✅' if ok else '❌'} vision_results → status: completed | session_id: {session_id}")

    return {
        "vision_job_id":      vision_job_id,
        "session_id":         session_id,
        "status":             "completed",
        "global_score":       synthesizer_result.get("global_score", 0),
        "nivel_general":      synthesizer_result.get("nivel_general", ""),
        "active_agents":      active_agents,
        "camera_orientation": camera_orientation,
        "equipment_used":     equipment_used,
    }


# ─── LOCAL ENTRYPOINT (testing desde Colab) ──────────────────
@app.local_entrypoint()
def main(vision_job_id: str = "", session_type: str = "paleteo", user_id: str = ""):
    result = run_agents_pipeline.remote(
        vision_job_id=vision_job_id,
        session_type=session_type,
        user_id=user_id or None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
