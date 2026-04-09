# ✅ CHECKLIST FINAL — TennisAI Backend Repository

## 🎯 Objetivo
Tener un repositorio GitHub limpio, documentado y listo para desplegar en Modal.

---

## 📋 FASE 1: Preparación (Local)

### Clonar repo de GitHub
```
[ ] Crear nuevo repo en GitHub.com (privado)
    URL: https://github.com/aachondo1/TennisAI_Backend
    
[ ] Clonar localmente:
    git clone https://github.com/aachondo1/TennisAI_Backend.git
    cd TennisAI_Backend
```

### Crear estructura de carpetas
```
[ ] mkdir -p vision_pipeline
[ ] mkdir -p agents_pipeline
[ ] mkdir -p docs
[ ] mkdir -p deployment
```

---

## 📁 FASE 2: Copiar Archivos Python

### Vision Pipeline
```
[ ] vision_pipeline/vision_pipeline_v3.py
    └─ Origen: vision_pipeline_v3__13_.py
    └─ Tamaño esperado: ~50 KB
    └─ Verificar: @app.function decorators, Modal imports OK

[ ] vision_pipeline/helpers.py
    └─ Origen: helpers.py
    └─ Tamaño esperado: ~10 KB
    └─ Verificar: Supabase functions, MediaPipe utils
```

### Agents Pipeline
```
[ ] agents_pipeline/agents_pipeline_v8.py
    └─ Origen: agents_pipeline_v8__13_.py
    └─ Tamaño esperado: ~40 KB
    └─ Verificar: add_local_python_source incluye todos los módulos

[ ] agents_pipeline/agent_coordinator.py
    └─ Origen: agent_coordinator.py
    └─ Tamaño esperado: ~15 KB

[ ] agents_pipeline/agent_specialists.py
    └─ Origen: agent_specialists__7___1_.py
    └─ Tamaño esperado: ~60 KB
    └─ Verificar: forehand, backhand, saque agents

[ ] agents_pipeline/agent_synthesizer.py
    └─ Origen: agent_synthesizer__2_.py
    └─ Tamaño esperado: ~25 KB
    └─ Verificar: root_cause, delta_analysis, insights

[ ] agents_pipeline/agent_coach.py
    └─ COPIAR si existe, si no CREAR vacío
    └─ Responsabilidad: plan de entrenamiento

[ ] agents_pipeline/coordinator_precompute.py
    └─ Origen: coordinator_precompute.py
    └─ Tamaño esperado: ~100 KB
    └─ Verificar: compute_stroke_stats, compute_tactical_context

[ ] agents_pipeline/bone_mapping_builder.py
    └─ Origen: bone_mapping_builder__2_.py
    └─ Tamaño esperado: ~25 KB
```

---

## 📄 FASE 3: Copiar Archivos de Configuración (Raíz)

### README y documentación principal
```
[ ] README.md
    └─ Copiar desde: /home/claude/TennisAI_Backend_README.md
    └─ Verificar: Stack, quick start, troubleshooting
    └─ Verificar: URLs actualizadas (GitHub, Modal, Supabase)
```

### Dependencias
```
[ ] requirements.txt
    └─ Copiar desde: /home/claude/requirements.txt
    └─ Verificar: Modal >= 1.0, Anthropic >= 0.40, torch == 2.4.0
    └─ Verificar: MediaPipe 0.10.14, YOLOv8 8.2.0
```

### Variables de entorno
```
[ ] .env.example
    └─ Copiar desde: /home/claude/.env.example
    └─ Verificar: SUPABASE_URL, SUPABASE_SERVICE_KEY (eyJ...), ANTHROPIC_API_KEY
    └─ Verificar: comentarios explicativos
```

### Git ignore
```
[ ] .gitignore
    └─ Copiar desde: /home/claude/gitignore (renombrar a .gitignore)
    └─ Verificar: Excluye venv/, .env, __pycache__, *.mp4, etc.
```

### Licencia
```
[ ] LICENSE
    └─ Crear (Propietario o MIT)
```

---

## 📚 FASE 4: Documentación Específica

### Pipeline de Visión
```
[ ] vision_pipeline/README.md
    └─ Crear nuevo archivo
    └─ Incluir: Descripción, archivos, uso, entrada/salida
    └─ Referencia: Template en QUICK_SETUP_GUIDE.md
```

### Pipeline de Agentes
```
[ ] agents_pipeline/README.md
    └─ Crear nuevo archivo
    └─ Incluir: Arquitectura 6-agentes, uso, entrada/salida
    └─ Referencia: Template en QUICK_SETUP_GUIDE.md
```

### Arquitectura
```
[ ] docs/architecture.md
    └─ Crear nuevo archivo
    └─ Incluir: Diagrama de flujo, componentes, tecnologías
    └─ Referencia: Template en QUICK_SETUP_GUIDE.md
```

### Modelo de datos
```
[ ] docs/data_model.md
    └─ Crear nuevo archivo
    └─ Incluir: Schema Supabase (sessions, vision_results, synthesizer_metadata, etc.)
    └─ RLS policies, check constraints
```

### Guía de despliegue
```
[ ] docs/deployment_guide.md
    └─ Crear nuevo archivo
    └─ Incluir: Setup de secretos en Modal, comandos de deploy
    └─ Monitoreo: modal logs, modal app list
```

### Troubleshooting
```
[ ] docs/troubleshooting.md
    └─ Crear nuevo archivo
    └─ Incluir: Problemas comunes, soluciones
    └─ Ejemplos: Vision timeout, Supabase INSERT fallos, truncation
```

### Roadmap
```
[ ] docs/roadmap.md
    └─ Crear nuevo archivo (OPCIONAL)
    └─ Incluir: Fase B, C, D (TrackNetV3, ViTPose, Stroke Phase Detector)
```

---

## 🔧 FASE 5: Validación Pre-Commit

### Estructura completa
```
[ ] tree o ls -R muestra:
    ✓ vision_pipeline/ con 3 archivos
    ✓ agents_pipeline/ con 7 archivos
    ✓ docs/ con 4-5 archivos
    ✓ README.md, requirements.txt, .env.example, .gitignore en raíz
```

### Python syntax
```
[ ] Verificar import errors:
    python -m py_compile vision_pipeline/vision_pipeline_v3.py
    python -m py_compile agents_pipeline/agents_pipeline_v8.py
    
    (No debería haber SyntaxError)
```

### Contenido crítico
```
[ ] README.md contiene:
    ✓ Quick start
    ✓ Stack técnico
    ✓ Componentes principales
    ✓ Data model
    ✓ Secrets requeridos
    ✓ Troubleshooting

[ ] requirements.txt contiene:
    ✓ modal-client >= 1.0.0
    ✓ anthropic >= 0.40.0
    ✓ torch == 2.4.0
    ✓ mediapipe == 0.10.14
    ✓ ultralytics == 8.2.0
    ✓ huggingface_hub == 0.34.0

[ ] .env.example contiene:
    ✓ SUPABASE_URL
    ✓ SUPABASE_SERVICE_KEY (eyJ... format)
    ✓ ANTHROPIC_API_KEY
    ✓ Comentarios explicativos

[ ] .gitignore contiene:
    ✓ venv/, __pycache__, *.pyc
    ✓ .env (variables locales)
    ✓ *.mp4, *.mov (videos)
    ✓ .modal/, modal_volume/
    ✓ .vscode/, .idea/
```

---

## 🚀 FASE 6: Git & Push

### Inicializar Git
```
[ ] git init (si es nuevo directorio)
    [ ] git add .
    [ ] git config user.name "Tu Nombre"
    [ ] git config user.email "tu@email.com"
    [ ] git commit -m "Initial backend structure: vision & agents pipelines"
```

### Push a GitHub
```
[ ] git branch -M main
[ ] git remote add origin https://github.com/aachondo1/TennisAI_Backend.git
[ ] git push -u origin main
[ ] Verificar: https://github.com/aachondo1/TennisAI_Backend → Files tree
```

---

## 🔐 FASE 7: Configuración en GitHub (Opcional pero Recomendado)

### Branch protection
```
[ ] Settings → Branches → Add rule
    ✓ Branch name: main
    ✓ Require pull request reviews before merging
    ✓ Require status checks to pass before merging
```

### Collaborators
```
[ ] Settings → Collaborators & teams
    ✓ Agregar miembros del equipo
```

### Secrets (GitHub)
```
[ ] Settings → Secrets and variables → Actions
    ✓ SUPABASE_SERVICE_KEY
    ✓ ANTHROPIC_API_KEY (si necesitas CI/CD)
```

---

## 🚀 FASE 8: Setup Modal & Secretos

### Crear secretos en Modal
```
[ ] modal secret create supabase-key \
      SUPABASE_URL=https://dqsjprndqltuwfilevmb.supabase.co \
      SUPABASE_SERVICE_KEY=eyJ...

[ ] modal secret create anthropic-key \
      ANTHROPIC_API_KEY=sk-...

[ ] Verificar: modal secret list
```

### Deploy Vision Pipeline
```
[ ] modal deploy vision_pipeline/vision_pipeline_v3.py
    [ ] Esperar a que finalize (~10 min)
    [ ] Verificar: modal app list → "tennis-vision-pipeline-v3"
    [ ] Verificar logs: modal logs tennis-vision-pipeline-v3
```

### Deploy Agents Pipeline
```
[ ] modal deploy agents_pipeline/agents_pipeline_v8.py
    [ ] Esperar a que finalize (~2-3 min)
    [ ] Verificar: modal app list → "tennis-agents-pipeline"
    [ ] Verificar logs: modal logs tennis-agents-pipeline
```

---

## ✅ VERIFICACIÓN FINAL

### GitHub
```
[ ] Repo visible en https://github.com/aachondo1/TennisAI_Backend
[ ] Archivos correctos en cada carpeta
[ ] README.md se renderiza correctamente
[ ] .gitignore funciona (no hay __pycache__/ commiteado)
```

### Modal
```
[ ] Ambos apps deployados: vision-pipeline-v3, agents-pipeline
[ ] Secretos creados correctamente
[ ] Logs sin errores críticos
```

### Local (verificación rápida)
```
[ ] git remote -v → origin apunta a GitHub
[ ] git log --oneline → al menos 1 commit
[ ] ls -la vision_pipeline/ → 3 archivos
[ ] ls -la agents_pipeline/ → 7-8 archivos
```

---

## 📊 Resumen de Entregas

### Documentos Generados (listos en `/home/claude/`)

```
✅ TennisAI_Backend_README.md      → README.md (raíz del repo)
✅ QUICK_SETUP_GUIDE.md           → Referencia paso-a-paso
✅ BACKEND_FILE_MAPPING.md        → Mapeo detallado
✅ BACKEND_STRUCTURE_SUMMARY.md   → Sumario visual
✅ requirements.txt               → requirements.txt (raíz)
✅ .env.example                   → .env.example (raíz)
✅ gitignore                      → .gitignore (raíz)
```

### Archivos a Copiar (del proyecto)

```
De tu proyecto actual:
✅ vision_pipeline_v3__13_.py         → vision_pipeline/vision_pipeline_v3.py
✅ helpers.py                         → vision_pipeline/helpers.py
✅ agents_pipeline_v8__13_.py         → agents_pipeline/agents_pipeline_v8.py
✅ agent_coordinator.py               → agents_pipeline/agent_coordinator.py
✅ agent_specialists__7___1_.py       → agents_pipeline/agent_specialists.py
✅ agent_synthesizer__2_.py           → agents_pipeline/agent_synthesizer.py
✅ coordinator_precompute.py          → agents_pipeline/coordinator_precompute.py
✅ bone_mapping_builder__2_.py        → agents_pipeline/bone_mapping_builder.py
```

### Archivos a Crear (nuevo)

```
README.md en carpetas:
  ☐ vision_pipeline/README.md
  ☐ agents_pipeline/README.md

Documentación general:
  ☐ docs/architecture.md
  ☐ docs/data_model.md
  ☐ docs/deployment_guide.md
  ☐ docs/troubleshooting.md
  ☐ docs/roadmap.md (opcional)
```

---

## 🎯 OBJETIVO FINAL

```
✓ Repositorio público/privado en GitHub
✓ Estructura clara: vision_pipeline/, agents_pipeline/, docs/
✓ Documentación completa: README, architecture, deployment
✓ Código limpio: sin versiones viejas, sin .docx, sin secretos
✓ Deployable: compatible con Modal, secrets configurados
✓ Colaborativo: branch protection, documentación clara, issues template
```

---

## 📞 ¿Dudas?

Si algo no está claro:
1. Consulta **BACKEND_FILE_MAPPING.md** para detalle de archivos
2. Consulta **QUICK_SETUP_GUIDE.md** para paso-a-paso
3. Consulta **BACKEND_STRUCTURE_SUMMARY.md** para estructura visual

---

**✅ Cuando completes todo este checklist, tendrás un repositorio backend production-ready.**
