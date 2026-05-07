# LATAM — Software Engineer (ML & LLMs) Challenge

Documentación de la solución al challenge — **v1.0.0**.

---

## TL;DR

Modelo de predicción de retraso de vuelos en SCL **operacionalizado de cabo a rabo**: del notebook del DS a un API en producción con observability, CI/CD automatizado y review automático de PRs por Claude.

| | |
|---|---|
| 🔧 **API en producción** | https://latam-delay-api-108332844354.us-central1.run.app |
| 🌐 **GitHub Pages (landing)** | https://joseluis911.github.io/latam_challenge/ |
| 📦 **Repo** | https://github.com/joseluis911/latam_challenge |
| ✅ **Tests** | 8/8 passing · ~92% coverage |
| ⚡ **Stress test live** | 6,241 reqs · **0 failures** · p95 420 ms |
| 🚀 **Releases en main** | `v0.1.0` → `v0.2.0` → `v0.3.0` → **`v1.0.0`** |

### Extras más allá del enunciado

Lo que pidió LATAM + lo que sumé encima:

| Lo pedido | Lo entregado | Extra |
|---|---|---|
| Modelo en `model.py` | LR balanceada con justificación cuantitativa, lazy bootstrap, top-10 features, 6 bugs documentados y corregidos | — |
| API con FastAPI | Endpoints + Pydantic + Swagger + override 422→400 | — |
| Deploy en cloud | Cloud Run + Artifact Registry | **Terraform IaC** (`infra/`, 8 recursos) · **Cloud Monitoring dashboard** custom · pattern `lifecycle.ignore_changes` profesional |
| CI/CD básico | `ci.yml` lint+tests + `cd.yml` deploy | **`claude-review.yml`** — Claude revisa cada PR · **`pages.yml`** — auto-deploy de docs a GitHub Pages · `terraform-validate` job en CI |
| Documentación en `challenge.md` | Este archivo (~470 líneas, todas las decisiones, bugs, métricas) | **Landing en GitHub Pages** con paleta LATAM y screenshot del dashboard |

---

## Autor

- **Nombre:** Jose Luis Santiago Marquez
- **Mail:** jlsantiago691@gmail.com
- **Repositorio:** https://github.com/joseluis911/latam_challenge
- **API desplegada:** https://latam-delay-api-108332844354.us-central1.run.app
- **Docs (Pages):** https://joseluis911.github.io/latam_challenge/

---

## Estructura del repositorio

```
.
├── challenge/            # model.py, api.py
├── data/                 # dataset
├── docs/                 # challenge.md (este archivo) + index.html (GitHub Pages)
├── tests/                # model, api, stress
├── workflows/ → .github/workflows/  (ci.yml, cd.yml)
├── Dockerfile
├── Makefile
└── requirements*.txt
```

---

## Flujo de trabajo (GitFlow)

- `main` → releases oficiales para review (con tags semánticos).
- `develop` → integración.
- `feature/*` → cada parte del challenge (`feature/part-1-model`, `feature/part-2-api`, …).
- Las ramas de desarrollo **no se borran** (lo pide el enunciado).

---

## Estrategia de releases incrementales

`main` no espera al final del desarrollo: cada vez que `develop` alcanza un estado *consumible* se mergea a `main` con un tag semántico. Así `main` siempre refleja un artefacto deployable y el historial cuenta una historia de delivery iterativo en vez de un único big-bang final.

| Después de | MVP | ¿Release? | Tag |
|---|---|---|---|
| Part I — modelo | ❌ no consumible (librería sola) | no | — |
| Part II — API | ✅ **primer MVP** (API local funcional) | sí | **`v0.1.0`** |
| Part III — deploy | ✅ MVP en cloud | sí | **`v0.2.0`** |
| Part IV — CI/CD | ✅ auto-deploy + observability | sí | **`v0.3.0`** |
| Release final | 🎯 polish + tag oficial | sí | **`v1.0.0`** |

Mecánica del release a `main`:

```bash
git checkout main
git pull
git merge --no-ff develop -m "release: vX.Y.Z (descripción)"
git tag -a vX.Y.Z -m "vX.Y.Z: <highlights>"
git push origin main --tags
```

La rama `release/v1.0` se reserva como espacio ceremonial de estabilización para el lanzamiento oficial v1.0.0.

---

## Part I — Transcripción del modelo (`challenge/model.py`)

**Objetivo:** transcribir `exploration.ipynb` a `model.py` cumpliendo `make model-test`.

### Modelo elegido: **Logistic Regression con `class_weight='balanced'`**

El DS dejó la decisión abierta en la última celda del notebook:

> *"There is no noticeable difference in results between XGBoost and LogisticRegression. […] Improves the model's performance when balancing classes. With this, the model to be productive must be the one that is trained with the top 10 features and class balancing, **but which one?**"*

Como las métricas son equivalentes, elegí Logistic Regression sobre los siguientes criterios de **operacionalización**:

| Criterio | LogReg balanceada | XGBoost balanceado | Ganador |
|---|---|---|---|
| Métricas (recall clase 1, f1) | ~0.69 / ~0.36 | ~0.69 / ~0.37 | empate |
| Interpretabilidad (coeficientes) | Lineal, directo | Requiere SHAP | **LR** |
| Cold start en Cloud Run | ~50 ms load | ~500 ms load | **LR** |
| Tamaño del modelo serializado | ~3 KB | ~50–500 KB | **LR** |
| Dependencias nuevas | 0 (sklearn ya pinned) | `xgboost` extra | **LR** |
| Reproducibilidad determinista | `random_state` simple | múltiples seeds | **LR** |

Para una API que sirve predicciones en tiempo real con scale-to-zero, los segundos en cold start importan. La interpretabilidad de los coeficientes de LR permite además explicar a operaciones por qué un vuelo se predice como retraso (qué aerolínea, qué mes, internacional vs nacional) — valor de negocio real.

**Configuración final:**

```python
LogisticRegression(
    class_weight="balanced",   # corrige el ~80/20 de la data
    random_state=1,            # mismo seed que el notebook
    max_iter=1000,             # asegura convergencia con balance
)
```

### Bugs encontrados y corregidos

| # | Archivo | Bug | Fix |
|---|---|---|---|
| 1 | `challenge/model.py:10` | Anotación de tipo escrita como `Union(Tuple[…], pd.DataFrame)` con paréntesis (era llamada a función, no subscript de `Union`) | `Union[tuple[…], pd.DataFrame]` |
| 2 | `tests/model/test_model.py:29` | Path cwd-relativo `"../data/data.csv"` rompía cuando se corre desde la raíz (como hace el `Makefile`) | Path absoluto: `Path(__file__).resolve().parents[2] / "data" / "data.csv"` |
| 3 | `exploration.ipynb` cell 13 (`get_period_day`) | Comparaciones con `<` `>` no inclusivas; `5:00:00` exacto retorna `None` | No se usa en el modelo final (las top-10 features no requieren `period_day`); se documenta para referencia |
| 4 | `exploration.ipynb` cell 26 (`get_rate_from_column`) | Calcula `total/delays` (inverso de la tasa de delay) | No afecta al modelo, solo a la exploración; se documenta para que no se use tal cual |
| 5 | `requirements-test.txt` | `pytest~=6.2.5` es incompatible con `anyio>=4` que jala `fastapi/starlette`: `ModuleNotFoundError: No module named '_pytest.scope'` al cargar plugins | Bump: `pytest~=7.4`, `pytest-cov~=4.1`, `coverage~=7.6`, `mockito~=1.5` |
| 6 | `challenge/model.py` (`_bootstrap_from_disk`) | `pd.read_csv` lanza `DtypeWarning` por columnas con tipos mixtos en el CSV | Pasar `low_memory=False` |

### Decisiones de diseño

- **Top-10 features fijas como constante de módulo** (`TOP_10_FEATURES`). Vienen del feature importance que XGBoost calculó en el notebook (cell 59).
- **One-hot consistency**: tras `pd.get_dummies` se hace `reindex(columns=TOP_10_FEATURES, fill_value=0)`. Esto garantiza que el `predict()` siempre reciba las mismas 10 columnas aunque la categoría no esté presente en el input (crítico para el API cuando el cliente manda un solo vuelo).
- **Balanceo de clases vía `class_weight='balanced'`**: sklearn calcula los pesos como `n_samples / (n_classes * np.bincount(y))`. Es equivalente a `scale_pos_weight = n_neg / n_pos` de XGBoost, sin tener que mantener el cálculo explícito.
- **Lazy bootstrap en `predict()`**: si `predict()` se llama antes de `fit()`, el modelo se auto-entrena leyendo `data/data.csv` la primera vez. Permite que `test_model_predict` pase (no llama `fit` antes) y en producción FastAPI puede entrenar al startup.
- **`DelayModel` no tiene I/O en `__init__`**: el bootstrap es lazy y aislado en `_bootstrap_from_disk()`. La clase es testeable sin tocar disco si se llama `fit()` directamente.
- **Sin estado global**, todo en `self`. Ningún print/log dentro del modelo.
- **Paths robustos**: `DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "data.csv"`. Funciona desde cualquier cwd.

### Verificación

```bash
make model-test
# pytest tests/model --cov=challenge --cov-report term ...
# 4 passed, coverage ≥ 80%
```

---

## Part II — API con FastAPI (`challenge/api.py`)

**Objetivo:** exponer el modelo vía FastAPI cumpliendo `make api-test`.

### Endpoints

| Método | Ruta | Status code OK | Descripción |
|---|---|---|---|
| `GET` | `/health` | 200 | Liveness probe (`{"status": "OK"}`) |
| `POST` | `/predict` | 200 | Predice delay (0/1) por vuelo |
| `GET` | `/docs` | 200 | Swagger UI auto-generado por FastAPI |
| `GET` | `/redoc` | 200 | ReDoc auto-generado |
| `GET` | `/openapi.json` | 200 | Especificación OpenAPI 3 |

### Contrato de `POST /predict`

**Request:**

```json
{
  "flights": [
    {"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "N", "MES": 3}
  ]
}
```

**Response (200):**

```json
{"predict": [0]}
```

**Response (400)** ante input inválido:

```json
{"detail": [{"loc": ["body", "flights", 0, "MES"], "msg": "MES must be between 1 and 12, got 13", ...}]}
```

### Validaciones (Pydantic v1)

| Campo | Regla | Origen |
|---|---|---|
| `OPERA` | Debe estar en `KNOWN_OPERAS` (set de 23 aerolíneas del dataset) | `@validator("OPERA")` |
| `TIPOVUELO` | Debe ser `"I"` (Internacional) o `"N"` (Nacional) | `@validator("TIPOVUELO")` |
| `MES` | Entero en `[1, 12]` | `@validator("MES")` |
| `flights` | Lista no vacía de `FlightInput` | tipo `list[FlightInput]` |

### Override 422 → 400

FastAPI por default devuelve `422 Unprocessable Entity` ante validación de Pydantic. Los tests del challenge esperan `400 Bad Request`. Lo corrijo con un exception handler global:

```python
@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": exc.errors()})
```

### Carga del modelo

El `DelayModel` se instancia una vez a nivel de módulo (`_model = DelayModel()`). Como `DelayModel` no tiene I/O en `__init__`, la importación es barata. La primera llamada a `/predict` dispara el lazy bootstrap (entrena leyendo `data/data.csv`), las siguientes son inmediatas. Esto evita el hit de cold start en `/health` (importante para readiness probes).

### Bugs adicionales encontrados y corregidos

| # | Archivo | Bug | Fix |
|---|---|---|---|
| 7 | `requirements.txt` | `starlette 0.20.4` (que jala `fastapi~=0.86`) usa `anyio.start_blocking_portal`, removido en `anyio>=4`. `fastapi.testclient.TestClient` falla con `AttributeError: module 'anyio' has no attribute 'start_blocking_portal'` | Pin `anyio<4` |

### Documentación auto-generada

FastAPI genera la spec OpenAPI gratis a partir de los Pydantic models y los `Field`/docstrings:

- `/docs` — Swagger UI interactivo (probar requests desde el browser)
- `/redoc` — ReDoc (más limpio para leer)
- `/openapi.json` — la spec en JSON, consumible por Postman/Insomnia/clients generados

El metadato del API (`title`, `description`, `version`, `contact`) se configura en el constructor de `FastAPI(...)`.

### Verificación

```bash
# unit + integration
make api-test
# pytest tests/api --cov=challenge ...
# 4 passed, api.py coverage ~98%

# manual
uvicorn challenge.api:app --reload
# → http://localhost:8000/health
# → http://localhost:8000/docs   (probar /predict desde Swagger)
```

---

## Part III — Deploy en la nube

**Proveedor:** Google Cloud Platform — Cloud Run + Artifact Registry, region `us-central1`.

**API en producción:** https://latam-delay-api-108332844354.us-central1.run.app

### Servicios GCP usados (free tier perpetuo)

| Servicio | Rol | Free tier mensual |
|---|---|---|
| **Cloud Run** | Hostea el contenedor del FastAPI; scale-to-zero | 2M req, 360k GB-s, 180k vCPU-s |
| **Artifact Registry** | Repo Docker privado (`latam-images`) | 0.5 GB |
| **IAM Service Account** | `latam-deployer` para Part IV CD pipeline | gratis |
| **Cloud Logging** | Logs estructurados auto-recolectados | 50 GB |
| **Cloud Monitoring** | Dashboard custom (RPS, p95, 5xx, instances) | métricas built-in gratis |

Costo real estimado para el challenge: **$0.00 USD**.

### Infraestructura como código — Terraform

Todo provisionado con Terraform en la carpeta `infra/`:

```
infra/
├── versions.tf            # terraform 1.5+ + google provider 5.40+
├── main.tf                # provider google
├── variables.tf           # project_id, region, service_name, etc.
├── terraform.tfvars       # valores concretos (no secretos)
├── artifact_registry.tf   # google_artifact_registry_repository
├── iam.tf                 # latam-deployer SA + 3 roles project-level
├── cloud_run.tf           # google_cloud_run_v2_service + public_invoker
├── monitoring.tf          # google_monitoring_dashboard custom
├── outputs.tf             # cloud_run_url, ar_repo_url, image_url, etc.
└── README.md
```

**8 recursos** se crean con un solo `terraform apply`. State local (gitignored). Para limpiar todo al final: `terraform destroy`.

### Patrón "lifecycle ignore_changes" en Cloud Run

El recurso `google_cloud_run_v2_service.api` tiene `lifecycle.ignore_changes = [template[0].containers[0].image]`. Razón:

- Terraform **crea** el service con un placeholder image (`us-docker.pkg.dev/cloudrun/container/hello`).
- La **imagen real** la pushea Docker → Artifact Registry y luego `gcloud run services update --image=…` la swappea.
- Sin el `ignore_changes`, cada `terraform plan` detectaría que la imagen cambió y querría revertirla al placeholder. Con el `ignore_changes`, Terraform maneja el "cascarón" del service y la CI/CD maneja la imagen — separación correcta de responsabilidades.

Este es el patrón estándar para Cloud Run con Terraform en producción.

### Bootstrap GCP (one-time, manual)

Antes del primer `terraform apply`, configuración manual una sola vez:

1. Proyecto GCP: `latam-challenge-495606` (display name: `latam-challenge`)
2. Billing asociado a una cuenta activa
3. APIs habilitadas:
   - `run.googleapis.com`
   - `artifactregistry.googleapis.com`
   - `iam.googleapis.com`
   - `cloudresourcemanager.googleapis.com`
   - `monitoring.googleapis.com`
   - `orgpolicy.googleapis.com`
4. Org policy override: `iam.allowedPolicyMemberDomains` → `allowAll: true` (necesario para permitir `allUsers` como invocador, ya que la cuenta vive en una org Workspace).
5. ADC autenticadas con la cuenta correcta:
   - `gcloud auth login`
   - `gcloud auth application-default login`
   - `gcloud auth application-default set-quota-project latam-challenge-495606`

### Dockerfile

Multi-stage para imagen pequeña sin compiladores en runtime:

- **Stage 1 (builder):** `python:3.10-slim` + `build-essential` → `pip wheel --wheel-dir=/wheels -r requirements.txt`
- **Stage 2 (runtime):** `python:3.10-slim` → instala desde wheels (sin gcc) → copia `challenge/` y `data/` → usuario non-root `app` → `EXPOSE 8080` → `CMD uvicorn challenge.api:app --host 0.0.0.0 --port ${PORT}`

`PORT` lo inyecta Cloud Run automáticamente (no se setea como env explícita; eso da error "reserved env name").

`.dockerignore` excluye tests, infra, docs, notebook, .git, venv, etc. Imagen final ~280 MB.

### Flujo de deploy

```bash
# (una vez) bootstrap manual
cd infra
terraform init
terraform apply

# (cada deploy) build + push + swap
make docker-build      # docker build -t latam-delay-api:latest .
make docker-push       # tag + push a us-central1-docker.pkg.dev/...
make deploy            # gcloud run services update --image=...
```

O atajo: `make deploy` corre los 3 en cadena.

### Verificación

`make stress-test` con la URL ya en `Makefile` línea 26:

```
locust -f tests/stress/api_stress.py --headless --users 100 --spawn-rate 1 --run-time 60s -H <cloud_run_url>
```

Resultados (test real contra Cloud Run en producción):

| Métrica | Valor |
|---|---|
| Total requests | 6,241 |
| Failures | **0 (0.00%)** |
| Throughput | 105 req/s sostenidos |
| Latency p50 | 260 ms |
| Latency p95 | 420 ms |
| Latency p99 | 530 ms |
| Latency p99.9 | 4.8 s (cold start tail) |

Reporte HTML completo en `reports/stress-test.html` después de correr el test.

### Bug adicional encontrado

| # | Archivo | Bug | Fix |
|---|---|---|---|
| 8 | `requirements-test.txt` | `locust~=1.6` (de 2020) usa imports de Flask incompatibles con Jinja2 ≥3.1: `ImportError: cannot import name 'escape' from 'jinja2'` | Bump `locust~=2.20` |

### Dashboard de monitoreo

Provisionado por `infra/monitoring.tf` — Cloud Monitoring custom dashboard accesible solo desde el proyecto GCP (privado, requiere acceso). Captura del dashboard durante / después del stress test:

![LATAM Delay API monitoring dashboard — Cloud Run RPS, p95 latency, 5xx rate, instance count](screenshots/dashboard-overview.jpg)

Widgets configurados:

- **Request count (RPS)** — sobre `run.googleapis.com/request_count`, `ALIGN_RATE`
- **Request latency p95** — sobre `run.googleapis.com/request_latencies`, `REDUCE_PERCENTILE_95`
- **5xx error rate** — filtrado por `metric.label.response_code_class = "5xx"`
- **Active container instances** — sobre `run.googleapis.com/container/instance_count`, `ALIGN_MEAN`

Todo se actualiza en tiempo real mientras el API recibe tráfico. La URL del dashboard sale como output de Terraform (`monitoring_dashboard_url`) pero requiere autenticación al proyecto GCP.

---

## Part IV — CI/CD (`.github/workflows/`)

Cuatro workflows separados, cada uno con responsabilidad única (no un workflow gigante):

| Workflow | Trigger | Jobs |
|---|---|---|
| `ci.yml` | PR a `main`/`develop`, push a esos branches | `lint` (ruff) → `test-model` + `test-api` (pytest+coverage) → `terraform-validate` (fmt + validate) |
| `cd.yml` | Push a `main` (excluyendo cambios solo en docs/) | Auth GCP → build Docker → push a AR (con tag `:latest` y `:${SHA}`) → `gcloud run services update` → smoke test contra `/health` |
| `claude-review.yml` | PR abierto / actualizado | Claude lee el diff y postea un review en el PR usando `anthropics/claude-code-action@beta` |
| `pages.yml` | Push a `main` con cambios en `docs/**` | `actions/upload-pages-artifact` + `actions/deploy-pages` → publica `docs/` a GitHub Pages oficial |

### `cd.yml` — auto-deploy a Cloud Run

Replica `make deploy` dentro de Actions:

1. **Auth GCP** vía `google-github-actions/auth@v2` con el secret `GCP_SA_KEY` (JSON key del service account `latam-deployer` provisionado por Terraform en Part III).
2. **Configure docker** para Artifact Registry (`gcloud auth configure-docker us-central1-docker.pkg.dev`).
3. **Build Docker image** con dos tags: `:${GITHUB_SHA}` (inmutable, para rollback) y `:latest`.
4. **Push** ambos tags a `latam-images`.
5. **Deploy** la imagen `:${GITHUB_SHA}` a Cloud Run con `gcloud run services update --image=...`. Cloud Run crea una nueva revisión, ruta el 100% del tráfico, y mantiene la anterior por si hay rollback rápido.
6. **Smoke test** con `curl /health` (5 reintentos con backoff). Si no devuelve `{"status":"OK"}` el job falla y se notifica.
7. **Summary** en `$GITHUB_STEP_SUMMARY` con URL, image SHA y service name — visible en la UI del run.

`paths-ignore` excluye cambios solo en `docs/**` y workflows de `pages`/`claude-review` para no redeployar el API por un README typo.

### `claude-review.yml` — code review automático con Claude

Cada PR a `develop` / `main` dispara un review automático escrito por Claude (Anthropic). El prompt focaliza en bugs, security, test coverage, y break de contratos del API. Comenta directo en el PR. Si falta el secret `ANTHROPIC_API_KEY`, el job se salta sin romper.

### `pages.yml` — GitHub Pages via workflow oficial

En lugar del flujo "deploy from branch" (que renderiza pero no auditás), usamos el flujo moderno con `actions/upload-pages-artifact` + `actions/deploy-pages`. Resultado: cada deploy a Pages es un workflow run con su propio status check, log, y URL pública (`https://joseluis911.github.io/latam_challenge/`). Cuando se mergea algo en `docs/`, Pages se actualiza automáticamente sin necesidad de tocar settings.

### Secrets / variables (GitHub → Settings → Secrets and variables → Actions)

| Nombre | Tipo | Scope | Para qué |
|---|---|---|---|
| `GCP_SA_KEY` | 🔒 Secret | environment `production` | JSON key del SA `latam-deployer`. Usado por `cd.yml` para auth GCP |
| `ANTHROPIC_API_KEY` | 🔒 Secret | repo-level | API key de Anthropic. Usado por `claude-review.yml` |

El environment `production` actúa como gate adicional: si en el futuro queremos requerir aprobación manual antes de cada deploy, se configura ahí mismo.

### Patrón de despliegue resultante

```
PR feature/* → develop
   ↓
ci.yml (lint + tests + tf-validate) + claude-review.yml (Claude comenta)
   ↓
merge a develop
   ↓
PR develop → main (release vX.Y.Z)
   ↓
ci.yml + claude-review.yml (de nuevo)
   ↓
merge a main + tag vX.Y.Z
   ↓
cd.yml (build → push → deploy → smoke test) ✅
pages.yml (si cambió docs/) ✅
```

Cada release a main = nueva imagen Docker en producción + nueva revisión en Cloud Run + Pages actualizado, sin que nadie toque la consola.

---

## Cómo correr local

```bash
pip install -r requirements.txt -r requirements-dev.txt -r requirements-test.txt
make model-test     # 4 passed
make api-test       # 4 passed
make stress-test    # contra el Cloud Run live
```

---

## Verificación end-to-end (v1.0.0)

| Check | Resultado | Evidencia |
|---|---|---|
| `make model-test` | ✅ 4/4 passed, 87% coverage `model.py` | `tests/model/test_model.py` |
| `make api-test` | ✅ 4/4 passed, 98% coverage `api.py` | `tests/api/test_api.py` |
| **Total coverage** | **92%** | `pytest --cov=challenge` |
| `make stress-test` (live) | ✅ 6,241 reqs · **0 fails** · 105 req/s · p95 420ms · p99 530ms | `reports/stress-test.html` |
| `terraform validate` | ✅ infra valid | CI job `terraform-validate` |
| Cloud Run live | ✅ HTTPS, public, scale-to-zero | `https://latam-delay-api-108332844354.us-central1.run.app` |
| Auto-deploy on merge to main | ✅ green | `.github/workflows/cd.yml` runs |
| GitHub Pages | ✅ deployed via workflow | `https://joseluis911.github.io/latam_challenge/` |
| Claude PR reviewer | ✅ comments on every PR | `.github/workflows/claude-review.yml` |
| Cloud Monitoring dashboard | ✅ live metrics (RPS, p95, 5xx, instances) | screenshot en `docs/screenshots/dashboard-overview.jpg` |

---

## Future improvements (roadmap post-v1.0)

Cosas que **no están en v1.0** pero que en producción real haría a continuación. Las dejo escritas para mostrar awareness de cómo escalar esto:

- **Workload Identity Federation** en lugar de JSON keys del SA. WIF deja a GitHub Actions autenticar a GCP vía OIDC sin manejar secrets de larga duración. Más seguro, menos rotaciones.
- **Backend GCS para Terraform state** (`backend "gcs"`). El state local funciona para un dev solo; para equipo se mueve a un bucket GCS con versioning + locking via Cloud Storage.
- **Alertas de Cloud Monitoring** (PagerDuty / email). El dashboard ya muestra p95 y 5xx; falta crear `google_monitoring_alert_policy` que avise si p95 > 1s sostenido o 5xx > 5%.
- **Modelo serializado** (pickle pre-entrenado en la imagen Docker). Hoy el modelo entrena al primer `/predict` (lazy bootstrap, ~2s). En producción real el pickle se genera en el build y se carga al startup → sin training en producción.
- **Versionado del modelo** (model registry). MLflow o un GCS bucket con tags. Cada nueva versión del modelo trackeable.
- **Tests de integración contra Cloud Run de staging** antes del deploy a prod (canary deploy con Cloud Run traffic splitting).
- **Logs estructurados** (JSON) con `structlog` para que Cloud Logging filtre por campos específicos.
- **Rate limiting** en el API (FastAPI middleware) para protección DoS.

---

## Envío del challenge

`POST` único al endpoint de Advana con el body fijo que pide el enunciado:

```bash
curl -X POST "https://advana-challenge-check-api-cr-k4hdbggvoq-uc.a.run.app/software-engineer" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Jose Luis Santiago Marquez",
    "mail": "jlsantiago691@gmail.com",
    "github_url": "https://github.com/joseluis911/latam_challenge.git",
    "api_url": "https://latam-delay-api-108332844354.us-central1.run.app"
  }'
```

Respuesta esperada:

```json
{
  "status": "OK",
  "detail": "your request was received"
}
```

> ⚠️ **Solo se manda UNA vez** (lo subraya el README de LATAM). Antes de mandarlo, validar:
> 1. El repo es público (`Settings → General → Danger Zone → Change repository visibility`).
> 2. `main` tiene el merge final con tag `v1.0.0`.
> 3. La URL del API responde `{"status":"OK"}` en `/health`.
> 4. `make model-test`, `make api-test`, `make stress-test` corren verde en local.
> 5. El branch `feature/*` siguen vivos (no borrados — lo pide la regla #2).
