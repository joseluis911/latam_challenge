# LATAM — Software Engineer (ML & LLMs) Challenge

Documentación de la solución al challenge.

---

## Autor

- **Nombre:** Jose Luis Santiago Marquez
- **Mail:** jlsantiago691@gmail.com
- **Repositorio:** https://github.com/joseluis911/latam_challenge
- **API desplegada:** _<api_url>_ _(disponible al completar Part III)_

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

**Proveedor:** _GCP (Cloud Run) — por confirmar._

- URL del API: _<pegar en `Makefile` línea 26>_
- Pasos de despliegue: _por completar._
- `make stress-test` ✅ / ❌

---

## Part IV — CI/CD (`.github/workflows/`)

- `ci.yml` → linter + tests (`model-test`, `api-test`) en cada PR a `develop`/`main`.
- `cd.yml` → build de imagen + deploy a Cloud Run en merge a `main`.
- Secrets requeridos: _por completar._

---

## Cómo correr local

```bash
pip install -r requirements.txt -r requirements-dev.txt -r requirements-test.txt
make model-test
make api-test
make stress-test
```

---

## Envío del challenge

`POST` único a `https://advana-challenge-check-api-cr-k4hdbggvoq-uc.a.run.app/software-engineer` con el body indicado en el README.
