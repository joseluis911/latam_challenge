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

- `main` → releases oficiales para review.
- `develop` → integración.
- `feature/*` → cada parte del challenge (`feature/part-1-model`, `feature/part-2-api`, …).
- Las ramas de desarrollo **no se borran** (lo pide el enunciado).

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

**Objetivo:** exponer el modelo vía FastAPI y pasar `make api-test`.

- Endpoint(s):
  - `POST /predict` — _por completar_
  - `GET /health`
- Validación de input: _por completar._

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
