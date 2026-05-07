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

**Objetivo:** pasar el notebook `exploration.ipynb` a `model.py` y dejarlo testeable con `make model-test`.

### Modelo elegido

_Por completar._ Justificación de cuál de los modelos propuestos por el DS se eligió y por qué (métricas, balance de clases, etc.).

### Bugs encontrados / corregidos

_Por completar._

### Decisiones de diseño

_Por completar._ (features usadas, manejo de desbalance, encoding, etc.)

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
