"""FastAPI service exposing the DelayModel as a prediction endpoint."""

import fastapi
import pandas as pd
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

from challenge.model import DelayModel

# Airlines present in the SCL dataset. Used to validate `OPERA` in the request.
KNOWN_OPERAS: set[str] = {
    "Aerolineas Argentinas",
    "Aeromexico",
    "Air Canada",
    "Air France",
    "Alitalia",
    "American Airlines",
    "Austral",
    "Avianca",
    "British Airways",
    "Copa Air",
    "Delta Air",
    "Gol Trans",
    "Grupo LATAM",
    "Iberia",
    "JetSmart SPA",
    "K.L.M.",
    "Lacsa",
    "Latin American Wings",
    "Oceanair Linhas Aereas",
    "Plus Ultra Lineas Aereas",
    "Qantas Airways",
    "Sky Airline",
    "United Airlines",
}

VALID_TIPOVUELO: set[str] = {"I", "N"}


class FlightInput(BaseModel):
    """Input for a single flight to score."""

    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("OPERA")
    def _opera_must_be_known(cls, v: str) -> str:
        if v not in KNOWN_OPERAS:
            raise ValueError(f"Unknown OPERA: {v!r}")
        return v

    @validator("TIPOVUELO")
    def _tipovuelo_must_be_valid(cls, v: str) -> str:
        if v not in VALID_TIPOVUELO:
            raise ValueError(f"TIPOVUELO must be 'I' or 'N', got {v!r}")
        return v

    @validator("MES")
    def _mes_must_be_valid(cls, v: int) -> int:
        if not 1 <= v <= 12:
            raise ValueError(f"MES must be between 1 and 12, got {v}")
        return v


class PredictRequest(BaseModel):
    flights: list[FlightInput]


class PredictResponse(BaseModel):
    predict: list[int]


app = fastapi.FastAPI(
    title="LATAM Flight Delay Prediction API",
    description="Predicts delay (>15 min) for flights at SCL airport.",
    version="1.0.0",
    contact={"name": "Jose Luis Santiago Marquez", "email": "jlsantiago691@gmail.com"},
)

# Single shared model instance: trains lazily on the first prediction.
_model = DelayModel()


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return 400 (Bad Request) for invalid input instead of FastAPI's default 422."""
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


@app.get("/health", status_code=200)
async def get_health() -> dict:
    """Liveness probe."""
    return {"status": "OK"}


@app.post("/predict", status_code=200, response_model=PredictResponse)
async def post_predict(payload: PredictRequest) -> PredictResponse:
    """Predict delay (1) or no delay (0) for each flight in the payload."""
    raw = pd.DataFrame([flight.dict() for flight in payload.flights])
    features = _model.preprocess(raw)
    predictions = _model.predict(features)
    return PredictResponse(predict=predictions)
