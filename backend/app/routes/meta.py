"""GET endpoints: health check and the limits the front end validates against."""
from fastapi import APIRouter, Request

from app.config import get_settings
from app.ml.sentences import MIN_WORDS
from app.ml.service import BANDS
from app.schemas import ConfigResponse, HealthResponse

router = APIRouter(prefix='/api', tags=['meta'])


@router.get(
    '/health',
    response_model=HealthResponse,
    summary='Check the server is up',
    description='Cheap check that never touches the model. Used in the DoS demo to show '
                'the server still answers while analyses are running.',
    responses={200: {'content': {'application/json': {
        'example': {'status': 'ok', 'model_loaded': True}}}}},
)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(status='ok', model_loaded=hasattr(request.app.state, 'detector'))


@router.get(
    '/config',
    response_model=ConfigResponse,
    summary='Input limits and score bands',
    description='The front end reads these so its validation and colour bands always '
                'match the back end. The back end still enforces every limit itself.',
    responses={200: {'content': {'application/json': {'example': {
        'max_words': 1000, 'max_chars': 8000, 'min_sentence_words': 5,
        'bands': [{'name': 'Human', 'upper': 0.25}, {'name': 'Possibly AI', 'upper': 0.5},
                  {'name': 'Likely AI', 'upper': 0.75},
                  {'name': 'Very likely AI', 'upper': 1.0}]}}}}},
)
async def config() -> ConfigResponse:
    settings = get_settings()
    return ConfigResponse(max_words=settings.max_words, max_chars=settings.max_chars,
                          min_sentence_words=MIN_WORDS,
                          bands=[{'name': name, 'upper': upper} for upper, name in BANDS])
