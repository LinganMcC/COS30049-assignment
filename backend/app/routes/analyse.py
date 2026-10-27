"""POST /api/analyse - the one endpoint that runs the model."""
from fastapi import APIRouter, Request

from app.config import get_settings
from app.errors import AppError
from app.ml.service import Analysis, NoScorableSentences
from app.schemas import AnalyseRequest, AnalyseResponse, ErrorResponse

router = APIRouter(prefix='/api', tags=['analysis'])


def _error(description: str, code: str, message: str) -> dict:
    """An /docs entry showing the shared error format for one status code."""
    example = {'error': {'code': code, 'message': message, 'details': []}}
    return {'model': ErrorResponse, 'description': description,
            'content': {'application/json': {'example': example}}}


@router.post(
    '/analyse',
    response_model=AnalyseResponse,
    summary='Score a text for AI-generated content',
    description=(
        'Splits the text into sentences, scores every sentence of 5 or more words '
        'with the transformer, and combines them into a document score with a ~95% '
        'confidence range. Look-alike and invisible Unicode characters are normalised '
        'first. The submitted text is never stored or logged.'),
    responses={
        413: _error('Body larger than the size limit.', 'payload_too_large',
                    'The request body is larger than 32 KB.'),
        422: _error('Text missing, too long, over the word limit, or with no sentence '
                    'of 5+ words.', 'invalid_input', 'The request is not valid.'),
        503: _error('Too many analyses running, or this one timed out.', 'server_busy',
                    'The detector is busy. Please try again in a few seconds.'),
        500: _error('Unexpected server error.', 'internal_error',
                    'Something went wrong on our side. Please try again.'),
    },
)
async def analyse(body: AnalyseRequest, request: Request) -> AnalyseResponse:
    detector = request.app.state.detector
    try:
        if get_settings().dos_protection:
            result: Analysis = await request.app.state.gate.run(detector.analyse, body.text)
        else:
            # VULNERABLE BASELINE, for the video only (DOS_PROTECTION=false):
            # the model runs directly on the event loop, so while it works no
            # other request - from any user - can be answered.
            result = detector.analyse(body.text)
    except NoScorableSentences as exc:
        raise AppError(422, 'no_scorable_sentences', str(exc)) from None

    return AnalyseResponse(document=result.document, sentences=result.sentences,
                           excluded_sentences=result.excluded_sentences,
                           characters_normalised=result.characters_normalised)
