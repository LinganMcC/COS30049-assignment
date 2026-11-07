# Inkprint: AI-generated writing detector

COS30049 Computing Technology Innovation Project, Assignment 3. This folder is the web app;
the Assignment 2 pipeline that produced the model lives in the folders above it.

Inkprint takes a passage of English writing and estimates how likely it is to be AI-generated.
It scores every sentence, shades the passage by score, and reports an overall estimate with a
range showing how much the sentence scores agree. Underneath is the **Logistic Regression model from our Assignment 2**: 15
sentence-level style features, sqrt-length pooling and a 0.5 threshold, loaded unchanged.

![Result screen](docs/screenshot-result.png)

---

## 1. Run it

You need **Python 3.15** and **Node.js 20.19+ or 22.12+**. Every Python package is pinned to a version
with a ready-made build for Python 3.15, so nothing has to be compiled. Use two terminals.

**Terminal 1: back end**

Git Bash (Windows), macOS or Linux:
```bash
cd webapp/backend
py -3.15 -m venv .venv               # macOS/Linux: python3.15 -m venv .venv
source .venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
python --version                     # must print Python 3.15.x
pip install -r requirements.txt      # also installs the spaCy English model
cp .env.example .env
python -m app                        # API on http://127.0.0.1:8000, docs at /docs
```

Windows PowerShell: the same, but activate with `.venv\Scripts\Activate.ps1` and copy with
`copy .env.example .env`. (If scripts are blocked, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.)

**Terminal 2: front end**

```bash
cd webapp/frontend
npm ci                               # exact versions from package-lock.json
cp .env.example .env                 # PowerShell: copy .env.example .env
npm run dev                          # http://localhost:5173
```

Open **http://localhost:5173** (use `localhost`, not `127.0.0.1`: CORS allows exactly
`http://localhost:5173`).

Tests: `cd webapp/backend && python -m pytest` (32 tests). Lint: `cd webapp/frontend && npm run lint`.
Speed check: with the API running, `python scripts/measure_latency.py`.

> Start the API with `python -m app`, not plain `uvicorn`: it turns off two uvicorn defaults we
> don't want (the `Server` header and trusting client-supplied `X-Forwarded-For`), see section 7B.

> **If `pip install` starts "Building wheel" or "Preparing metadata" for numpy or spaCy**, the
> virtual environment is on the wrong Python version. Delete `.venv` and recreate it with
> `py -3.15 -m venv .venv`.

---

## 2. Architecture

```mermaid
flowchart LR
  U[Browser] -->|types or opens .txt| FE[React + MUI + D3<br/>localhost:5173]
  FE -->|POST /api/detect {text}| MW
  FE -->|GET /api/model-info| MW
  subgraph API[FastAPI · 127.0.0.1:8000]
    MW[CORS → security headers → body limit 128 KB → log] --> V[Pydantic validation<br/>20–1,000 words]
    V --> G[Analysis guard<br/>max 4 at once, else 503]
    G --> TP[worker thread]
    TP --> P[normalise → clean → spaCy split]
    P --> F[15 features per sentence]
    F --> M[(Logistic Regression<br/>loaded once at start-up)]
    M --> AGG[sqrt-length pooling + range + explanations]
  end
  AGG -->|JSON result| FE
```

**One request, end to end**

1. `TextInputForm` checks the text locally (20–1,000 words) and shows any message beside the field.
2. `api/client.js` sends `POST /api/detect` with `{"text": "..."}`.
3. Middleware applies CORS, adds security headers, and rejects bodies over 128 KB with **413**.
4. `DetectRequest` (Pydantic) rejects bad input with **422**; nothing reaches the model.
5. The analysis guard returns **503** if 4 analyses are already running; otherwise the work runs in a thread.
6. `app/ml/detector.py`: normalise invisible and look-alike characters → Assignment 2 cleaning and spaCy
   splitting → 15 features → sentence probabilities → pooled score → sentence-agreement range → feature pushes.
7. The page shows the verdict, the shaded passage, two charts and the sentence detail.

**Same processing as training.** `app/ml/features.py` and the cleaning and splitting in
`app/ml/text_pipeline.py` are copied from Assignment 2. We checked it on Python 3.15 against the original
Assignment 2 code: on 16 essays and on 2 passages full of accented names and words (Việt, Hà Nội, résumé,
Señor Pérez), the app produced **identical feature values** and the same document scores. The only step
added before the model is the evasion normalisation (section 8), which changes nothing in ordinary text.

**Speed** (`scripts/measure_latency.py`, 30 runs over HTTP on a 2-core machine, exactly 500 words under
the app's counting rule): median **57 ms**, 95th percentile 70 ms, slowest 111 ms. 1,000 words: median
105 ms, slowest 124 ms. Assignment 1's target was under 2 s for 500 words. This times the API; the browser
adds drawing time on top, which you can see is small when you run a check.

---

## 3. Project structure

```
webapp/backend/
  app/__main__.py          `python -m app`: starts uvicorn with proxy headers OFF
  app/main.py              app factory: lifespan (model loaded once), middleware, CORS, routes
  app/config.py            every setting from environment variables / .env
  app/schemas.py           Pydantic request/response models with limits and /docs examples
  app/errors.py            ONE error format for every status code
  app/security.py          analysis guard, body-size limit, security headers
  app/routers/detect.py    POST /api/detect, GET /api/model-info, GET /api/health
  app/ml/detector.py       model loading, scoring, pooling, range, explanations
  app/ml/text_pipeline.py  evasion normalisation + Assignment 2 cleaning and splitting
  app/ml/features.py       the 15 Assignment 2 features (unchanged)
  model/                   frozen model, config, Assignment 2 test metrics
  tests/                   pytest: API behaviour and every defence
  scripts/attacks/         dos_attack.py, evasion_attack.py
  scripts/measure_latency.py  500-word speed check
  scripts/data/samples.json  AI and human samples used by the evasion experiment
webapp/frontend/
  src/main.jsx             theme, router, session-history provider
  src/App.jsx              routes /, /about, * (404), inside an ErrorBoundary
  src/api/client.js        the only file that calls the back end
  src/pages/               DetectPage (idle/loading/success/error), AboutPage, NotFoundPage
  src/components/          form, verdict, highlighted passage, sentence detail, two D3 charts
```

---

## 4. How it meets the brief

| Requirement (spec section) | Where |
|---|---|
| Input validated before sending, message beside the field (2.1) | `TextInputForm.jsx`, `lib/text.js`; `DetectPage.runCheck` validates again, so "Try again" can't skip it |
| Result and confidence most prominent (2.1, 2.5) | `ResultSummary.jsx`: verdict ("Uncertain result" when the range crosses the line), score, sentence-agreement range, sticky beside the input |
| Loading, success, error states (2.1) | `DetectPage.jsx`, `ErrorPanel` (network, timeout, 413, 503 with countdown, 500) |
| 404 page and crash fallback (2.1) | `NotFoundPage.jsx`, `ErrorBoundary.jsx` |
| Two visualisations, different questions, interactive (2.1) | `SentenceChart` (this text: select a bar to find the sentence, ← →) and `HistoryChart` (all checks on this page: filter by verdict, select a dot for its details); both have tooltips and a table view |
| Consistent design, responsive, accessible (2.1) | `theme.js`, `styles.css`; labels on every input, keyboard access to sentences and chart marks, visible focus, colour-blind-checked palette, scores shown as text as well as colour |
| More than one input type (A2 brief, innovation) | Paste, or open a `.txt` file (read in the browser, never uploaded) |
| Pydantic models with constraints (2.2) | `schemas.py` (`min_length`, `max_length`, word-count validator, `extra="forbid"`) |
| Two HTTP methods, genuine purposes (2.2) | `POST /api/detect` analyses; `GET /api/model-info` feeds the About page; `GET /api/health` |
| Correct status codes, one JSON error format (2.2) | `errors.py`: 400, 404, 405, 413, 422, 500, 503 all as `{"error": {"code", "message", "fields"}}` |
| Model loaded once (2.2) | `lifespan` in `main.py` |
| User input never rendered as HTML (2.3) | All text rendered as React text; chart tooltips are React elements, not `innerHTML` |
| Size limits enforced on the back end (2.3) | 20–1,000 words (Pydantic); safety nets of 20,000 characters and a 128 KB body |
| CORS only our front end (2.3) | `FRONTEND_ORIGIN` → exactly one origin; methods GET/POST; header Content-Type; outermost middleware so even a 413 reaches the page |
| No secrets in code (2.3) | `.env` is git-ignored; `.env.example` documents every setting |
| Pinned versions (2.4) | `requirements.txt` (`==`), `package.json` (exact) + `package-lock.json` |

**What counts as a word.** For the limits, a word is a whitespace-separated piece containing a letter
or digit, so *résumé*, *work-life*, *don't* and *2024* each count once and a lone dash doesn't. The front
end (`countWords`) and back end (`count_input_words`) use the same rule. This is separate from the
model's own `n_words` feature, which keeps its Assignment 2 definition so predictions don't change.

**Input limits, and why.** *1,000 words* is the limit agreed in Assignment 1. *At least 20 words*:
with fewer there are only one or two sentences, so the score rests on a single sentence and the range
means nothing. *20,000 characters* and a *128 KB body* are safety nets only: English averages about
6 characters per word including spaces, so 1,000 real words never reach them, but they stop a single
50,000-letter "word" or a huge body reaching the model.

**Privacy.** Submitted text is not stored or logged anywhere. The API log records method, path,
status and time only, and validation errors don't echo the input. The comparison chart keeps only
numbers (score, range, verdict, counts) in the page's memory: no text, no sentence results, nothing in
browser storage. **Clear** returns the page to its starting state: text, result and the session
chart's scores all go. Reloading or closing the page does the same.

---

## 5. API

Full schema with examples: **http://127.0.0.1:8000/docs**

| Method | Path | Purpose | Success | Errors |
|---|---|---|---|---|
| POST | `/api/detect` | Score a passage | 200 `DetectResponse` | 400, 413, 422, 503, 500 |
| GET | `/api/model-info` | Model, features, test metrics, limits | 200 | 503, 500 |
| GET | `/api/health` | Liveness | 200 | none |

`DetectResponse` contains `label`, `band`, `score`, `range {low, high}`, `certainty`,
`sentences[]` (score, band, top features each way), `passage[]` (segments for highlighting),
`style_summary[]` and `normalisation` (hidden characters removed).

**How the numbers are made**
- *Sentence score*: the Logistic Regression probability for that sentence.
- *Document score*: sentence scores averaged with weights √(words): Assignment 2's `sqrt_length` pooling.
- *Sentence-agreement range*: score ± 1.96·√((variance + 0.01) / n), the team's formula from
  `transformer/train_transformer.py`, using the weighted variance and Kish's effective n. It widens
  for short texts or when sentences disagree. **It is a heuristic, not a validated 95% confidence
  interval**: sentences from one document are not independent, and we have not measured how often
  the true label falls inside it. When it crosses the 0.5 line the interface says "Uncertain result".
- *Bands*: Human ≤ 25%, Possibly AI ≤ 50%, Likely AI ≤ 75%, Very likely AI above (team mockup).
- *Feature pushes*: each feature's contribution to a sentence's **log-odds** (standardised value ×
  coefficient), as in Assignment 2's error analysis. They are not percentage points, and the document-level
  list averages them, so it shows associations rather than an exact breakdown of the document percentage.
- *Sentence labels*: the model was trained with every sentence inheriting its essay's label
  (`label_origin: document_inherited`), so a sentence score is an estimate of how AI-like its style
  is, not a verified label for that sentence. The About page says so.

---

## 6. Threat model

**Who would attack it?** Mainly students or writers who used AI and want their text to pass as
human-written, and people tuning AI output to beat detectors in general. A smaller group
just wants to knock the service over, or abuse it as free computing.

**What do they want?**
1. A "human" verdict for AI-written text (the main goal, named in the spec for this topic).
2. To learn how the detector decides, by submitting edits over and over until one passes.
3. To make the service unusable for others.

**What can they actually do?** They don't need our interface: anyone can send HTTP requests
with curl or a script, with any body, any headers and as many requests as they like. They can
send any Unicode the browser can paste. They cannot reach the server's files or a database
(there isn't one).

**Risks from spec section 2.7 that apply to this design**

| Risk | Applies? | Why |
|---|---|---|
| Denial of service | **Yes: nominated (7A)** | Each analysis runs spaCy and the model, far more costly than sending a request. We also found Assignment 2's HTML regex backtracks catastrophically. |
| Misconfiguration / vulnerable dependencies | **Yes: nominated (7B)** | Applies to every application; we had real findings. |
| Missing rate limiting | Yes, not nominated | Goal 2 above (repeated probing). Only two defences are marked; the concurrency cap and input limits already bound the cost of a flood, so we chose the defences with the most measurable risk. |
| Unsafe model loading | Low | The model is our own file, committed to our repo and loaded from disk; it is never downloaded or uploaded by users. |
| Detection evasion (2.8) | **Yes** | Goal 1: hidden and look-alike characters change what the model sees. |
| SQL injection | No | No database. |
| Unsafe file upload | No | `.txt` files are read in the browser and sent as plain JSON text; the server accepts no files. |
| Command injection | No | The back end never runs system commands. |
| SSRF | No | The back end makes no outbound requests from user input. |
| Broken access control | No | No accounts and no stored records to reach. |

---

## 7. Security defences (section 2.7)

**The two defences nominated for marking:**

| | Defence | Level | Why our threat model needs it |
|---|---|---|---|
| **A** | Denial of service | Advanced | Goal 3: a cheap request can cost us a lot of CPU, and we found one that costs minutes |
| **B** | Misconfiguration and vulnerable dependencies | Intermediate | Every attacker benefits from known library flaws and loose defaults; we found real ones |

The denial-of-service defence can be switched off in `.env` (`DOS_GUARD_ENABLED=false`) so the video can
show the attack working without it and failing with it. Restart the API after changing `.env`, and set
it back to `true` afterwards.

### A. Denial of service (Advanced)

**The attack we found.** Assignment 2's `clean_text()` strips HTML with
`(?:\s+[^<>]*?)?\s*/?>`. Whitespace can be split between three parts in every possible way,
so text like `<p` followed by many form feeds (which survive the earlier whitespace collapse)
makes the regex engine try them all: cubic time. We measured **1,500 characters ≈ 6 s,
2,000 ≈ 15 s, 4,000 ≈ 118 s**. A 12,000-character request (inside every size limit) would hold
a CPU for most of an hour.

**The defence, in three layers** (`routers/detect.py`, `security.py`, `text_pipeline.py`):
1. **Fix the cause**: an equivalent linear regex `(?:/|\s[^<>]*)?>`. It accepts exactly the same
   tags, so cleaning output is unchanged: identical on 200,000 random strings, and a test checks 20,000 every run.
2. **Keep the server responsive**: analysis runs in a worker thread, not on the event loop.
3. **Cap the work**: at most `MAX_CONCURRENT_ANALYSES` (4) at once. Extra requests get an
   immediate **503** with `Retry-After`, instead of queueing up a backlog.

**Evidence: a controlled comparison.** Same machine, same command
(`python scripts/attacks/dos_attack.py --size 1500 --flood 30`), only `DOS_GUARD_ENABLED` changed:

| | Defence OFF | Defence ON |
|---|---|---|
| One expensive request (1.8 KB) | other users waited up to **4,828 ms**; the attack took 5.1 s | other users waited at most **4 ms**; the attack took under 0.1 s |
| Flood of 30 analyses at once | all 30 queued; health check 250 ms | 4 served, **26 rejected with 503**; health check 60 ms |

Optional variation, defence ON only: `--size 11000` is also handled in about 0.1 s. Don't run it with the
defence OFF: that request would freeze the server for most of an hour.

**Cost.** Under a burst some real users get "busy, try again" (503) rather than waiting. The
front end shows a retry button. Four slots suits a 4-core laptop; raise it on bigger machines.

### B. Misconfiguration and vulnerable dependencies (Intermediate)

What we found and fixed:

| Finding | Tool | Fix |
|---|---|---|
| axios 1.13.1: SSRF, prototype-pollution and DoS advisories | `npm audit` (high) | upgraded to 1.21.0 |
| react-router-dom 7.9.5, vite 7.1.12 | `npm audit` (high) | upgraded to 7.18.5 and 7.3.7 |
| starlette 0.49.3: PYSEC-2026-2280/2281 | `pip-audit` | upgraded to 1.10.0 (FastAPI 0.162.0) |
| pytest 8.4.2: PYSEC-2026-1845 | `pip-audit` | upgraded to 9.2.0 |
| Uvicorn trusts `X-Forwarded-For` from localhost by default, so a local client can fake its address | review + our own test | `python -m app` runs with `proxy_headers=False` |
| No security headers by default | review | `SecurityHeadersMiddleware`: nosniff, `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy`; API also gets `CSP: default-src 'none'` and `Cache-Control: no-store` |
| Server header reveals the software | review | `server_header=False` |
| Interactive docs always on | review | `SHOW_DOCS=false` turns off `/docs` and `/openapi.json` in production |
| CORS was applied after the body-size check, so a 413 reached the browser without a CORS header and showed as a network error | our own test | CORS is now the outermost middleware |

Both `npm audit` and `pip-audit -r requirements.txt` report **0 known vulnerabilities** (checked on
Python 3.15). `en_core_web_sm` is not on PyPI, so pip-audit cannot check it; it is pinned to an exact
release URL.

**Still to do by the team:** run OWASP ZAP (baseline scan) against `http://localhost:5173` and
`http://127.0.0.1:8000`, then record each finding and either fix it or justify it in this table.

**Cost.** Pinned, audited versions must be re-audited and upgraded deliberately; the stricter headers mean
the API can't be embedded in other sites (which we don't need).

---

## 8. Advanced security (section 2.8): detection evasion

**Attack.** Change how AI text is *encoded* without changing how it *looks*. The features count
only ASCII letters (`[A-Za-z']+`), so a zero-width space or a Cyrillic "а" splits words in two.
That lowers *Long words*, the strongest AI-direction feature, and the score falls toward human.

**Defence.** `normalise_for_evasion()` runs on the text exactly as received, before the Assignment 2
cleaning. It removes invisible format characters (Unicode category Cf, variation selectors, Hangul fillers),
maps cross-script look-alikes (Unicode TR39 confusables for Cyrillic, Greek and IPA) to Latin, and removes
a separate combining accent typed straight after a plain a–z letter. Genuine accented letters (café,
résumé, Việt) are single characters, so they pass through untouched and the model sees exactly what
Assignment 2 would have produced. The interface reports anything removed in neutral terms, because such
characters can also come from copying text out of some apps.

**Measured** with `python scripts/attacks/evasion_attack.py` on **10 AI-written passages** (all 10 score
≥ 0.5 before any attack). Random attacks were drawn 5 times per passage with different seeds (50 variants);
deterministic attacks give the same text every time, so they were run once per passage (10 variants).
*Success* = scored ≥ 0.5 before the attack and < 0.5 after it. We ran it twice: in-process, and through the
real `POST /api/detect` endpoint with `EVASION_DEFENCE_ENABLED` false and then true (`--api`). Both gave
the same numbers:

| Attack | Variants | Defence OFF: success | mean score | Defence ON: success | mean score |
|---|---|---|---|---|---|
| none | | | 0.905 | | 0.905 |
| zero-width spaces in long words | 10 | **100%** | 0.314 | **0%** | 0.905 |
| 10% look-alike letters (random) | 50 | 0% | 0.774 | 0% | 0.905 |
| 30% look-alike letters (random) | 50 | **60%** | 0.486 | **0%** | 0.905 |
| 100% look-alike letters | 10 | **100%** | 0.069 | **0%** | 0.905 |
| combining accent every 4th letter | 10 | **100%** | 0.187 | **0%** | 0.905 |
| zero-width + 50% look-alikes (random) | 50 | **100%** | 0.171 | **0%** | 0.905 |
| visible edits (drop commas and transition words) | 10 | 0% | 0.777 | 0% | 0.777 |

**Interpretation.** For encoding tricks the defence works completely: every attacked text gets
the same score as the original, because normalisation restores the exact characters. It does
**nothing** against real rewriting. Deleting commas and transition words lowered the mean score
from 0.905 to 0.777 with or without the defence. Paraphrasing tools are a stronger version of the
same attack, and only a better model can resist them.

**Cost**, measured on ordinary text (score with the defence on minus off): 10 AI passages, 6 student
essays and 2 passages full of accented names and words all changed by **0.0000**. A passage containing
genuine Greek and Cyrillic words changed by 0.023, because those letters are transliterated. Normalising
700 words takes about 1.1 ms (a full analysis is about 10 ms). A genuine accent typed as a separate
combining character (rare: keyboards produce "é" as one character) is removed and reported.

**Limits of the evidence.** 10 source passages is small, and they were written by an AI assistant for
this test; the human samples are DRCAT essays. A larger run should use AI essays from the DRCAT test set.

---

## 9. Known limitations

- Trained on student essays (DRCAT). On HC3 question answers accuracy falls to about 61% (see the
  About page). The interface says it is weaker outside student essays.
- A score is an estimate, never proof. The footer and About page say so.
- The analysis guard counts in memory: one server process only. Several workers would need a shared
  store such as Redis.
- There is no rate limiting (not one of our two nominated defences), so one client can keep the
  detector busy within the concurrency cap.
- Only English is supported.
- Built and tested for Python 3.15 only.

---

## 10. References

- Bostock, M, Ogievetsky, V & Heer, J 2011, 'D³: data-driven documents', *IEEE Transactions on Visualization and Computer Graphics*, vol. 17, no. 12, pp. 2301–2309.
- Davis, M & Suignard, M 2023, *Unicode Technical Standard #39: Unicode security mechanisms*, Unicode Consortium, viewed 7 November 2026, <https://www.unicode.org/reports/tr39/>.
- Guo, B, Zhang, X, Wang, Z, Jiang, M, Nie, J, Ding, Y, Yue, J & Wu, Y 2023, 'How close is ChatGPT to human experts? Comparison corpus, evaluation, and detection', *arXiv preprint* arXiv:2301.07597.
- Honnibal, M, Montani, I, Van Landeghem, S & Boyd, A 2020, *spaCy: industrial-strength natural language processing in Python*, Zenodo, doi:10.5281/zenodo.1212303.
- Kish, L 1965, *Survey sampling*, John Wiley & Sons, New York.
- OWASP Foundation n.d., *Regular expression denial of service (ReDoS)*, viewed 7 November 2026, <https://owasp.org/www-community/attacks/Regular_expression_Denial_of_Service_-_ReDoS>.
- Pedregosa, F et al. 2011, 'Scikit-learn: machine learning in Python', *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830.
- Ramírez, S n.d., *FastAPI*, viewed 7 November 2026, <https://fastapi.tiangolo.com/>.
- DAIGT V2 Train Dataset (DRCAT) n.d., Kaggle, viewed 7 November 2026, <https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset>.

*(Check these against your unit's Harvard style guide and add the course materials you used.)*

---

## 11. Generative AI statement

**To be completed by the team.** This web application, its tests and the attack scripts were
drafted with the help of an AI assistant (Claude, Anthropic), working from our Assignment 2 code,
the Assignment 3 specification and the unit's materials. Describe in your own words what you
asked for, what you reviewed and changed, and how you checked it. Every member must be able to
explain every part of the code in the video.
