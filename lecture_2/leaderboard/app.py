"""FastAPI leaderboard application for resume scoring."""

import csv
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .database import DEFAULT_DB_PATH, add_submission, get_all_submissions, init_db, reset_db

API_KEY = os.environ.get("LEADERBOARD_API_KEY", "lecture2-secret-key")

DATA_DIR = Path(__file__).parent.parent / "data"
TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="Resume Scoring Leaderboard")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Load valid resume IDs on startup
VALID_RESUME_IDS: set[str] = set()


@app.on_event("startup")
def startup():
    init_db(DEFAULT_DB_PATH)
    csv_path = DATA_DIR / "resumes_final.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                VALID_RESUME_IDS.add(row["ID"])


def _check_api_key(x_api_key: str | None):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class SubmissionRequest(BaseModel):
    team_name: str
    resume_id: str
    score: float


@app.get("/", response_class=HTMLResponse)
async def leaderboard_page(request: Request):
    submissions = get_all_submissions(DEFAULT_DB_PATH)
    return templates.TemplateResponse(
        "leaderboard.html", {"request": request, "submissions": submissions}
    )


@app.post("/api/submit")
async def submit_score(
    body: SubmissionRequest, x_api_key: str | None = Header(default=None)
):
    _check_api_key(x_api_key)

    if VALID_RESUME_IDS and body.resume_id not in VALID_RESUME_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid resume_id: {body.resume_id}",
        )

    if not 0 <= body.score <= 100:
        raise HTTPException(
            status_code=400, detail="Score must be between 0 and 100"
        )

    add_submission(DEFAULT_DB_PATH, body.team_name, body.resume_id, body.score)
    return {"status": "ok", "team_name": body.team_name, "resume_id": body.resume_id, "score": body.score}


@app.post("/api/reset")
async def reset_leaderboard(x_api_key: str | None = Header(default=None)):
    _check_api_key(x_api_key)
    reset_db(DEFAULT_DB_PATH)
    return {"status": "ok", "message": "Leaderboard reset"}


@app.get("/api/submissions")
async def get_submissions():
    return get_all_submissions(DEFAULT_DB_PATH)


@app.get("/api/health")
async def health():
    return {"status": "healthy"}
