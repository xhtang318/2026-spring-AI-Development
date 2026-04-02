"""FastAPI leaderboard application for resume scoring."""

import csv
import os
import random
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

    # Pivot: rows=resume_ids, cols=team_names, cells={score, submitted_at}
    team_names: list[str] = []
    resume_ids: list[str] = []
    seen_teams: set[str] = set()
    seen_resumes: set[str] = set()
    grid: dict[tuple[str, str], dict] = {}

    for s in submissions:
        tid, rid = s["team_name"], s["resume_id"]
        if tid not in seen_teams:
            team_names.append(tid)
            seen_teams.add(tid)
        if rid not in seen_resumes:
            resume_ids.append(rid)
            seen_resumes.add(rid)
        grid[(rid, tid)] = {"score": s["score"], "submitted_at": s["submitted_at"]}

    team_names.sort()
    resume_ids.sort()

    return templates.TemplateResponse(
        request=request,
        name="leaderboard.html",
        context={
            "team_names": team_names,
            "resume_ids": resume_ids,
            "grid": grid,
            "api_key": API_KEY,
        },
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


@app.post("/api/seed")
async def seed_test_data(x_api_key: str | None = Header(default=None)):
    """Seed the database with test data: 15 teams x 10 resume IDs."""
    _check_api_key(x_api_key)
    teams = [f"Team {name}" for name in [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon",
        "Zeta", "Eta", "Theta", "Iota", "Kappa",
        "Lambda", "Mu", "Nu", "Xi", "Omicron",
    ]]
    resume_ids = sorted(random.sample(list(VALID_RESUME_IDS), 10)) if len(VALID_RESUME_IDS) >= 10 else [str(i) for i in range(10)]
    for team in teams:
        for rid in resume_ids:
            score = round(random.uniform(30, 99), 1)
            add_submission(DEFAULT_DB_PATH, team, rid, score)
    return {"status": "ok", "teams": len(teams), "resumes": len(resume_ids)}
