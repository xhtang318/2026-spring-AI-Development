PORT ?= 8000

.PHONY: serve

serve:
	uv run uvicorn leaderboard.app:app --host 0.0.0.0 --port $(PORT)
