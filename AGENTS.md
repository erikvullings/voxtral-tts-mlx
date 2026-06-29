# Environment Setup
- Local macOS capabilities and optimized CLI tools are mapped in `~/.config/ai/tools.md`. Read this file to use optimized search/replace and parsing binaries.

# Repository Guidelines

## Project Structure & Module Organization

This repository is a **single-file Python service** wrapping the Voxtral TTS model as a FastAPI API. All core logic lives in one module:

| Path | Purpose |
|---|---|
| `server.py` | Entire application — models, routes, engine/aligner logic |
| `pyproject.toml` | Project metadata and dependencies (managed by **uv**) |
| `main.py` | Entry-point stub (not actively used) |
| `generated_lessons/` | Output: generated audio (`.mp3`) and transcripts (`.json`) |
| `voices/` | Voice reference files for cloning |
| `.github/workflows/macos-ci.yml` | macOS CI workflow |
| `scripts/install-mac.sh` | One-command macOS setup helper |

## Build, Test, and Development Commands

Install dependencies via **uv** (Python package manager):

```bash
uv sync                          # Install deps into .venv
uv run server                    # Start FastAPI on localhost:8000
```

The server runs with `reload=True`, so code changes are picked up automatically. Swagger UI is available at `/docs`.

No formal test suite exists yet. Verify manually or via `curl` against the documented endpoints.

## Coding Style & Naming Conventions

- **Python 3.14** (see `.python-version`).
- **Indentation:** 4 spaces, no tabs.
- **Formatting:** PEP 8–inspired. `snake_case` for functions/variables; `PascalCase` for classes.
- **Type hints:** Enabled throughout — annotate parameters and return types using modern syntax (e.g., `list[str]`).
- **Imports:** Standard library → third-party → local, alphabetical within each group.

## Testing Guidelines

When tests are added:

- Use **pytest** (`uv add --dev pytest`).
- Place files alongside `server.py` or in a top-level `tests/` directory.
- Name files `test_*.py` and functions `test_*`.
- Prioritize coverage of route handlers (`/v1/audio/speech`, `/v1/voxtral/speech`, `/v1/voxtral/transcript`).

## Commit & Pull Request Guidelines

The project follows **[Conventional Commits](https://www.conventionalcommits.org/)**:

| Prefix | Meaning | Example |
|---|---|---|
| `feat:` | New functionality | `feat: add audio cleanup method to trim silence` |
| `fix:` | Bug fix | `fix: improve audio processing logic for silence handling` |
| `docs:` | Documentation change | `docs: update README to clarify platform support` |
| `refactor:` | Restructuring (no behavior change) | `refactor: restructure code for improved readability` |
| `chore:` | Maintenance | `chore: initial version` |

**Pull requests** should include: a clear summary, linked issues (`Closes #N`), verification steps (endpoint + payload), and screenshots or sample output where relevant.

## macOS Setup & CI

- **First-time install:** `brew install rubberband && bash scripts/install-mac.sh` or `brew install rubberband && uv sync`.
- `rubberband` is a required system library for `pyrubberband` (pitch-preserving speed adjustment); `uv sync` will fail without it.
- **CI:** the macOS workflow in `.github/workflows/macos-ci.yml` validates on every push/PR.
