"""Compatibility shim for legacy server entrypoint.

Prefer importing/running `server_voxtral:app` directly.
"""

from server_voxtral import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_voxtral:app", host="0.0.0.0", port=8000, reload=True)
