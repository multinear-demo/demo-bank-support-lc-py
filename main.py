"""
Entry point for the Bank Customer Support RAG application.
Starts the FastAPI server using uvicorn with hot-reload enabled for development.
The application runs on http://localhost:8080 and serves both the API endpoints
and static frontend files.
"""

import uvicorn


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8080, reload=True, use_colors=True)
