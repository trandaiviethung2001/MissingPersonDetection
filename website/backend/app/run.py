import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "main:app",          # change this if your file is different
        host="0.0.0.0",
        port=8000,
        reload=False,        # set True only for development
        workers=1,           # important for camera / GPU pipelines
        log_level="info"
    )