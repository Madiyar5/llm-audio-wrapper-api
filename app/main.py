from fastapi import FastAPI
from fastapi.responses import JSONResponse
 
from app.config import settings
from app.routes.analyze import router as analyze_router
 
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
 
app.include_router(analyze_router)
 
 
@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
        "model": settings.model_name,
    }
 
 
@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": f"Internal server error: {str(exc)}"
        },
    )

