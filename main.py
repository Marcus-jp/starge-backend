from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.predict_router import router as predict_router

app = FastAPI(title="STARGE Medical API")

# Allow Flutter app connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include router
app.include_router(predict_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "STARGE backend running"}
