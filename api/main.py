from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from db.database import init_db
from data.seed import seed
from api.routes import query, trace, eval, approve, reeval


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs on startup
    init_db()   # create all tables
    seed()      # seed research database
    yield
    # runs on shutdown (nothing to clean up)


app = FastAPI(
    title="Mega AI",
    description="Real-Time Multi-Agent LLM Orchestration and Evaluation System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# register all 5 routes
app.include_router(query.router)
app.include_router(trace.router)
app.include_router(eval.router)
app.include_router(approve.router)
app.include_router(reeval.router)


@app.get("/health")
def health():
    return {"status": "ok"}