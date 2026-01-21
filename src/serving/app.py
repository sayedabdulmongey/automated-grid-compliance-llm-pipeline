"""
FastAPI Inference Server for Grid Compliance QA Model

Serves the fine-tuned Qwen 2.5-7B model for grid compliance questions.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# =============================================================================
# Configuration
# =============================================================================

MODEL_ID = os.getenv("MODEL_ID", "sayedsalem/qwen2.5-7b-grid-compliance")
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

SYSTEM_PROMPT = """You are a UK Grid Compliance Expert specializing in G99, UKPN EDS, and SPEN EV Fleet regulations. Provide accurate, technical answers based on official documentation."""

# Global model and tokenizer
model = None
tokenizer = None


# =============================================================================
# Pydantic Models
# =============================================================================


class QuestionRequest(BaseModel):
    """Request model for asking a compliance question."""

    question: str = Field(
        ..., min_length=5, description="The compliance question to ask"
    )
    max_tokens: Optional[int] = Field(
        256, ge=32, le=512, description="Max tokens to generate"
    )
    temperature: Optional[float] = Field(
        0.7, ge=0.1, le=1.5, description="Sampling temperature"
    )


class AnswerResponse(BaseModel):
    """Response model containing the answer."""

    question: str
    answer: str
    model: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    device: str
    model_id: str


# =============================================================================
# Model Loading
# =============================================================================


def load_model():
    """Load the fine-tuned model with QLoRA adapters."""
    global model, tokenizer

    print(f"Loading model from {MODEL_ID}...")
    print(f"Device: {DEVICE}")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with quantization
    if DEVICE == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        # CPU fallback (slower, no quantization)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

    model.eval()
    print(
        f"Model loaded successfully! Memory: {model.get_memory_footprint() / 1e9:.2f} GB"
    )


# =============================================================================
# Inference
# =============================================================================


def generate_answer(
    question: str, max_tokens: int = 256, temperature: float = 0.7
) -> str:
    """Generate an answer for the given question."""
    # Format prompt using Qwen ChatML template
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{question}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract assistant response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
        response = response.replace("<|im_end|>", "").strip()

    return response


# =============================================================================
# FastAPI App
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    load_model()
    yield
    # Cleanup (if needed)
    print("Shutting down...")


app = FastAPI(
    title="Grid Compliance QA API",
    description="API for answering UK grid compliance questions using a fine-tuned Qwen 2.5-7B model.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Grid Compliance QA API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the model is loaded and ready."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=DEVICE,
        model_id=MODEL_ID,
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a grid compliance question and get an answer.

    Example questions:
    - "What is the minimum Registered Capacity for an Embedded Medium Power Station?"
    - "What nominal LV voltage is used for calculating all LV protection settings?"
    - "What is the 30% rule for EV fleet load management?"
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        answer = generate_answer(
            question=request.question,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    latency_ms = (time.time() - start_time) * 1000

    return AnswerResponse(
        question=request.question,
        answer=answer,
        model=MODEL_ID,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/batch", response_model=list[AnswerResponse])
async def batch_questions(requests: list[QuestionRequest]):
    """Process multiple questions in a batch."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    responses = []
    for req in requests:
        start_time = time.time()
        answer = generate_answer(
            question=req.question,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        latency_ms = (time.time() - start_time) * 1000

        responses.append(
            AnswerResponse(
                question=req.question,
                answer=answer,
                model=MODEL_ID,
                latency_ms=round(latency_ms, 2),
            )
        )

    return responses


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
