"""
FastAPI Web Interface for Writing Coach
Provides REST API endpoints for writing evaluation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Union
import logging
import asyncio
from datetime import datetime
import tempfile
import os

from coach import WritingCoachSystem, EvaluationResult, print_evaluation_report
from config import get_config, validate_environment

# Get configuration
config = get_config()

# Configure logging
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Writing Coach API",
    description="Automatic English Writing Evaluation System powered by Autogen",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global writing coach instance
coach_system = None


@app.on_event("startup")
async def startup_event():
    """Initialize coach system on startup"""
    global coach_system
    try:
        logger.info("Initializing Writing Coach System...")
        
        # Validate environment configuration
        if not validate_environment():
            raise ValueError("Environment configuration is invalid. Please check your .env file.")
        
        # Initialize with default provider from config
        coach_system = WritingCoachSystem()
        logger.info("✓ Writing Coach System initialized successfully")
        available_providers = config.get_available_providers()
        logger.info(f"✓ Available providers: {', '.join(available_providers)}")
        
        # Show active provider
        logger.info(f"✓ Active provider: {coach_system.provider} with model: {coach_system.model}")
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize Writing Coach System: {e}")
        raise


# ==================== REQUEST/RESPONSE MODELS ====================

class WritingSample(BaseModel):
    """Request model for writing evaluation"""
    text: str = Field(..., min_length=10, description="The writing sample to evaluate")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    title: Optional[str] = Field(None, description="Optional title of the writing")

    class Config:
        example = {
            "text": "The internet have changed how people communicate...",
            "user_id": "student_123",
            "title": "Impact of Internet Communication"
        }


class EvaluationResponse(BaseModel):
    """Response model for evaluation results"""
    style_and_topic: str
    strengths: list[str]
    weaknesses: list[str]
    improvement_suggestions: list[str]
    refined_sample: str
    timestamp: str
    user_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    timestamp: str


# ==================== API ENDPOINTS ====================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check system health and readiness"""
    if coach_system is None:
        raise HTTPException(status_code=503, detail="Writing Coach System not initialized")
    
    return HealthResponse(
        status="healthy",
        message="Writing Coach System is ready",
        timestamp=datetime.now().isoformat()
    )


@app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate_writing(sample: WritingSample):
    """
    Evaluate a writing sample
    
    Returns:
    - style_and_topic: Analysis of writing style, topic, tone, and genre
    - strengths: List of strong points in the writing
    - weaknesses: List of areas for improvement
    - improvement_suggestions: Actionable suggestions for better writing
    - refined_sample: Improved version of the original writing
    - timestamp: When the evaluation was performed
    """
    if coach_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        logger.info(f"Evaluating writing sample (user: {sample.user_id}, length: {len(sample.text)})")
        
        # Run evaluation
        result: EvaluationResult = coach_system.evaluate_writing(sample.text)
        
        logger.info(f"✓ Evaluation complete for user: {sample.user_id}")
        
        return EvaluationResponse(
            style_and_topic=result.style_and_topic,
            strengths=result.strengths,
            weaknesses=result.weaknesses,
            improvement_suggestions=result.improvement_suggestions,
            refined_sample=result.refined_sample,
            timestamp=result.timestamp,
            user_id=sample.user_id
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


@app.post("/evaluate-image", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate_image(
    image: UploadFile = File(..., description="Image file containing text to evaluate"),
    user_id: Optional[str] = Form(None, description="Optional user identifier"),
    title: Optional[str] = Form(None, description="Optional title of the writing")
):
    """
    Evaluate writing from an uploaded image using OCR
    
    Accepts image files (JPG, PNG, etc.) and extracts text using OCR technology,
    then provides the same comprehensive writing evaluation.
    
    Returns:
    - style_and_topic: Analysis of writing style, topic, tone, and genre (includes OCR note)
    - strengths: List of strong points in the writing
    - weaknesses: List of areas for improvement  
    - improvement_suggestions: Actionable suggestions for better writing
    - refined_sample: Improved version of the original writing
    - timestamp: When the evaluation was performed
    """
    if coach_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Validate file type
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    temp_file = None
    try:
        logger.info(f"Processing image upload (user: {user_id}, filename: {image.filename})")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Run OCR and evaluation
        result: EvaluationResult = coach_system.evaluate_writing_from_image(temp_file_path)
        
        logger.info(f"✓ Image evaluation complete for user: {user_id}")
        
        return EvaluationResponse(
            style_and_topic=result.style_and_topic,
            strengths=result.strengths,
            weaknesses=result.weaknesses,
            improvement_suggestions=result.improvement_suggestions,
            refined_sample=result.refined_sample,
            timestamp=result.timestamp,
            user_id=user_id
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Image evaluation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Image evaluation failed: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
async def batch_evaluate(samples: list[WritingSample], background_tasks: BackgroundTasks):
    """
    Batch evaluate multiple writing samples
    Returns immediately with task ID; results available via /results endpoint
    """
    if coach_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    task_id = f"batch_{datetime.now().timestamp()}"
    logger.info(f"Starting batch evaluation with task_id: {task_id}")
    
    async def evaluate_batch():
        results = []
        for idx, sample in enumerate(samples, 1):
            try:
                logger.info(f"Processing batch item {idx}/{len(samples)}")
                result = coach_system.evaluate_writing(sample.text)
                results.append({
                    "user_id": sample.user_id,
                    "status": "success",
                    "result": result.to_dict()
                })
            except Exception as e:
                logger.error(f"Batch item {idx} failed: {e}")
                results.append({
                    "user_id": sample.user_id,
                    "status": "error",
                    "error": str(e)
                })
        
        logger.info(f"✓ Batch evaluation complete: {task_id}")
        return results
    
    background_tasks.add_task(evaluate_batch)
    
    return {
        "task_id": task_id,
        "message": "Batch evaluation started",
        "samples_count": len(samples)
    }


@app.get("/", tags=["Info"])
async def root():
    """API information and available endpoints"""
    return {
        "name": "Writing Coach API",
        "description": "Automatic English Writing Evaluation powered by Autogen",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health - Check system status",
            "evaluate": "POST /evaluate - Evaluate a single writing sample",
            "evaluate_image": "POST /evaluate-image - Evaluate text from uploaded image (OCR)",
            "batch_evaluate": "POST /batch-evaluate - Evaluate multiple samples",
            "docs": "GET /docs - Interactive API documentation (Swagger UI)",
            "openapi": "GET /openapi.json - OpenAPI schema"
        }
    }


# ==================== ERROR HANDLERS ====================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    return HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return HTTPException(status_code=500, detail="Internal server error")


# ==================== FOR LOCAL TESTING ====================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Writing Coach API server...")
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level.lower(),
        reload=config.debug
    )
