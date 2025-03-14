"""
API for ThaiNLP library.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from .models.large_language_model import ThaiLargeLanguageModel
from .context_analysis import ContextAnalyzer
import time

app = FastAPI(
    title="ThaiNLP API",
    description="API for Thai Natural Language Processing",
    version="1.0.0"
)

# Initialize models
llm = ThaiLargeLanguageModel()
context_analyzer = ContextAnalyzer()

class TextInput(BaseModel):
    text: str
    context: Optional[str] = None

class AnalysisResponse(BaseModel):
    results: Dict[str, Any]
    processing_time: float

@app.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(input_data: TextInput):
    """
    Analyze text using ThaiNLP models.
    
    Args:
        input_data: Text input and context
        
    Returns:
        Analysis results
    """
    try:
        start_time = time.time()
        
        # Use LLM for general text analysis
        llm_results = llm.analyze_text(input_data.text)
        
        # Use context analyzer if context is provided
        context_results = {}
        if input_data.context:
            if input_data.context == "social_media":
                context_results = context_analyzer.analyze_social_media(input_data.text)
            elif input_data.context == "email":
                context_results = context_analyzer.analyze_email(input_data.text)
            elif input_data.context == "legal":
                context_results = context_analyzer.analyze_legal_document(input_data.text)
        
        # Combine results
        results = {
            "general_analysis": llm_results,
            "context_specific_analysis": context_results
        }
        
        processing_time = time.time() - start_time
        
        return AnalysisResponse(
            results=results,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/text")
async def generate_text(input_data: TextInput):
    """
    Generate text using ThaiNLP models.
    
    Args:
        input_data: Text input and context
        
    Returns:
        Generated text
    """
    try:
        generated_text = llm.generate_text(input_data.text)
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check API health status."""
    return {"status": "healthy"} 