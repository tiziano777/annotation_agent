from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class State(BaseModel):
    title: Optional[str] = Field(default=None, description="Titolo in input")
    url: Optional[str] = Field(default=None, description="url di input")
    language: Optional[str] = Field(default=None, description="Lingua di input")
    text: Optional[str] = Field(default=None, description="Testo di input")
    
    segmented_text: Optional[List[str]] = Field(default=None, description="Testo di input segmentato")
    
    clickbait: Optional[int] = Field(default=None, description="IS Titolo clickbait? 0/1")
    segmented_signals: Optional[List[Dict[str, List[str]]]] = Field(default=None, description="disinformation signals per ogni elemento di testo")
    
    span_signals: Optional[List[Dict[str, Any]]] = Field(default=None, description="Span disiniformation signals con posizioni start/end")
    
    error_status: Optional[str] = Field(default=None, description="Stato dell'errore")
    
    input_tokens: Optional[int] = Field(default=None, description="Numero di token in input")
    output_tokens: Optional[int] = Field(default=None, description="Numero di token in output")

    refined_once: bool = Field(default=False, description="Flag per indicare se il raffinamento è in corso o completato")