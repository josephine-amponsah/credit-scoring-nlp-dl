# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import torch, joblib, json, numpy as np, pandas as pd
import shap, os
from fastapi.responses import HTMLResponse
from pydantic.dataclasses import dataclass

app = FastAPI()

@dataclass
class Selection(BaseModel):
    X : list[float]
    Start: Optional[datetime] = None
    End: Optional[datetime] = None
    
class GraphOutput(BaseModel):
    Start: Optional[datetime] = None
    End: Optional[datetime] = None
    Metric : list[float]
