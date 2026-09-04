# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch, joblib, json, numpy as np, pandas as pd
import shap, os
from fastapi.responses import HTMLResponse
import torch.nn as nn

app = FastAPI()

