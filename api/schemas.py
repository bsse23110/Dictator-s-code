"""
api/schemas.py
--------------
Pydantic models for request validation and response serialization.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class PatientInput(BaseModel):
    """Raw patient data submitted via the form."""
    gender: int = Field(..., ge=0, le=1, description="0 = Male, 1 = Female")
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    scholarship: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes (Bolsa Familia)")
    hypertension: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    diabetes: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    alcoholism: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    handicap: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    sms_received: int = Field(..., ge=0, le=1, description="0 = No SMS, 1 = Received SMS reminder")
    scheduled_date: date = Field(..., description="Date the appointment was booked")
    appointment_date: date = Field(..., description="Date of the actual appointment")
    neighbourhood: str = Field(..., min_length=1, description="Neighbourhood name")
    previous_appointments: int = Field(0, ge=0, description="Number of prior appointments (0 if new patient)")
    previous_no_shows: int = Field(0, ge=0, description="Number of prior no-shows (0 if new patient)")


class PredictionResult(BaseModel):
    """Prediction output returned to the frontend."""
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of no-show (0 to 1)")
    prediction: int = Field(..., ge=0, le=1, description="0 = Will Show, 1 = Likely No-Show")
    risk_level: str = Field(..., description="Low Risk / Medium Risk / High Risk")
    explanation: str = Field(..., description="Human-readable explanation")
    features_used: dict = Field(default_factory=dict, description="Engineered feature values")


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    features: int
