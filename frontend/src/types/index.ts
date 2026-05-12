export interface PatientFormData {
  gender: number;
  age: number;
  scholarship: number;
  hypertension: number;
  diabetes: number;
  alcoholism: number;
  handicap: number;
  sms_received: number;
  scheduled_date: string;
  appointment_date: string;
  neighbourhood: string;
  previous_appointments: number;
  previous_no_shows: number;
}

export interface PredictionResponse {
  probability: number;
  prediction: number;
  risk_level: string;
  explanation: string;
  features_used: Record<string, number>;
}

export interface HealthResponse {
  status: string;
  model: string;
  features: number;
}
