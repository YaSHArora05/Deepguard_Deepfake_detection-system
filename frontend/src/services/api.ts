const API_URL = "http://127.0.0.1:8000";

export interface PredictionResult {
  filename: string;
  file_type: string;
  prediction: string;
  confidence?: number;
  fake_probability?: number;
  real_probability?: number;
  message?: string;
}

export async function predictFile(
  file: File
): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      data.detail || data.message || "Prediction failed"
    );
  }

  return data;
}