import os
import uvicorn
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np

# Inicializar la aplicación FastAPI
app = FastAPI(title="API para Predicción con Red Neuronal")

# --- Rutas a los archivos de modelo y preprocesamiento ---
model_path = "modelo_entrenado.h5"
poly_path = "poly.pkl"
scaler_path = "scaler.pkl"

# --- Cargar el modelo y objetos de preprocesamiento ---
model = tf.keras.models.load_model(model_path)
with open(poly_path, "rb") as f:
    poly = pickle.load(f)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# --- Definir el esquema para múltiples conjuntos de datos ---
class BatchInputData(BaseModel):
    datasets: List[List[float]]

# --- Endpoint para la predicción ---
@app.post("/predict_batch")
def predict_batch(data: BatchInputData):
    expected_features = 6
    error_messages = []

    # Validar cada conjunto de datos
    for idx, dataset in enumerate(data.datasets, start=1):
        if len(dataset) != expected_features:
            error_messages.append(f"La hilera {idx} tiene {len(dataset)} entradas, se esperaba {expected_features}.")

    if error_messages:
        return {"error": error_messages}

    # Convertir la entrada a un array NumPy
    input_data = np.array(data.datasets)
    input_poly = poly.transform(input_data)
    input_scaled = scaler.transform(input_poly)
    predictions = model.predict(input_scaled)

    # Devolver las predicciones
    return {
        "Predicciones": f"La predicción para el conjunto de datos entregados es {sum(predictions.flatten().tolist())} MwH."
    }

# --- Endpoint de prueba ---
@app.get("/")
def read_root():
    return {"mensaje": "La API está funcionando correctamente en Render."}

# --- Punto de entrada para iniciar la aplicación ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Usar la variable PORT proporcionada por Render
    uvicorn.run("main:app", host="0.0.0.0", port=port)

