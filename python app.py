from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
import json

app = Flask(__name__)

# Lista de doenças e sintomas (para treinamento do modelo)
diseases = [
    "Gripe", "COVID-19", "Diabetes Tipo 2", "Hipertensão Arterial", "Asma",
    "Pneumonia", "Alergia Alimentar", "Doença de Alzheimer", "Depressão",
    "Anemia", "Câncer de Pulmão", "Doença de Crohn", "Fibromialgia",
    "Hipotireoidismo", "Esclerose Múltipla", "Síndrome do Intestino Irritável",
    "Artrite Reumatoide", "Doença Renal Crônica", "Enxaqueca", "Lúpus"
]

symptoms = [
    "Febre", "dor de garganta", "tosse", "dor no corpo", "tosse seca",
    "falta de ar", "perda de olfato", "dor de cabeça", "Sede excessiva",
    "fome frequente", "perda de peso", "visão embaçada", "fadiga",
    "Dor de cabeça", "tontura", "palpitações", "Urticária",
    "inchaço", "dor abdominal", "diarreia"
]

# Dados de treinamento para o modelo (simplificado)
training_data = np.array([
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Gripe
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # COVID-19
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Diabetes Tipo 2
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],  # Hipertensão Arterial
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # Asma
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],  # Pneumonia
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Alergia Alimentar
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Doença de Alzheimer
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Depressão
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0],  # Anemia
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # Câncer de Pulmão
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0],  # Doença de Crohn
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # Fibromialgia
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],  # Hipotireoidismo
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],  # Esclerose Múltipla
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0],  # Síndrome do Intestino Irritável
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],  # Artrite Reumatoide
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],  # Doença Renal Crônica
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],  # Enxaqueca
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]   # Lúpus
])

# Inicializando o modelo
def build_model():
    model = Sequential([
        tf.keras.layers.Input(shape=(len(symptoms),)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(diseases), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.fit(training_data, np.array([np.zeros(len(diseases)) for _ in range(len(training_data))]), epochs=100, batch_size=4, verbose=0)

def symptoms_to_vector(symptom_list):
    return [1 if symptom in symptom_list else 0 for symptom in symptoms]

def predict_disease(symptom_list):
    symptom_vector = symptoms_to_vector(symptom_list)
    prediction = model.predict(np.array([symptom_vector]))
    disease_index = np.argmax(prediction)
    confidence = prediction[0][disease_index] * 100
    return diseases[disease_index], confidence

@app.route("/diagnose", methods=["POST"])
def diagnose():
    data = request.get_json()

    name = data.get("name")
    age = data.get("age")
    gender = data.get("gender")
    contact = data.get("contact")
    symptoms_input = data.get("symptoms")
    observations = data.get("observations")

    # Prevendo a doença
    disease, confidence = predict_disease(symptoms_input)

    # Retornando o resultado como JSON
    return jsonify({
        "predicted_disease": disease,
        "confidence": confidence
    })

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", symptoms=symptoms)

if __name__ == "__main__":
    app.run(debug=True)
