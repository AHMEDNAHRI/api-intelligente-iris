# API Intelligente - Iris

API FastAPI qui predit l'espece d'une fleur Iris.

## Installation
python -m venv venv
venv\Scripts\activate
python -m pip install fastapi uvicorn scikit-learn joblib pytest httpx slowapi

## Entrainer le modele
python train_model.py

## Lancer l'API
python -m uvicorn main:app --reload
Documentation : http://127.0.0.1:8000/docs

## Cles API (TP seulement)
- user-key-123  (role USER)
- admin-key-456 (role ADMIN)
Header : X-API-Key

## Tests
python -m pytest -v
(attendre 1 minute entre deux lancements, a cause de la limite 5/minute)
