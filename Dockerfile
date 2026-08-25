FROM python:3.11-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY api_prediction_avc.py .
COPY modele_xgb_avc.pkl .
COPY encodeur_sexe.pkl .
COPY encodeur_type_avc.pkl .
COPY encodeur_sor_mod.pkl .

EXPOSE 8000

CMD ["uvicorn", "api_prediction_avc:app", "--host", "0.0.0.0", "--port", "8000"]
