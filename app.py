from fastapi import FastAPI, HTTPException
#permet de vérifier les données envoyées à l’API éviter les erreurs (ex: CV vide, mauvais format) 
from pydantic import BaseModel
#tokenizer = transforme texte → nombres
#model = IA qui décide select/reject
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# 1. Initialisation(crée ton serveur API)(c’est le cœur de ton projet)
app = FastAPI(title="HR AI Resume Screener", description="API pour le screening de CV avec LLM fine-tuné")

# 2. Configuration(dossier où est ton modèle)(pour charger ton IA)
MODEL_PATH = "./model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Chargement du modèle (factice pour l'instant)
print(f"Démarrage sur {DEVICE}...")

# Modèle factice (en attendant le vrai modèle fine-tuné)
class DummyModel:
    def eval(self): 
        pass
    def to(self, device): 
        pass

try:
    # Essaye de charger un petit modèle de test
    #transforme texte en tokens
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    #modèle qui décide (select/reject)   
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.to(DEVICE)
    #mode test
    model.eval()
    print("Modèle de test chargé avec succès !")
except Exception as e:
    print(f"Utilisation du modèle factice : {e}")
    tokenizer = None
    model = DummyModel()

# 4. Structure d'entrée(garantir format correct)
class ResumeRequest(BaseModel):
    resume_text: str
    job_description: str

# 5. Endpoint de prédiction(recevoir CV + job)
@app.post("/predict")
#fonction principale
async def predict(request: ResumeRequest):
    """
    Reçoit un CV et une description de poste, retourne select/reject.
    """
    try:
        #vérifie si modèle est fake
        if isinstance(model, DummyModel) or tokenizer is None:
            # Mode démo : retourne une réponse factice
            return {
                "decision": "select", 
                "confidence": [0.3, 0.7],
                "message": "Mode démo - En attente du modèle fine-tuné"
            }
        
        # Combinaison des textes
        input_text = f"Resume: {request.resume_text}\nJob Description: {request.job_description}"
        
        # Tokenisation
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Prédiction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
        
        result = "select" if prediction == 1 else "reject"
        
        return {
            "decision": result,
            "confidence": torch.softmax(logits, dim=-1).tolist()[0]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")

# 6. Endpoint de santé(vérifie si API marche)
@app.get("/health")
def health():
    return {
        "status": "OK",
        "model_loaded": not isinstance(model, DummyModel),
        "device": DEVICE
    }

# 7. Endpoint racine(page d’accueil API)
@app.get("/")
def root():
    return {
        "message": "API HR Resume Screener",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    #démarre serveur
    uvicorn.run(app, host="127.0.0.1", port=8000)