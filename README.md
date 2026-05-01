#  HR AI Resume Screener

##  Description
Ce projet est une API intelligente de screening de CV utilisant un modèle de Machine Learning (LLM fine-tuné avec QLoRA).

Il permet d’analyser automatiquement un CV et une description de poste pour décider si le candidat est :
-  SELECT (admis)
-  REJECT (refusé)

---

##  Fonctionnalités

-  API REST avec FastAPI
-  Modèle IA (Transformers / PyTorch)
- Analyse de CV (texte + PDF)
-  OCR pour PDF scannés (Tesseract + Poppler)
-  Interface utilisateur avec Gradio
-  Tests de l’API
-  Retour de confiance du modèle

---

## Architecture

Utilisateur → Gradio Interface  
↓  
FastAPI (/predict)  
↓  
OCR + NLP Model  
↓  
Résultat (Select / Reject)

---
## pour lancer le projet
 Créer l'environnement virtuel:
cd hr_ai_api
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate  #linux
lancer:python app.py(Lancer l'API seule)
Accès : http://127.0.0.1:8000/docs
lancer:python gradio_app.py(Lancer l'interface Gradio)
Accès : http://127.0.0.1:7860
Lancer tout d'un coup (Windows):
start.bat




