# 🤖 HR AI Resume Screener

## 📌 Description
Ce projet est une API intelligente de screening de CV utilisant un modèle de Machine Learning (LLM fine-tuné avec QLoRA).

Il permet d’analyser automatiquement un CV et une description de poste pour décider si le candidat est :
- ✅ SELECT (admis)
- ❌ REJECT (refusé)

---

## 🚀 Fonctionnalités

- 🔥 API REST avec FastAPI
- 🤖 Modèle IA (Transformers / PyTorch)
- 📄 Analyse de CV (texte + PDF)
- 🧠 OCR pour PDF scannés (Tesseract + Poppler)
- 🎨 Interface utilisateur avec Gradio
- 🧪 Tests de l’API
- 📊 Retour de confiance du modèle

---

## 🏗️ Architecture

Utilisateur → Gradio Interface  
↓  
FastAPI (/predict)  
↓  
OCR + NLP Model  
↓  
Résultat (Select / Reject)

---

## ⚙️ Installation

### 1. Cloner le projet
```bash
git clone https://github.com/ton-username/hr-ai-resume-screener.git
cd hr-ai-resume-screener