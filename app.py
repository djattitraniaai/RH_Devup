from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

app = FastAPI(title="HR AI Resume Screener", description="API pour le screening de CV avec LLM fine-tuné")

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_MODEL = "abd-bk-2/RH_Devup_adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Démarrage sur {DEVICE}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
    model = PeftModel.from_pretrained(base, ADAPTER_MODEL)
    model.eval()
    print("Modèle LoRA chargé avec succès !")
except Exception as e:
    raise RuntimeError(f"Erreur chargement modèle: {e}")

class ResumeRequest(BaseModel):
    resume_text: str
    job_description: str

def build_prompt(resume_text, job_description):
    system = "You are an HR AI assistant. Return exactly two lines:\nDecision: select|reject\nReason: <short reason>"
    user = f"Job Description:\n{job_description}\n\nResume:\n{resume_text}"
    return f"<s>[INST]\n{system}\n[/INST]\n\n{user}\n\n[/INST]"

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            bad_words_ids=tokenizer(["[/INST]", "<s>"], add_special_tokens=False).input_ids,
        )
    gen = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)

    lines = []
    for line in text.splitlines():
        if line.startswith("Decision:") or line.startswith("Reason:"):
            lines.append(line)
        if len(lines) == 2:
            break
    return "\n".join(lines)

@app.post("/predict")
async def predict(request: ResumeRequest):
    try:
        prompt = build_prompt(request.resume_text, request.job_description)
        result_text = generate_answer(prompt)

        decision = ""
        reason = ""
        for line in result_text.splitlines():
            if line.lower().startswith("decision:"):
                decision = line.split(":",1)[1].strip().lower()
            if line.lower().startswith("reason:"):
                reason = line.split(":",1)[1].strip()

        return {"decision": decision, "reason": reason, "raw": result_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")