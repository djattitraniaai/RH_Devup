import gradio as gr
import requests
import json
import PyPDF2
import pytesseract
API_URL = "http://127.0.0.1:8000/predict"
#config chemins (adapter selon ton PC)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Users\dell\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"
#OCR pour PDF image
def read_pdf_ocr(file_path):
    images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text
def read_pdf(file):
    text = ""

    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except:
        pass

    #si vide → OCR
    if not text.strip():
        print("PDF scanné détecté → utilisation OCR...")
        text = read_pdf_ocr(file)

    return text
def analyze_resume(pdf_file,resume_text, job_description):
    """Fonction appelée par Gradio"""
    
    if not job_description:
        return "Ajoute une description de poste"

    #priorité au PDF
    if pdf_file is not None:
        resume_text = read_pdf(pdf_file)

    #si rien du tout
    if not resume_text:
        return "❌ Ajoute un CV (texte ou PDF)"
    payload = {
        "resume_text": resume_text,
        "job_description": job_description
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        decision = result["decision"]
        
        if "confidence" in result:
            conf = result["confidence"][1] if decision == "select" else result["confidence"][0]
            conf_text = f" (confiance: {conf:.2f})"
        else:
            conf_text = ""
        
        if "message" in result:
            return f"{result['message']}\n\n### Décision : {decision.upper()}{conf_text}"
        
        # Emoji selon la décision
        emoji = "✅" if decision == "select" else "❌"
        return f"{emoji} **Décision : {decision.upper()}**{conf_text}"
    
    except Exception as e:
        return f"❌ Erreur : {str(e)}"

# Interface Gradio
with gr.Blocks(title="Screener de CV - IA RH") as demo:
    gr.Markdown("""
    # 🤖 Système de Screening de CV par IA
    
    Cette démo utilise un **LLM fine-tuné avec QLoRA** pour analyser automatiquement les CV.
    
    ### Comment ça marche ?
    1. Collez le contenu d'un CV
    2. Collez la description du poste
    3. L'IA prédit si le candidat est **sélectionné (select)** ou **rejeté (reject)**
    """)
    
    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="📄 Upload CV (PDF)", file_types=[".pdf"])
            resume_input = gr.Textbox(
                label="📄 CV (texte complet)",
                lines=15,
                placeholder="Collez le CV ici...\n\nExemple:\nDéveloppeur Python avec 5 ans d'expérience...",
                elem_id="resume-box"
            )
            job_input = gr.Textbox(
                label="💼 Description du poste",
                lines=8,
                placeholder="Collez la description du poste ici...",
                elem_id="job-box"
            )
            submit_btn = gr.Button("🔍 Analyser", variant="primary")
        
        with gr.Column():
            output = gr.Markdown(label="📊 Résultat", value="En attente d'analyse...")
    
    # Exemples
    gr.Markdown("### 📝 Exemples à tester")
    
    examples = [
        [
            "Développeur Full Stack avec 7 ans d'expérience, expert React et Node.js, a dirigé une équipe de 5 personnes",
            "Nous recherchons un Lead Developer Full Stack avec expérience en management d'équipe"
        ],
        [
            "Junior développeur, 1 an de stage, connaissances de base en HTML/CSS",
            "Senior Software Engineer avec 5+ ans d'expérience en architecture cloud"
        ]
    ]
    
    gr.Examples(
        examples=examples,
        inputs=[resume_input, job_input],
        label="Cliquez sur un exemple pour le tester"
    )
    
    submit_btn.click(fn=analyze_resume, inputs=[pdf_input,resume_input, job_input], outputs=output)

if __name__ == "__main__":
    demo.launch(share=False)  