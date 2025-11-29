import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        logger.critical("GOOGLE_API_KEY não foi encontrada no ambiente.")
        raise ValueError("GOOGLE_API_KEY não definida")
        
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Modelo GenAI configurado com sucesso.")
except Exception as e:
    logger.error(f"Erro ao configurar a API do Gemini: {e}")
    model = None

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health_check():
    app.logger.info("Health check solicitado.")
    return jsonify({"status": "healthy"}), 200

@app.route("/ask", methods=["POST"])
def ask_llm():
    
    if model is None:
        app.logger.error("Requisição recebida, mas o modelo LLM não foi inicializado.")
        return jsonify({"error": "Modelo LLM não foi inicializado."}), 500

    data = request.json
    if not data or 'question' not in data:
        app.logger.warning("Requisição inválida. 'question' não encontrada no JSON.")
        return jsonify({"error": "Requisição inválida. 'question' não encontrada no JSON."}), 400

    prompt = data['question']
    app.logger.info(f"Nova pergunta recebida: '{prompt[:50]}...'")

    try:
        response = model.generate_content(prompt)
        app.logger.info("Resposta do LLM gerada com sucesso.")
        return jsonify({"answer": response.text})
    
    except Exception as e:
        app.logger.error(f"Erro ao processar a requisição no modelo: {str(e)}")
        return jsonify({"error": f"Erro ao processar a requisição: {str(e)}"}), 503

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)