import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import chatbot_reply

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Backend funcionando 🚀"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    mensaje = data.get("mensaje", "")
    inventario = data.get("inventario", [])
    ruta = data.get("rutaActual", "")

    result = chatbot_reply(mensaje, inventario, ruta)

    return jsonify({
        "respuesta": result["response"],
        "intencion": result["intent"],
        "confianza": result["confidence"]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)