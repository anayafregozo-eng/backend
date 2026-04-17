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
    data = request.get_json(silent=True)

    if not data or "mensaje" not in data:
        return jsonify({"respuesta": "No se recibió ningún mensaje."}), 400

    mensaje = str(data["mensaje"]).strip()
    inventario = data.get("inventario", [])
    ruta_actual = data.get("rutaActual", "")

    if not mensaje:
        return jsonify({"respuesta": "Escribe un mensaje para poder ayudarte."}), 400

    result = chatbot_reply(mensaje, inventario, ruta_actual)

    return jsonify({
        "respuesta": result["response"],
        "intencion": result["intent"],
        "confianza": result["confidence"]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)