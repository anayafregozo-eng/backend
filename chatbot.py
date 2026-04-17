import random
import json
import pickle
import numpy as np
import nltk
import re
import unicodedata
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

with open("intents.json", encoding="utf-8") as file:
    intents = json.load(file)

with open("words.pkl", "rb") as file:
    words = pickle.load(file)

with open("classes.pkl", "rb") as file:
    classes = pickle.load(file)

model = load_model("chatbot_model.h5")

ERROR_THRESHOLD = 0.65

context = {
    "last_intent": None,
    "last_product": None,
    "last_quantity": None,
    "last_module": None
}

KNOWN_MODULES = {
    "productos": ["productos", "inventario", "almacen", "almacén"],
    "ventas": ["ventas", "venta"],
    "usuarios": ["usuarios", "usuario"],
    "proveedores": ["proveedores", "proveedor"],
    "finanzas": ["finanzas", "financiero"],
    "alertas": ["alertas", "notificaciones", "avisos"],
    "dashboard": ["dashboard", "inicio", "panel"]
}

HELP_KEYWORDS = [
    "ayuda", "ayudar", "puedes hacer", "que puedes hacer",
    "qué puedes hacer", "como funciona", "cómo funciona",
    "en que me puedes ayudar", "en qué me puedes ayudar"
]

FALLBACK_PRODUCTS = [
    "laptop", "laptops", "mouse", "mouses", "teclado", "teclados",
    "monitor", "monitores", "impresora", "impresoras", "cable", "cables"
]

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_up_sentence(sentence: str):
    sentence = normalize_text(sentence)
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence: str):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

def predict_intents(sentence: str):
    bow = bag_of_words(sentence)
    predictions = model.predict(np.array([bow]), verbose=0)[0]

    results = []
    for i, prob in enumerate(predictions):
        results.append({
            "intent": classes[i],
            "probability": float(prob)
        })

    results.sort(key=lambda x: x["probability"], reverse=True)
    return results

def find_intent_data(tag: str):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return intent
    return None

def choose_response(tag: str):
    intent_data = find_intent_data(tag)
    if intent_data and intent_data.get("responses"):
        return random.choice(intent_data["responses"])
    return "No entendí muy bien tu mensaje. ¿Puedes escribirlo de otra forma?"

def extract_quantity(text: str):
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))
    return None

def extract_module(text: str):
    for module, aliases in KNOWN_MODULES.items():
        for alias in aliases:
            if alias in text:
                return module
    return None

def looks_like_help(text: str):
    return any(keyword in text for keyword in HELP_KEYWORDS)

def normalizar_nombre(texto: str) -> str:
    return normalize_text(texto)

def extract_product_from_inventory(text: str, inventario):
    normalized_text = normalize_text(text)

    if inventario:
        mejor_match = None
        mejor_score = 0

        for item in inventario:
            nombre = item.get("nombre", "")
            nombre_norm = normalize_text(nombre)

            palabras = nombre_norm.split()

            score = 0
            for p in palabras:
                if p in normalized_text:
                    score += 1

            if score > mejor_score:
                mejor_score = score
                mejor_match = nombre

        return mejor_match

    return None

def buscar_producto_en_inventario(nombre_producto, inventario):
    if not nombre_producto:
        return None

    objetivo = normalizar_nombre(nombre_producto)

    for item in inventario:
        nombre = normalizar_nombre(item.get("nombre", ""))
        if objetivo in nombre or nombre in objetivo:
            return item

    return None

def productos_stock_bajo(inventario, limite=10):
    return [p for p in inventario if int(p.get("stock", 0)) <= limite]

def productos_visibles(inventario):
    visibles = [p for p in inventario if p.get("visible", True)]
    return visibles if visibles else inventario

def analyze_message(message: str, inventario=None):
    if inventario is None:
        inventario = []

    normalized = normalize_text(message)
    predictions = predict_intents(message)

    best_intent = predictions[0]["intent"]
    best_prob = predictions[0]["probability"]

    product = extract_product_from_inventory(normalized, inventario)
    quantity = extract_quantity(normalized)
    module = extract_module(normalized)

    if looks_like_help(normalized):
        best_intent = "ayuda"

    # Regla extra para preguntas de stock / existencia
    if "stock" in normalized or "hay" in normalized or "cuantas" in normalized or "cuántas" in message.lower():
        if product:
            best_intent = "consultar_inventario"

    if best_prob < ERROR_THRESHOLD and best_intent != "ayuda" and best_intent != "consultar_inventario":
        best_intent = "desconocido"

    analysis = {
        "message": message,
        "normalized": normalized,
        "intent": best_intent,
        "confidence": best_prob,
        "product": product,
        "quantity": quantity,
        "module": module,
        "top_predictions": predictions[:3]
    }

    return analysis

def build_dynamic_response(analysis, inventario, ruta_actual):
    intent = analysis["intent"]
    product = analysis["product"]
    quantity = analysis["quantity"]
    module = analysis["module"]

    inventario_actual = productos_visibles(inventario)

    if intent == "saludo":
        return random.choice([
            "Hola, ¿en qué puedo ayudarte?",
            "Hola, puedo ayudarte con productos, stock y alertas.",
            "Hola, dime qué necesitas revisar."
        ])

    if intent == "ayuda":
        if inventario_actual:
            return (
                "Puedo ayudarte a consultar productos, revisar stock, detectar artículos bajos, "
                "buscar un producto específico y orientarte dentro del sistema."
            )
        return (
            "Puedo ayudarte a consultar inventario, buscar productos, actualizar stock "
            "y orientarte dentro del sistema."
        )
    

    if intent == "consultar_inventario":
        if product:
            producto_encontrado = buscar_producto_en_inventario(product, inventario_actual)

            if producto_encontrado:
                return f"{producto_encontrado['nombre']} tiene {producto_encontrado['stock']} unidades disponibles."

            return f"No encontré {product} en el inventario."

        if inventario_actual:
            nombres = [p["nombre"] for p in inventario_actual[:4]]
            return f"Tienes {len(inventario_actual)} productos en inventario como: {', '.join(nombres)}."

        return "No hay productos en el inventario."

    if intent == "buscar_producto":
        producto_encontrado = buscar_producto_en_inventario(product, inventario_actual)
        if producto_encontrado:
            return (
                f"Sí, encontré {producto_encontrado['nombre']}. "
                f"Tiene {producto_encontrado['stock']} unidades y cuesta ${float(producto_encontrado['precio']):.2f}."
            )
        if product:
            return f"No encontré {product} en el inventario visible."
        return "Dime qué producto quieres buscar."

    if intent == "info_producto":
        producto_encontrado = buscar_producto_en_inventario(product, inventario_actual)
        if producto_encontrado:
            return (
                f"{producto_encontrado['nombre']} pertenece al tipo {producto_encontrado['tipo']}, "
                f"lo surte {producto_encontrado['proveedor']}, tiene {producto_encontrado['stock']} unidades "
                f"y su precio es de ${float(producto_encontrado['precio']):.2f}."
            )
        if product:
            return f"No encontré información de {product} en la tabla actual."
        return "Dime qué producto quieres consultar."

    if intent == "alertas_stock":
        bajos = productos_stock_bajo(inventario_actual)
        if bajos:
            lista = ", ".join([f"{p['nombre']} ({p['stock']})" for p in bajos[:5]])
            return f"Los productos con stock bajo son: {lista}."
        return "No veo productos con stock bajo en este momento."

    if intent == "agregar_producto":
        if product and quantity:
            return f"Si quieres agregar {quantity} unidades de {product}, puedes hacerlo desde el módulo de productos."
        return "Puedes agregar un nuevo producto desde la sección de productos."

    if intent == "actualizar_stock":
        producto_encontrado = buscar_producto_en_inventario(product, inventario_actual)
        if producto_encontrado and quantity is not None:
            return (
                f"Actualmente {producto_encontrado['nombre']} tiene {producto_encontrado['stock']} unidades. "
                f"Si quieres, puedes actualizarlo a {quantity} desde el módulo de productos."
            )
        if product:
            return f"Puedes modificar el stock de {product} desde la sección de productos."
        if quantity is not None and context.get("last_product"):
            return f"Puedes actualizar {context['last_product']} a {quantity} unidades desde productos."
        return "Puedes actualizar el stock desde el módulo de productos."

    if intent == "eliminar_producto":
        if product:
            return f"Si quieres eliminar {product}, puedes hacerlo desde el módulo de productos seleccionando ese artículo."
        return "Puedes eliminar productos desde la sección de productos."

    if intent == "navegacion_modulos":
        if module:
            return f"Claro, puedes ir al módulo de {module} para continuar."
        return "Dime a qué módulo quieres entrar."

    if intent == "funciones_sistema":
        return "El sistema permite gestionar productos, ventas, usuarios, proveedores, finanzas y alertas."

    if intent == "despedida":
        return random.choice([
            "Hasta luego.",
            "Nos vemos.",
            "Aquí estaré si me necesitas."
        ])

    if intent == "agradecimiento":
        return random.choice([
            "De nada.",
            "Con gusto.",
            "Para eso estoy."
        ])

    return random.choice([
        "No entendí muy bien lo que necesitas. ¿Puedes escribirlo de otra forma?",
        "No me quedó del todo claro. ¿Podrías decirlo de otra manera?",
        "No estoy segura de lo que quieres consultar. Intenta escribirlo diferente."
    ])

def update_context(analysis):
    context["last_intent"] = analysis["intent"]

    if analysis["product"]:
        context["last_product"] = analysis["product"]

    if analysis["quantity"]:
        context["last_quantity"] = analysis["quantity"]

    if analysis["module"]:
        context["last_module"] = analysis["module"]

def chatbot_reply(message: str, inventario=None, ruta_actual=""):
    if inventario is None:
        inventario = []

    analysis = analyze_message(message, inventario)
    response = build_dynamic_response(analysis, inventario, ruta_actual)
    update_context(analysis)

    return {
        "response": response,
        "intent": analysis["intent"],
        "confidence": analysis["confidence"],
        "context": context.copy()
    }

if __name__ == "__main__":
    print("Chatbot listo (escribe 'salir' para terminar)\n")

    while True:
        message = input("Tú: ").strip()

        if not message:
            print("Bot: Escribe algo para poder ayudarte.\n")
            continue

        if normalize_text(message) in ["salir", "adios", "adiós", "bye"]:
            print("Bot: Hasta luego.\n")
            break

        result = chatbot_reply(message)

        print(f"Bot: {result['response']}")
        print(f"(intención: {result['intent']} | confianza: {result['confidence']:.2f})")
        print(f"(contexto: {result['context']})\n")