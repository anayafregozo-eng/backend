import json

intents = {"intents": []}

while True:
    tag = input("\nNombre de la intención (o 'salir'): ")
    if tag.lower() == "salir":
        break

    patterns = []
    print("Escribe ejemplos (patterns). Escribe 'listo' para terminar:")
    while True:
        p = input(">> ")
        if p.lower() == "listo":
            break
        patterns.append(p)

    responses = []
    print("Escribe respuestas. Escribe 'listo' para terminar:")
    while True:
        r = input(">> ")
        if r.lower() == "listo":
            break
        responses.append(r)

    intents["intents"].append({
        "tag": tag,
        "patterns": patterns,
        "responses": responses
    })

# guardar archivo
with open("intents.json", "w", encoding="utf-8") as file:
    json.dump(intents, file, indent=4, ensure_ascii=False)

print("\nDataset generado automáticamente")