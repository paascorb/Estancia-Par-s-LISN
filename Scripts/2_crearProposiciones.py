import json
import transformers
import torch
from huggingface_hub import login
import pickle

def main():
    f = open('Datasets/wikipedia.json')
    data = json.load(f)
    resultados = []

    model_id = "CohereLabs/aya-expanse-8b"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        device="cuda",
    )

    c=0
    maxim = 100
    for elem in data:
        print(f"Fase {c} de {maxim}", end='\r')
        outputs = pipeline(
            get_messages(elem["Texto"])
        )
        resultados.extend([x for x in outputs[0]["generated_text"][-1]["content"].splitlines() if x != ""])
        c += 1
        if c > maxim:
            break

    with open('Datasets/resultado_proposiciones.pkl', 'wb') as fp:
        pickle.dump(resultados, fp)

def get_messages(texto):
    messages = [
            {"role": "system", "content": "Eres un chatbot encargado de extraer proposiciones atómicas de un texto. Una proposición es atómica cuando es autocontenida: no necesita contexto u otra información para entenderse, por ejemplo: 'Morten Harket sufrió acoso escolar' es correcta mientras que 'Sufrió acoso escolar' no es autocontenida ya que no se sabe a quién hace referencia; es mínimal: no se puede descomponer en proposiciones más sencillas; y es coherente: tiene sentido, es decir, es una frase completa con sujeto y predicado, una palabra suelta o una sigla no es una proposición. Solo debes listar las proposiciones atómicas separadas por puntos sin dar ninguna información extra; por ejemplo: París es una ciudad. París es la capital de Francia. Por último, es importante que si no logras extraer proposiciones de ninguna forma pongas: 'No se pueden extaer proposciones.'."},
            {"role": "user", "content": f"Dado el siguiente texto: {texto}\n Lista las proposiciones atómicas."},
        ]
    return messages

if __name__ == "__main__":
    main()