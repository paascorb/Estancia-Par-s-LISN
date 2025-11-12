from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import ast
import numpy as np

def main():

    # Cargamos el modelo para comparar frases
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Cargamos los bancos de datos
    dataset = load_dataset("paascorb/ProposicionesValidacion_Gold-Standard")
    """ Evaluamos el primer modelo: Gemma3 """
    df_gemma3 = pd.read_csv("Datasets/Validacion/evaluacion_proposiciones_gemma3.csv")

    # Lista de los umbrales a evaluar
    umbrales = np.arange(0.05, 1, 0.05)

    # Evaluamos cada frase con las del gold-standard
    resultados = []
    cont = 0
    for umbral in umbrales:
        print(f"Fase {cont} de {len(umbrales)}", end='\r')
        for index, row in df_gemma3.iterrows():
            props = dataset["train"][index]["Proposiciones"]
            for proposicion in ast.literal_eval(row.Proposiciones):
                prop_emb = model.encode(proposicion, convert_to_tensor=True)
                embs = [model.encode(x, convert_to_tensor=True) for x in props]
                res = [util.pytorch_cos_sim(prop_emb, x)[0][0].item() for x in embs]
                valor = max(res) if max(res) >= umbral else 0
                resultados.append({"Modelo": "Gemma3_27b", "ID": index, "Frase": proposicion, "Valor": valor, "Umbral": umbral.item()})
        cont += 1
    df_val_gemma3 = pd.DataFrame(resultados)
    df_val_gemma3.to_csv("Datasets/Validacion/gemma3_27b_propVal.csv", index=False)

if __name__ == "__main__":
    main()