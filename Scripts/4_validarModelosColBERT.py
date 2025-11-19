from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import ast
import numpy as np
from pylate import indexes, models, retrieve

def main():

    # Cargamos el modelo para comparar frases
    model = models.ColBERT(model_name_or_path="LiquidAI/LFM2-ColBERT-350M").to("cuda")
    model.tokenizer.pad_token = model.tokenizer.eos_token

    # Cargamos los bancos de datos
    dataset = load_dataset("paascorb/ProposicionesValidacion_Gold-Standard")
    """ Evaluamos el primer modelo: Gemma3 """
    df_gemma3 = pd.read_csv("Datasets/evaluacion_proposiciones_gemma3.csv")

    # Lista de los umbrales a evaluar
    umbrales = np.arange(0.05, 1, 0.05)

    # Metodo para calcular la distancia multivector
    def calcular_distancia(a, b):
        res = []
        for elem in a:
            aux = []
            for comp in b:
                aux.append(np.dot(elem, comp))
            res.append(max(aux))
        return sum(res)

    # Evaluamos cada frase con las del gold-standard
    resultados = []
    cont = 0
    for umbral in umbrales:
        print(f"Fase {cont} de {len(umbrales)}", end='\r')
        for index, row in df_gemma3.iterrows():
            props = dataset["train"][index]["Proposiciones"]
            for proposicion in ast.literal_eval(row.Proposiciones):
                prop_emb = model.encode(
                    [proposicion],
                    batch_size=32,
                    is_query=False
                )[0]
                embs = model.encode(props,
                                        batch_size=32,
                                        is_query=False
                                    ) 
                res = [calcular_distancia(prop_emb, x) for x in embs]
                valor = max(res) if max(res) >= umbral else 0
                resultados.append({"Modelo": "Gemma3_27b", "ID": index, "Frase": proposicion, "Valor": valor, "Umbral": umbral.item()})
        cont += 1
    df_val_gemma3 = pd.DataFrame(resultados)
    df_val_gemma3.to_csv("Datasets/gemma3_27b_propVal_ColBERT.csv", index=False)

if __name__ == "__main__":
    main()