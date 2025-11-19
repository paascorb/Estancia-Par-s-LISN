from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import ast
import numpy as np
from pylate import indexes, models, retrieve

def main(url, modelo, destino, destino_rec):

    # Cargamos el modelo para comparar frases
    model = models.ColBERT(model_name_or_path="LiquidAI/LFM2-ColBERT-350M").to("cuda")
    model.tokenizer.pad_token = model.tokenizer.eos_token

    # Cargamos los bancos de datos
    dataset = load_dataset("paascorb/ProposicionesValidacion_Gold-Standard")
    df = pd.read_csv(url)

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
    recalls = []
    cont = 0
    for umbral in umbrales:
        print(f"Fase {cont} de {len(umbrales)}", end='\r')
        for index, row in df.iterrows():
            props = dataset["train"][index]["Proposiciones"]
            aux = []
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
                i_max = np.argmax(res)
                maxi = res[i_max]
                if maxi < umbral and props[i_max] in aux:
                    maxi = 0
                else:
                    aux.append(props[i_max])
                resultados.append({"Modelo": modelo, "ID": index, "Frase": proposicion, "Valor": maxi, "Umbral": umbral.item()})
            recalls.append({"Modelo": modelo, "Texto": row.Texto, "Umbral": umbral.item(), "Seleccionados": aux, "Faltan": list(set(props) - set(aux))})
        cont += 1
    resu_df = pd.DataFrame(resultados)
    recal_df = pd.DataFrame(recalls)
    resu_df.to_csv(destino, index=False)
    recal_df.to_csv(destino_rec, index=False)

if __name__ == "__main__":
    main("Datasets/evaluacion_proposiciones_gemma3.csv", "Gemma3 27B", "Datasets/gemma3_27b_propVal_ColBERT.csv", "Datasets/gemma3_27b_propVal_ColBERT_recalls.csv")