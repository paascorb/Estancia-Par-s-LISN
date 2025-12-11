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

from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import ast
import numpy as np
from numpy import dot
from numpy.linalg import norm

def main(url, modelo, destino, destino_rec):

    # Cargamos el modelo para comparar frases
    model = model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to("cuda")

    # Cargamos los bancos de datos
    dataset = load_dataset("paascorb/ProposicionesValidacion_Gold-Standard")
    df = pd.read_csv(url)

    # Lista de los umbrales a evaluar
    umbrales = np.arange(0.05, 1, 0.05)

    # Metodo para calcular la distancia multivector
    def calcular_distancia(a, b):
        return dot(a, b)/(norm(a)*norm(b))

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
                    [proposicion])[0]
                embs = model.encode(props) 
                res = [calcular_distancia(prop_emb, x) for x in embs]
                i_max = np.argmax(res)
                maxi = res[i_max]
                if props[i_max] in aux or maxi < umbral:
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
    main("Datasets/evaluacion_proposiciones_LFM2_1,2B.csv", "LFM2 1'2B", "Datasets/LFM2_1,2B_propVal.csv", "Datasets/LFM2_1,2B_propVal_recalls.csv")

from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import ast
import numpy as np
from numpy import dot
from numpy.linalg import norm

def main(url, modelo, destino, destino_rec):

    # Cargamos el modelo para comparar frases
    model = model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to("cuda")

    # Cargamos los bancos de datos
    dataset = load_dataset("paascorb/ProposicionesValidacion_Gold-Standard")
    df = pd.read_csv(url)

    # Lista de los umbrales a evaluar
    umbrales = np.arange(0.05, 1, 0.05)

    # Metodo para calcular la distancia multivector
    def calcular_distancia(a, b):
        return dot(a, b)/(norm(a)*norm(b))

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
                maxi = 0
                prop_select = ""
                prop_emb = model.encode(
                        [proposicion])[0]
                for prop in props:
                    emb = model.encode(prop) 
                    res = calcular_distancia(prop_emb, emb)
                    if res > maxi:
                        maxi = res
                        prop_select = prop
                if prop_select in aux or maxi < umbral:
                    maxi = 0
                else:
                    aux.append(prop_select)
                resultados.append({"Modelo": modelo, "ID": index, "Frase": proposicion, "Valor": maxi, "Umbral": umbral.item()})
            recalls.append({"Modelo": modelo, "Texto": row.Texto, "Umbral": umbral.item(), "Seleccionados": aux, "Faltan": list(set(props) - set(aux))})
        cont += 1
    resu_df = pd.DataFrame(resultados)
    recal_df = pd.DataFrame(recalls)
    resu_df.to_csv(destino, index=False)
    recal_df.to_csv(destino_rec, index=False)

if __name__ == "__main__":
    main("Datasets/evaluacion_proposiciones_LFM2_2,6B_V2.csv", "LFM2 2'6B", "Datasets/LFM2_2,6B_propVal.csv", "Datasets/LFM2_2,6B_propVal_recalls.csv")