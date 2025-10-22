import json
import ollama
import time
import pandas as pd

f = open('Datasets/wikipedia.json')
data = json.load(f)

def obtener_respuesta(Texto):
    prompt=f"""Eres un chatbot encargado de extraer proposiciones atómicas de un texto. 
                Una proposición es atómica cuando es autocontenida: no necesita contexto u otra información para entenderse, por ejemplo: 'Morten Harket sufrió acoso escolar' es correcta mientras que 'Sufrió acoso escolar' no es autocontenida ya que no se sabe a quién hace referencia; es mínimal: no se puede descomponer en proposiciones más sencillas; y es coherente: tiene sentido, es decir, es una frase completa con sujeto y predicado, una palabra suelta o una sigla no es una proposición. 
                Solo debes listar las proposiciones atómicas separadas por puntos y un salto de línea ('\\n') sin dar ninguna información extra; por ejemplo: París es una ciudad. París es la capital de Francia.
                Dado el siguiente texto de ejemplo: 'Morten Harket nació en Kongsberg siendo el segundo de 5 hijos. Su padre, Reidar, fue médico en un hospital y su madre Henny fue maestra de economía del hogar en una escuela. Morten fue un chico soñador y sufrió acoso escolar en la escuela.'
                Serían proposiciones atómicas correctas: Morten Harket nació en Kongsberg. Morten Harket es el segundo de 5 hijos. El padre de Morten Harket se llama Reidar. Reidar Harket fue médico de un hospital.
                Serían proposciones átomicas incorrectas: Él nació en Kongsberg. (No autocontenida); Su madre fue maestra de economía del hogar. (No autocontenida); Morten Harket fue un chico soñador y sufrió acoso escolar. (No minimal, se puede separar en: Morten Harket fue un chico soñador. Morten Harket sufrió acoso escolar.); Morten Harket fue astronauta. (Incorrecta, ya que esa infromación no está en el texto);
                Por último, es importante que si no logras extraer proposiciones de ninguna forma pongas: 'No se pueden extaer proposiciones.'."\n
                Dado el siguiente texto: {Texto}\n
                Lista las proposiciones atómicas:
           """
    return prompt

resultado = []
cont = 0
inicio = 1
stop = 2500
for elem in data:
    cont += 1
    if cont > stop:
        break
    if cont <= inicio:
        continue
    print(f"Fase {cont} de {stop}", end='\r')
    start = time.time()
    instr = obtener_respuesta(elem["Texto"])
    resp = ollama.generate(model='gemmaCustomP', prompt=instr)["response"]
    end = time.time()
    resultado.append({'Id': cont-1, 'Título': elem["Título"], 'Sección': elem["Sección"], 'Texto': elem["Texto"], 'Instruccion': instr, 'Proposiciones': [x for x in resp.split("\n") if x], 'Tiempo': end - start})

with open('Datasets/Proposiciones_1.json', 'w', encoding='utf-8') as f:
    json.dump(resultado, f, ensure_ascii=False, indent=4)