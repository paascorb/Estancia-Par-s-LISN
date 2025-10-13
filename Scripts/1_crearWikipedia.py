import wikipediaapi
import requests
import json
from threading import Thread

def main():
    proletariado = 50
    Proletario.cond_parada = 100000
    prole = [Proletario() for p in range(1, proletariado)]
    [w.start() for w in prole]
    [w.join() for w in prole]

    with open('../Datasets/wikipedia.json', 'w') as fout:
        json.dump(Proletario.lista , fout, ensure_ascii=False, indent=2)

class Proletario(Thread):
    lista = []
    count = 0
    cond_parada = 0

    def run(self):
        while Proletario.count < Proletario.cond_parada:
            Proletario.count += 1
            results = self.extraer_pagina()
            Proletario.lista.extend(results)
            print(f"Fase {Proletario.count} de {Proletario.cond_parada}", end='\r')
    
    def extraer_pagina(self):
        result = []
        try:
            wiki = wikipediaapi.Wikipedia(user_agent='prevenIA (paascorb@unirioja.es)',
                                        language='es',
                                        extract_format=wikipediaapi.ExtractFormat.WIKI)
            titulo = titulo_aleatorio()
            p_wiki = wiki.page(titulo)
            for s in p_wiki.sections:
                result.append({'Título': titulo, 'Sección': s.title, 'Texto': s.text})
            return result
        except:
            return []

def titulo_aleatorio():
    headers = {'User-Agent': 'revenIA (paascorb@unirioja.es)'}
    while True:
        req = requests.get("https://es.wikipedia.org/api/rest_v1/page/random/summary", headers=headers)
        titulo = comprobar_clave(json.loads(req.content), "title")
        if titulo:
            return titulo

def comprobar_clave(dicti, clave):
    if clave in dicti:
        return dicti[clave]
    else:
        return False

if __name__ == "__main__":
    main()
