import wikipediaapi

def main():
    for i in range(1,2):
        wiki = wikipediaapi.Wikipedia(user_agent='prevenIA (paascorb@unirioja.es)',
                                    language='es',
                                    extract_format=wikipediaapi.ExtractFormat.WIKI)
        p_wiki = wiki.page("Balaenidae")
        titulo = p_wiki.title
        lista = []
        # cat = wiki.page("Categoría:Física")
        for s in p_wiki.sections:
            lista.append({'Título': titulo, 'Sección': s.title, 'Texto': s.text})
        print(lista)

if __name__ == "__main__":
    main()
