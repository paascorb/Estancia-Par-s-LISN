import pickle
import pandas as pd
from transformers import pipeline
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
import json
from transformers import T5ForConditionalGeneration
from huggingface_hub import login
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Usar llm ligero pqrq sustituir los <unk>
    # model = "paascorb/ProposicionadorES-mT5-large"
    # model = T5ForConditionalGeneration.from_pretrained(model)
    # dataset = load_from_disk("Datasets/ProposicionesES_2.hf")
    # dataset.push_to_hub("ProposicionesES")
    # f = open('Datasets/wikipediaLimpiado.json')
    # data = json.load(f)

    # corpus = []
    # for elem in data:
    #     corpus.append(elem["Texto"])

    # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

    # new_tokenizer = tokenizer.train_new_from_iterator(corpus, 52000)

    # tokens = new_tokenizer.tokenize("Hace mucho frío en España.áúóé")
    # print(tokens)

    # model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    # model.resize_token_embeddings(len(tokenizer), mean_resizing=True)

    # new_tokenizer.save_pretrained("Modelos/ProposicionadorES2.0-T5-large")
    # model.save_pretrained("Modelos/ProposicionadorES2.0-T5-large")
    
    # from transformers import file_utils
    # print(file_utils.default_cache_path)
    # login()
    # modelo="Modelos/ProposicionadorES-LFM2-2.6B/checkpoint-64860"
    # model = AutoModelForCausalLM.from_pretrained(
    #     modelo
    # )
    # model.push_to_hub("ProposicionadorES-LFM2-2.6B")
    # pipe = pipeline("text-generation", model=modelo)
    # prompt = """Durante la guerra franco-prusiana de 1870 a 1871, la Guardia Nacional francesa había defendido París, a la vez que el radicalismo de clase obrera crecía entre sus soldados. Tras el establecimiento de la Tercera República en septiembre de 1870 (bajo el mando del jefe del ejecutivo francés Adolphe Thiers desde febrero de 1871) y la completa derrota del ejército francés a manos de los alemanes en marzo de 1871, los soldados de la Guardia Nacional tomaron el control de la ciudad el 18 de marzo. Dieron muerte a dos generales del ejército francés y se negaron a aceptar la autoridad de la Tercera República, intentando en su lugar establecer un gobierno independiente. La Comuna gobernó París durante dos meses, estableciendo políticas que tendían hacia un sistema progresista y antirreligioso de su propio supuesto socialismo, que era una mezcla ecléctica de muchas escuelas del siglo XIX. 
    #             """
    # response = pipe(prompt)
    # print(response[0]["generated_text"])
    # last_checkpoint = "alfsnd/flan-t5-base-spanish-yoremnokki"
    # last_checkpoint = "Modelos/ProposicionadorES2.0-T5-large"
    last_checkpoint = "Modelos/ProposicionadorES-T5-large/checkpoint-5400"
    device = "cuda:0"

    model = T5ForConditionalGeneration.from_pretrained(last_checkpoint)
    model.to(device)
    # model.push_to_hub("ProposicionadorES-mT5-large")
    tokenizer = AutoTokenizer.from_pretrained(last_checkpoint)
    # tokenizer.add_tokens(['í', 'Í', 'ñ'], special_tokens=True)
    # special_tokens_dict = {'additional_special_tokens': ['í', 'Í', 'ñ']}
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # print(tokenizer.all_special_tokens)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    # my_question = "Hace mucho frío en España.áúóé"
    my_question = """Las ballenas tienen un largo cráneo de hasta un tercio de la longitud total de su cuerpo, que en edad adulta mide de 15 a 17 m y pesa de 50 a 80 tn.[1]​ Poseen un estrecho y arqueado maxilar, lo que da a estos animales un perfil convexo. Esta forma permite la presencia de largas barbas, las cuales miden de 5 a 25 m de longitud. A diferencia de los peces, las ballenas tienen la cola dispuesta en un plano horizontal, lo que les facilita la ascensión a la superficie, donde tienen que subir a respirar, aunque pueden aguantar hasta una hora bajo el agua, además, duermen la mitad de su cerebro para no hundirse. Tienen dos espiráculos, orificios nasales, situados en la cima de la cabeza, por los que expulsan vapor de agua acompañado a menudo de mucosidades. La gestación dura unos doce meses y casi siempre tienen un único ballenato, que en el momento de nacer mide cinco metros y medio y pesa alrededor de 3000 kg, el cual alimentan con una leche especialmente nutritiva. Su esperanza de vida es de unos treinta años. Hacen grandes migraciones desde los mares fríos, donde se alimentan, a los cálidos, donde se aparean y reproducen. Son cosmopolitas y también se encuentran en el mar Mediterráneo.[5]​ """
    inputs = tokenizer(my_question, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=4096)
    answer = tokenizer.decode(outputs[0])
    
    print(answer)


#     prompt = """<|startoftext|><|im_start|>system
#                 Eres un chatbot encargado de extraer proposiciones atómicas de un texto. 
#                 Una proposición es atómica cuando es autocontenida: no necesita contexto u otra información para entenderse, por ejemplo: 'Morten Harket sufrió acoso escolar' es correcta mientras que 'Sufrió acoso escolar' no es autocontenida ya que no se sabe a quién hace referencia; es mínimal: no se puede descomponer en proposiciones más sencillas; y es coherente: tiene sentido, es decir, es una frase completa con sujeto y predicado, una palabra suelta o una sigla no es una proposición. 
#                 Solo debes listar las proposiciones atómicas separadas por puntos y un salto de línea (\n)sin dar ninguna información extra; por ejemplo: París es una ciudad. París es la capital de Francia.
#                 Dado el siguiente texto de ejemplo: 'Morten Harket nació en Kongsberg siendo el segundo de 5 hijos. Su padre, Reidar, fue médico en un hospital y su madre Henny fue maestra de economía del hogar en una escuela. Morten fue un chico soñador y sufrió acoso escolar en la escuela. Se destacaba en las clases de religión y pensó ingresar en un seminario para ser ministro en la Iglesia Luterana. Entró a formar parte de A-ha el 14 de septiembre de 1982, por invitación de Paul Waaktaar-Savoy y Magne Furuholmen. Sin embargo antes de esto fue el vocalista de un grupo local llamado Soldier Blue. A-ha fue un grupo de gran éxito en los años 80, siendo su canción «Take on Me» la de mayor éxito, de su disco debut «Hunting High and low» vendiendo millones de discos, donde Morten Hartet se convirtió rápidamente en un rompe corazones por aquellas primeras imágenes del vídeo musical «Take on Me» en donde su aspecto de galán nórdico causó gran sensación entre el público femenino. Después de que A-ha se separara en 1994 por conflictos internos, Morten emprendió una carrera como solista cantante y compositor y participó en proyectos medioambientales. Más importante para él fue la lucha por liberar Timor Oriental de la ocupación indonesia, un objetivo que fue finalmente cumplido en 1999. En 1996, Morten es el encargado de presentar junto a Ingvild Bryn el Festival de Eurovisión que se celebró en Oslo, donde cantó Heaven's Not For Saints como introducción a la gala (no como representante de Noruega). Morten ha sacado tres álbumes como solista: «Poetenes Evangelium» (1993) y «Vogts Villa» (1996) en noruego y «Wild Seed» (1995) en inglés, este último fue un gran éxito en Noruega, vendiendo más de 160,000 copias. Otros proyectos de Morten son: la banda sonora de la película «Kamilla og Tyven» (1987), en la que además interpretó un personaje, y «Coneheads» (1993). Otras bandas sonoras incluyen el tema central del musical «Sophie's World», «A Jester In Out Town» y la canción «Jungle Of Beliefs» del disco «Cultures Span The World». En el año 2007 presenta «Letter from Egypt» su regreso como solista y dando una pausa a su participación con el trío noruego A-ha. El 14 de noviembre de 2007 se difundió su primer sencillo «Movies» en la radio en Noruega, colocándose rápidamente en los primeros lugares. En diciembre de 2007 tuvo una presentación en el Premio Nobel de la Paz.'
#                 Serían proposiciones atómicas correctas: Morten Harket nació en Kongsberg. Morten Harket es el segundo de 5 hijos. El padre de Morten Harket se llama Reidar. Reidar Harket fue médico de un hospital.
#                 Serían proposciones átomicas incorrectas: Él nació en Kongsberg. (No autocontenida); Su madre fue maestra de economía del hogar. (No autocontenida); Morten Harket fue un chico soñador y sufrió acoso escolar. (No minimal, se puede separar en: Morten Harket fue un chico soñador. Morten Harket sufrió acoso escolar.); Morten Harket fue astronauta. (Incorrecta, ya que esa infromación no está en el texto);
#                 Por último, es importante que si no logras extraer proposiciones de ninguna forma pongas: 'No se pueden extaer proposiciones.'.""

#                 Dado el siguiente texto:
#                 Durante la guerra franco-prusiana de 1870 a 1871, la Guardia Nacional francesa había defendido París, a la vez que el radicalismo de clase obrera crecía entre sus soldados. Tras el establecimiento de la Tercera República en septiembre de 1870 (bajo el mando del jefe del ejecutivo francés Adolphe Thiers desde febrero de 1871) y la completa derrota del ejército francés a manos de los alemanes en marzo de 1871, los soldados de la Guardia Nacional tomaron el control de la ciudad el 18 de marzo. Dieron muerte a dos generales del ejército francés y se negaron a aceptar la autoridad de la Tercera República, intentando en su lugar establecer un gobierno independiente. La Comuna gobernó París durante dos meses, estableciendo políticas que tendían hacia un sistema progresista y antirreligioso de su propio supuesto socialismo, que era una mezcla ecléctica de muchas escuelas del siglo XIX. 

#                 Lista las proposiciones atómicas:
#                 <|im_end|>
#                 <|im_start|>assistant
#                 """
    

#     tokenizer = AutoTokenizer.from_pretrained(modelo)
#     chat = [
#     {"role": "user", "content": """Martha Poma nació en la comunidad de Chiñaja del municipio de Ancoraimes en la provincia de Omasuyos del departamento de La Paz el 19 de noviembre de 1964. Poma es la hija única de 11 hermanos y tiene 4 hijos. 
# Desde sus 4 años (1968), Poma vivió entre la ciudad de El Alto y la Provincia de Omasuyos. Hizo sus estudios primarios y secundarios en el colegio Henriette de la Chevalier de la ciudad de La Paz, saliendo bachiller en 1982. Desde 1992 hasta 2010 trabajó en el Centro Pachamama Pastoral Social Caritas permaneciendo en ese puesto durante 18 años.
# Fue Secretaria general de la Asociación Artesanal “Pachamama”, ocupó cargos en juntas escolares, como también fue ejecutiva de la Confederación Nacional de Artesanos de Bolivia.
# """},
#     ]

#     response = pipe(tokenizer.apply_chat_template(chat, tokenize=False))
#     print(response[0]["generated_text"])
    
    # dataset = load_from_disk("Datasets/ProposicionesES.hf")
    # print(dataset["messages"][4567])
    # df = pd.read_csv("Datasets/ResultadoProposiciones.csv")
    # f = open('Datasets/wikipediaLimpiado.json')
    # data = json.load(f)
    # resultado = []
    # for index, row in df.iterrows():
    #     aux = [x for x in row.Respuesta.split("\n") if x]
    #     if len(aux) == 1:
    #         aux = [sentence + "." for sentence in aux[0].split(".") if sentence]
    #     resultado.append({'Texto': data[index]["Texto"], 'Proposiciones': aux})
    # with open('Datasets/Proposiciones_1.json', 'w', encoding='utf-8') as f:
    #     json.dump(resultado, f, ensure_ascii=False, indent=4)
    # print("\\nLa construcción se realizó bajo un acuerdo de 5 300 000 000 birr.".replace("\\n", ""))
    # f = open('Datasets/Proposiciones_2.json')
    # data = json.load(f)

    # for elem in data:
    #     aux = elem["Proposiciones"]
    #     if len(aux) == 1:
    #         aux = [x for x in aux[0].split("\n") if x]
    #     if len(aux) == 1:
    #         aux = [sentence + "." for sentence in aux[0].split(".") if sentence]
        
    #     aux2 = []
    #     for e in aux:
    #         aux2.append(e.replace("\\n", ''))
    #     aux = aux2
    #     elem["Proposiciones"] = aux
    # with open('Datasets/Proposiciones_2.json', 'w', encoding='utf-8') as f:
    #     json.dump(data, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    main()