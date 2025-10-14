import pickle

def main():
    with open('Datasets/resultado_proposiciones.pkl', 'rb') as f:
        resultado = pickle.load(f)
    for elem in resultado:
        print(elem)

if __name__ == "__main__":
    main()