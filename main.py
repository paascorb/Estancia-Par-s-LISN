import requests
from bs4 import BeautifulSoup

def main():

    URL = "https://es.wikipedia.org/w/api.php"
    PARAMS = {
            "action": "parse",
            "page": "Python (Programmiersprache)",
            "prop": "text",
            "section": 0,
            "format": "json"
            }

    results = requests.get(url=URL, params=PARAMS).json()
    soup = BeautifulSoup(results['parse']['text']['*'], 'html.parser')
    print(soup.prettify())


if __name__ == "__main__":
    main()
