FROM python:3.11

# ustawiamy katalog domyślny dla naszej aplikacji
WORKDIR /usr/src/app

# kopiujemy wszystkie pliki z obecnego katalogu do WORKDIR
COPY requirements.txt .
COPY script.py .

# instalujemy zalezności
RUN pip install -r requirements.txt

# definiujemy port który będzie otwarty w kontenerze
EXPOSE 5000

HEALTHCHECK CMD curl --fail http://localhost:5000

# definiujemy komendę startową
CMD ["streamlit", "run", "./script.py", "--server.port=5000", "--server.address=0.0.0.0"]