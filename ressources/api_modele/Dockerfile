# Bonus : le service dans une image légère (pas de torch : quelques centaines de Mo, pas plusieurs Go).
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY detecteur.py api.py ./
# Si vous avez fait l'exercice /parse : décommentez la ligne suivante.
# COPY mon_parser.py ./

# Aucun identifiant dans l'image : ils sont passés au lancement (docker run -e ...).
ENV SQUAD=team-rocket ALIAS=champion
EXPOSE 8001

# Le modèle est téléchargé au démarrage dans /app/modeles (montez un volume pour le conserver).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
