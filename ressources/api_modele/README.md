# Mettre votre champion en production : une API de modèle

**Quand ?** Jour 3, exercice **facultatif mais recommandé**, à répartir dans la squad (durée visée : 1 h 30).
**Prérequis :** un modèle `flexistaff_<squad>@champion` dans MLflow, et votre fichier `identifiants-<login>.json` du portail.

## Pourquoi ?

Jusqu'ici, votre modèle vivait dans un notebook : pour l'utiliser, il fallait Python, ultralytics, torch et
vos fichiers. Un modèle **en production** est un **service avec un contrat** : n'importe quel programme (l'application
CalEvent, une appli mobile, un script de collègue) envoie une image en HTTP et reçoit des détections en JSON, sans rien
savoir de YOLO. Le contrat, ce sont les routes, les formats d'entrée et de sortie, et les erreurs.

Ce service est volontairement léger : **pas de torch, pas d'ultralytics**. Le modèle est le fichier **ONNX** que vous
avez enregistré dans MLflow au jour 1 ; l'inférence est faite par `onnxruntime`.

```
 MLflow (registre)  --telechargement-->  api.py  <--POST /predict (image)--  client (curl, CalEvent...)
 flexistaff_<squad>@champion             detecteur.py (onnxruntime)  --JSON-->
```

## Contenu du dossier

| Fichier | Statut | Rôle |
|---|---|---|
| `detecteur.py` | fourni, complet | inférence ONNX (letterbox, décodage YOLOv8, NMS), `telecharger_champion()` |
| `api.py` | **à compléter** (6 TODO) | l'API FastAPI : contrat et utilitaires fournis, corps des routes à écrire |
| `mesurer_latence.py` | fourni | envoie N requêtes à `/predict` et affiche moyenne / p95 |
| `requirements.txt`, `Dockerfile`, `.dockerignore` | fournis | dépendances ; image Docker (bonus) |

Comme pour les autres ressources, **ne modifiez pas** ce dossier sur place : copiez-le (par exemple dans `livrables/api_modele/`) et travaillez dans la copie.

## Installation et lancement

Sur **votre poste** (pas dans Colab : il n'expose pas de port utilisable). Python 3.10 ou plus :

```bash
cd livrables/api_modele
python -m venv .venv && source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Identifiants MLflow : **jamais dans le code**, jamais dans git. Le plus simple est de pointer le fichier du portail :

```bash
export IDENTIFIANTS_JSON=identifiants-<login>.json      # Windows (PowerShell) : $env:IDENTIFIANTS_JSON="..."
export SQUAD=team-rocket                                  # team-rocket, team-magma ou team-aqua
uvicorn api:app --port 8001
```

(Autre possibilité : `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` et `MLFLOW_TRACKING_PASSWORD`.)
Ouvrez ensuite <http://localhost:8001/docs> : FastAPI génère la documentation interactive du contrat, avec un bouton « Try it out » par route.
Au début, les routes répondent `501` : c'est normal, elles attendent vos TODO.

## Les exercices, dans l'ordre

### 1. Chargement et `/health` (TODO 1, 2, 3)

- **TODO 1** `charger_modele(alias)` : télécharger le modèle et construire le détecteur.
- **TODO 2** `lifespan` : charger le modèle **une seule fois au démarrage**. Pourquoi pas à chaque requête ?
  (mesurez le temps de chargement, vous le verrez.)
- **TODO 3** `GET /health` : nom, version, alias et **classes** du modèle.

Vérification : `curl http://localhost:8001/health` doit afficher la version du registre qui porte `champion`
et **vos** noms de classes.

### 2. Prédire : `/predict` (TODO 4)

Image envoyée en *multipart* (champ `fichier`), réponse : la liste des détections `{classe, confiance, boite}` en pixels de
l'image envoyée.

```bash
curl -F "fichier=@test_ocr/planning_0000.jpg" "http://localhost:8001/predict?conf=0.25"
```

Vérifications : les détections correspondent à celles de `test_ocr_champion.ipynb` pour la même image et le même seuil
(mêmes classes, boîtes proches) ; un fichier qui n'est pas une image donne `400`, pas un plantage `500`.

### 3. Mesurer la latence

```bash
python mesurer_latence.py test_ocr/planning_0000.jpg -n 50
```

Relevez la moyenne et le **p95** (le temps que 95 % des requêtes ne dépassent pas : c'est lui qu'un utilisateur ressent).
Essayez deux images de tailles différentes : qu'est-ce qui change ? (Indice : l'image est toujours ramenée à 640 x 640 avant le modèle.)

### 4. Promotion et retour arrière : `/reload` (TODO 5)

`POST /reload` relit l'alias dans MLflow **sans redémarrer le service**. Scénario à jouer avec votre lead :

1. `curl http://localhost:8001/health` : notez la version du `champion` (par exemple `2`) ;
2. déplacez l'alias `champion` sur une autre version (la promotion du jour 2, ou une version plus ancienne) :
   `client.set_registered_model_alias("flexistaff_<squad>", "champion", "<version>")` ;
3. `curl -X POST http://localhost:8001/reload` : la réponse indique l'ancienne et la nouvelle version ;
   refaites un `/predict` : le champ `version` a changé, les détections aussi, **sans coupure** ;
4. **retour arrière** : remettez l'alias sur la version d'origine, puis `/reload`. C'est la sécurité d'un déploiement : revenir à la version précédente prend quelques secondes.

> L'alias `champion` est partagé (CalEvent, vos camarades) : prévenez la squad, et **remettez-le à sa place** à la fin.
> Variante sans risque : lancez un second service sur un autre port avec `ALIAS=challenger uvicorn api:app --port 8002`
> et comparez les réponses des deux services sur la même image.

Question : que doit faire le service si le rechargement échoue (MLflow injoignable, alias supprimé) ? Votre code le gère-t-il ?

### 5. Facultatif : `/parse` (TODO 6)

Copiez votre `parser(ocr)` du notebook du jour 3 dans `mon_parser.py` (à côté de `api.py`), puis implémentez
`POST /parse` : il reçoit le JSON de `ocr_champion.json` (un extrait suffit) et renvoie la liste d'événements.
Pourquoi ne pas mettre l'OCR (EasyOCR) dans ce service ? Regardez la taille que ça ajouterait.

### 6. Facultatif : Docker

```bash
docker build -t flexistaff-api .
docker run --rm -p 8001:8001 -e SQUAD=team-rocket \
  -e MLFLOW_TRACKING_URI=http://<ip_vm>:5000 -e MLFLOW_TRACKING_USERNAME=<login> -e MLFLOW_TRACKING_PASSWORD=<mot de passe> \
  flexistaff-api
```

Les identifiants sont passés **au lancement** : ils ne doivent jamais être copiés dans l'image (`.dockerignore` protège le dossier).
Depuis un conteneur, `localhost` n'est pas votre machine : utilisez l'adresse IP de la VM pour MLflow, comme ci-dessus.
Relevez `docker images flexistaff-api` (taille) et `docker stats` (mémoire).

## À mettre dans votre rapport du jour 3

Quelques lignes, **chiffrées** :

1. **Latence** : moyenne et p95 de `/predict` (sortie de `mesurer_latence.py`), la machine utilisée, la taille de l'image testée.
2. **Mémoire** : consommation du service en fonctionnement (`docker stats`, ou `ps -o rss= -p <pid>` sur Linux/macOS, le gestionnaire des tâches sous Windows).
3. **Taille de l'image Docker** (si fait) ou de l'environnement Python (`du -sh .venv`), comparée à celle d'un environnement avec torch + ultralytics.
4. **Pourquoi ONNX ?** Argumentez : dépendances, taille, démarrage, portabilité (autre langage, autre matériel), fichier unique versionné dans le registre.
5. **Ce qu'il resterait à faire pour une vraie production** : une ou deux idées (authentification, limite de taille, journalisation, plusieurs requêtes simultanées, surveillance de la qualité...).
6. Le résultat du scénario `/reload` : versions avant / après, et ce qui se passe en cas d'échec.
