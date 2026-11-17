"""Service de détection FlexiStaff : le champion de votre squad exposé en API (STARTER).

Une API de modèle = un CONTRAT (entrées, sorties, erreurs) que d'autres programmes peuvent utiliser
sans rien savoir de YOLO. Ce fichier fournit le contrat et les utilitaires ; à vous d'écrire le
corps des fonctions marquées « TODO » (6 en tout, le dernier est optionnel).

Lancer :   uvicorn api:app --port 8001          puis ouvrir  http://localhost:8001/docs
Variables : SQUAD (défaut team-rocket), ALIAS (défaut champion), CONF (défaut 0.25),
            MLFLOW_TRACKING_URI / MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD
            (ou IDENTIFIANTS_JSON=identifiants-<login>.json). Voir le README.
"""
from __future__ import annotations

import os
import threading
import time  # noqa: F401  (utile pour mesurer la durée dans /predict)
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from detecteur import Detecteur, telecharger_champion  # noqa: F401  (fournis : voir detecteur.py)

SQUAD = os.environ.get("SQUAD", "team-rocket")
ALIAS = os.environ.get("ALIAS", "champion")
CONF = float(os.environ.get("CONF", "0.25"))  # même seuil que le notebook test_ocr_champion
TAILLE_MAX = 15 * 1024 * 1024                 # 15 Mo : au-delà, on refuse (HTTP 413)


# ── Contrat de l'API (fourni) ──────────────────────────────────────────────────

class Boite(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionOut(BaseModel):
    classe: str
    id_classe: int
    confiance: float
    boite: Boite  # coordonnées en pixels dans l'image ENVOYÉE (pas dans l'image 640x640 du modèle)


class PredictionOut(BaseModel):
    modele: str       # ex. "flexistaff_team-rocket"
    version: str      # version du registre qui répond, ex. "3"
    largeur: int      # taille de l'image reçue
    hauteur: int
    duree_ms: float   # durée de l'inférence seule (hors lecture du fichier et réseau)
    detections: list[DetectionOut]


class SanteOut(BaseModel):
    statut: str       # "ok"
    modele: str
    version: str
    alias: str        # alias demandé au registre : "champion" par défaut
    classes: list[str]


class RechargementOut(BaseModel):
    ancienne_version: str | None
    nouvelle_version: str
    modifie: bool     # False si l'alias pointe toujours sur la même version


# ── État et fonctions utilitaires (fournis) ────────────────────────────────────

# Le modèle courant est UN dict {"detecteur", "version", "alias", "nom"} : /reload le remplace d'un
# bloc (affectation atomique), une requête en cours garde donc un ensemble cohérent.
etat: dict = {"modele": None}
verrou = threading.Lock()  # à utiliser dans /reload : un seul rechargement à la fois


def decoder_image(octets: bytes) -> np.ndarray:
    """Octets d'un fichier image (jpg, png...) -> tableau BGR. Erreur HTTP claire sinon."""
    if len(octets) > TAILLE_MAX:
        raise HTTPException(413, "Image trop volumineuse (maximum 15 Mo).")
    image = cv2.imdecode(np.frombuffer(octets, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Fichier illisible : envoyez une image jpg ou png.")
    return image


def modele_courant() -> dict:
    """Le modèle chargé (dict {"detecteur", "version", "alias", "nom"}), ou HTTP 503 s'il n'y en a pas."""
    if etat["modele"] is None:
        raise HTTPException(503, "Aucun modèle chargé.")
    return etat["modele"]


# ── À COMPLÉTER ────────────────────────────────────────────────────────────────

def charger_modele(alias: str) -> dict:
    """TODO 1 : télécharge le modèle `flexistaff_<SQUAD>@<alias>` et le prépare pour l'inférence.

    Renvoie un dict {"detecteur": Detecteur, "version": str, "alias": alias, "nom": "flexistaff_<SQUAD>"}.
    Étapes : `telecharger_champion(SQUAD, alias)` renvoie (chemin du .onnx, version) ;
    construisez ensuite un `Detecteur(chemin)`.
    Ne modifiez PAS `etat` ici : cette fonction ne fait que fabriquer le dict (utile pour /reload).
    """
    raise NotImplementedError("TODO 1")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """TODO 2 : charger le modèle UNE fois au démarrage (pas à chaque requête !).

    Avant `yield` : rangez le résultat de `charger_modele(ALIAS)` dans `etat["modele"]`.
    Après `yield` (arrêt du service) : remettez `etat["modele"]` à None.
    Si le téléchargement échoue (identifiants, réseau), laissez l'exception remonter :
    un service qui démarre sans modèle est pire qu'un service qui refuse de démarrer.
    """
    yield  # TODO 2 : remplacez cette ligne par le chargement + yield + nettoyage


app = FastAPI(title=f"Détecteur FlexiStaff ({SQUAD})", lifespan=lifespan)


@app.get("/health", response_model=SanteOut)
def health():
    """TODO 3 : état du service. Renvoie SanteOut(statut="ok", modele, version, alias, classes).

    Utilisez `modele_courant()` (503 si rien n'est chargé). Les noms de classes sont dans
    `detecteur.noms_classes` : ce sont VOS classes, lues dans le fichier ONNX.
    Un orchestrateur (Docker, Kubernetes) appelle cette route pour savoir si le service est prêt.
    """
    raise HTTPException(501, "TODO 3")


@app.post("/predict", response_model=PredictionOut)
def predict(fichier: UploadFile = File(...), conf: float = CONF):
    """TODO 4 : image envoyée (multipart, champ « fichier ») -> détections en JSON.

    1. `m = modele_courant()` : gardez cette référence pour toute la requête (un /reload
       concurrent ne doit pas mélanger deux versions dans la même réponse) ;
    2. `image = decoder_image(fichier.file.read())` ;
    3. mesurez la durée de `m["detecteur"].detecter(image, conf=conf)` avec `time.perf_counter()` ;
    4. convertissez chaque Detection (id_classe, classe, confiance, boite=[x1, y1, x2, y2]) en
       DetectionOut, puis renvoyez un PredictionOut (largeur = image.shape[1], hauteur = image.shape[0]).
    Pas de détection ? Une liste vide et un code 200 : ce n'est pas une erreur.
    """
    raise HTTPException(501, "TODO 4")


@app.post("/reload", response_model=RechargementOut)
def reload():
    """TODO 5 : relit l'alias dans le registre MLflow, sans redémarrer le service.

    C'est la boucle « promotion / retour arrière » du jour 2 vue depuis la production :
    le lead déplace l'alias `champion`, on appelle /reload, la nouvelle version répond.
    1. prenez le `verrou` (`with verrou:`) ;
    2. notez la version actuelle (`etat["modele"]`, qui peut être None) ;
    3. `nouveau = charger_modele(ALIAS)` ; si cela échoue, GARDEZ l'ancien modèle et renvoyez
       une HTTPException 502 (le service doit rester disponible) ;
    4. remplacez `etat["modele"]` par `nouveau` et renvoyez un RechargementOut.
    """
    raise HTTPException(501, "TODO 5")


@app.post("/parse")
def parse(ocr: dict = Body(...)) -> list[dict]:
    """TODO 6 (optionnel) : le JSON OCR (format de ocr_champion.json) -> liste d'événements.

    Copiez votre fonction `parser(ocr)` du notebook du jour 3 dans un fichier `mon_parser.py`
    à côté de ce fichier, importez-la et appelez-la. Si le fichier n'existe pas, répondez
    501 avec un message qui explique quoi faire (`try: from mon_parser import parser`).
    Une erreur de VOTRE parser sur une entrée bizarre ne doit pas faire tomber le service.
    """
    raise HTTPException(501, "TODO 6")
