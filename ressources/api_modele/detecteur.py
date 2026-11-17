"""Inférence YOLO avec onnxruntime seul : module FOURNI, ne le modifiez pas.

Pas de torch, pas d'ultralytics : uniquement onnxruntime + OpenCV + numpy. C'est ce qui permet
une image Docker légère et un démarrage rapide, là où l'entraînement demandait plusieurs Go.

Contenu
- ``Detecteur``              : charge un fichier .onnx et détecte les zones d'une image.
- ``telecharger_champion``   : télécharge ``flexistaff_<squad>@<alias>`` depuis le registre MLflow.

Le traitement (letterbox, décodage YOLOv8, seuil, NMS, retour aux coordonnées de l'image
d'origine, noms de classes lus dans les métadonnées ONNX) est celui de l'application CalEvent.

Identifiants MLflow : jamais dans le code. Variables d'environnement
``MLFLOW_TRACKING_URI``, ``MLFLOW_TRACKING_USERNAME``, ``MLFLOW_TRACKING_PASSWORD``
(ou ``IDENTIFIANTS_JSON=identifiants-<login>.json``, le fichier téléchargé depuis le portail).
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import onnxruntime as ort

TAILLE_ENTREE = 640  # côté de l'image carrée attendue par le modèle (imgsz de l'entraînement)


class Detection(NamedTuple):
    id_classe: int
    classe: str
    confiance: float
    boite: list[int]  # [x1, y1, x2, y2] dans les coordonnées de l'image ORIGINALE


# ── Noms de classes ────────────────────────────────────────────────────────────

def lire_noms_classes(session: ort.InferenceSession) -> list[str]:
    """Noms de classes écrits par l'export Ultralytics dans les métadonnées ONNX.

    La clé ``names`` contient un dict sous forme de texte, ex. ``"{0: 'jour', 1: 'evenement'}"``.
    Si elle est absente ou illisible, on renvoie des noms génériques ``class0``, ``class1``...
    (le nombre de classes est alors lu sur la forme de la sortie du modèle).
    """
    try:
        brut = session.get_modelmeta().custom_metadata_map["names"]
        noms = ast.literal_eval(brut)
        if isinstance(noms, dict) and noms:
            # classes numérotées 0..n-1 ; un trou éventuel devient "class<i>"
            return [str(noms.get(i, f"class{i}")) for i in range(max(int(k) for k in noms) + 1)]
        if isinstance(noms, (list, tuple)) and noms:
            return [str(n) for n in noms]
    except Exception:
        pass
    forme = session.get_outputs()[0].shape  # [1, 4 + nb_classes, 8400]
    nb = forme[1] - 4 if isinstance(forme[1], int) else 0
    return [f"class{i}" for i in range(max(nb, 0))]


# ── Détecteur ──────────────────────────────────────────────────────────────────

class Detecteur:
    def __init__(self, chemin_onnx: str | Path, conf: float = 0.45, iou: float = 0.45):
        self.session = ort.InferenceSession(str(chemin_onnx), providers=["CPUExecutionProvider"])
        self.noms_classes = lire_noms_classes(self.session)
        self.nom_entree = self.session.get_inputs()[0].name
        self.conf = conf  # seuil de confiance par défaut
        self.iou = iou    # seuil de recouvrement du NMS

    # Prétraitement
    def _pretraiter(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Redimensionne + padding letterbox (gris 114) -> tenseur NCHW float32 dans [0, 1]."""
        h, w = image.shape[:2]
        echelle = min(TAILLE_ENTREE / h, TAILLE_ENTREE / w)
        nh, nw = int(h * echelle), int(w * echelle)
        redim = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        carre = np.full((TAILLE_ENTREE, TAILLE_ENTREE, 3), 114, dtype=np.uint8)
        carre[:nh, :nw] = redim

        # BGR -> RGB, HWC -> CHW, uint8 -> float32 [0, 1]
        blob = carre[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob[np.newaxis], echelle

    # Post-traitement
    def _postraiter(self, brut: np.ndarray, echelle: float, forme_orig: tuple[int, int],
                    conf: float) -> list[Detection]:
        """Sortie YOLOv8 ONNX [1, 4+nc, 8400] -> liste de Detection (coordonnées d'origine)."""
        h_orig, w_orig = forme_orig
        preds = brut[0].T  # [8400, 4+nc]

        boites_cxcywh = preds[:, :4]
        scores = preds[:, 4:]

        ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(len(scores)), ids]

        masque = confs > conf
        if not masque.any():
            return []

        boites_cxcywh, confs, ids = boites_cxcywh[masque], confs[masque], ids[masque]

        # cx,cy,w,h -> x,y,w,h (coin haut-gauche), format attendu par cv2.dnn.NMSBoxes
        boites_xywh = np.stack(
            [
                boites_cxcywh[:, 0] - boites_cxcywh[:, 2] / 2,
                boites_cxcywh[:, 1] - boites_cxcywh[:, 3] / 2,
                boites_cxcywh[:, 2],
                boites_cxcywh[:, 3],
            ],
            axis=1,
        )

        resultats: list[Detection] = []
        for id_classe in np.unique(ids):  # NMS classe par classe
            m = ids == id_classe
            b, s = boites_xywh[m], confs[m]

            gardees = cv2.dnn.NMSBoxes(b.tolist(), s.tolist(), conf, self.iou)
            if len(gardees) == 0:
                continue

            for i in np.array(gardees).flatten():
                bx, by, bw, bh = b[i]
                # retour aux coordonnées de l'image d'origine, bornées à l'image
                x1 = int(max(0, bx / echelle))
                y1 = int(max(0, by / echelle))
                x2 = int(min(w_orig, (bx + bw) / echelle))
                y2 = int(min(h_orig, (by + bh) / echelle))
                k = int(id_classe)
                resultats.append(Detection(
                    id_classe=k,
                    classe=self.noms_classes[k] if k < len(self.noms_classes) else f"class{k}",
                    confiance=float(s[i]),
                    boite=[x1, y1, x2, y2],
                ))
        return resultats

    # Interface publique
    def detecter(self, image: np.ndarray, conf: float | None = None) -> list[Detection]:
        """Détecte les zones dans une image BGR (celle qu'OpenCV renvoie : ``cv2.imdecode``).

        ``conf`` : seuil de confiance pour cet appel (sinon celui du détecteur).
        """
        blob, echelle = self._pretraiter(image)
        sortie = self.session.run(None, {self.nom_entree: blob})
        seuil = self.conf if conf is None else conf
        return self._postraiter(sortie[0], echelle, image.shape[:2], seuil)


# ── Téléchargement depuis MLflow ───────────────────────────────────────────────

def _configurer_mlflow():
    """Prépare le client MLflow depuis l'environnement, et renvoie le module ``mlflow``."""
    # Serveur injoignable : échouer vite plutôt que de réessayer pendant des minutes
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

    # Option pratique : le fichier identifiants-<login>.json du portail (clé « mlflow »)
    fichier = os.environ.get("IDENTIFIANTS_JSON")
    if fichier and not os.environ.get("MLFLOW_TRACKING_URI"):
        with open(fichier, encoding="utf-8") as f:
            ident = json.load(f)["mlflow"]
        os.environ["MLFLOW_TRACKING_URI"] = ident["tracking_uri"]
        os.environ["MLFLOW_TRACKING_USERNAME"] = ident["username"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = ident["password"]

    manquantes = [v for v in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME",
                              "MLFLOW_TRACKING_PASSWORD") if not os.environ.get(v)]
    if manquantes:
        raise RuntimeError(
            "Identifiants MLflow manquants : " + ", ".join(manquantes)
            + " (ou IDENTIFIANTS_JSON=identifiants-<login>.json)."
        )

    import mlflow  # import tardif : inutile (et lent) tant qu'on ne parle pas au registre
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    return mlflow


def telecharger_champion(squad: str, alias: str = "champion",
                         dossier: str | Path = "modeles") -> tuple[Path, str]:
    """Télécharge ``models:/flexistaff_<squad>@<alias>`` et renvoie ``(chemin_onnx, version)``.

    ``squad``  : slug de l'équipe, ex. ``"team-rocket"``.
    ``alias``  : ``"champion"`` (par défaut) ou ``"challenger"``.
    ``version``: numéro de la version portant l'alias à l'instant de l'appel, ex. ``"3"``.

    Le fichier est mis en cache dans ``<dossier>/flexistaff_<squad>/v<version>.onnx`` : redemander
    un alias qui pointe toujours sur la même version ne retélécharge rien.
    Lève ``RuntimeError`` (identifiants), ``mlflow.exceptions.MlflowException`` (modèle ou alias
    inconnu, serveur injoignable) ou ``FileNotFoundError`` (aucun .onnx dans la version).
    """
    mlflow = _configurer_mlflow()
    nom = f"flexistaff_{squad}"

    # 1. Quelle version porte l'alias en ce moment ?
    version = str(mlflow.MlflowClient().get_model_version_by_alias(nom, alias).version)

    # 2. Téléchargement (une seule fois par version)
    cible = Path(dossier) / nom / f"v{version}.onnx"
    if not cible.exists():
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{nom}/{version}", dst_path=tmp))
            fichiers = sorted(local.rglob("*.onnx")) if local.is_dir() else [local]
            if not fichiers:
                raise FileNotFoundError(f"Aucun fichier .onnx dans {nom} version {version}")
            cible.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(fichiers[0], cible)
    return cible, version
