<div align="center">
  <img src="./static/logo_white.png" alt="FlexiStaff logo" width="450">
</div>

---

Cas pratique sur 3 jours pour la formation **Développeur IA** (École IA Brest - ISEN Ouest).

Votre mission : Contribuer au développement d'une application qui permet de **photographier un planning**, d'en **extrait les créneaux** et les **propose à l'ajout dans un agenda partagé** (CalDAV).

```
 CVAT   →   YOLO   →   OCR   →   Parsing   →   CalDAV
annoter    détecter     lire      structurer     publier
```

> Le guide complet du cas pratique est distribué en formation. Ce README en est la version courte.

## Le contexte

FlexiStaff est une agence d'intérim (logistique, santé, BTP) : 12 agences, 340 intérimaires, 47 entreprises clientes, différents formats de planning. Chaque semaine, les gestionnaires ressaisissent à la main des plannings reçus en photo, en scan ou en PDF : 14 h par semaine et par agence, 1 créneau sur 12 erroné, 48 h de délai pour répercuter un changement.

| Dans le périmètre | Hors périmètre |
|---|---|
| ✓ Détecter les zones utiles sur la photo | ✗ Développer l'application mobile |
| ✓ En extraire le texte | ✗ Entraîner un moteur OCR |
| ✓ Reconstituer des créneaux structurés | |
| ✓ Les publier dans l'agenda (CalDAV) | |

## Les 3 jours

| Jour | Au programme | Livrables |
|---|---|---|
| **J1** | Labellisation (CVAT), préparation du dataset, premier entraînement YOLO **tracé dans MLflow** | Guide de labellisation, dataset exporté, premier run MLflow |
| **J2** | Évaluation, amélioration, comparaison des runs, registre de modèles, export ONNX | 2 runs comparés, modèle enregistré, `best.onnx` |
| **J3** | Application CalEvent : OCR, parsing, publication CalDAV | Pipeline de bout en bout, démonstration |

## Infrastructure

### Les services déployés

Les services tournent sur une VM fournie par l'intervenant. Son adresse (`<IP_VM>`) et vos identifiants vous sont remis en cours.

| Service | Rôle | Port | URL |
|---|---|---|---|
| CVAT | Annotation des images | 8080 | `http://<IP_VM>:8080` |
| MLflow | Suivi des entraînements, registre de modèles | 5000 | `http://<IP_VM>:5000` |
| RustFS (console) | Stockage S3, interface web | 9001 | `http://<IP_VM>:9001` |
| RustFS (API) | Stockage S3 | 9000 | `http://<IP_VM>:9000` |
| Radicale | Serveur CalDAV | 5232 | `http://<IP_VM>:5232/.web/` |
| CalEvent | Backend de l'application (sur votre poste) | 8000 | `http://localhost:8000` |

Vous pouvez tester la connexion aux service :

```bash
curl -I http://<IP_VM>:8080            # CVAT
curl    http://<IP_VM>:5000/health     # MLflow
curl -I http://<IP_VM>:5232/.web/      # Radicale
```

> **Ne committez jamais d'identifiants** (mots de passe, clés) dans ce dépôt.

### Portail de formation

Afin de vous faciliter l'accès aux différents service, un portail a été mis en place qui regroupe des accès directe aux différents services. Un identifiant et un mot de passe unique vous permettrons de vous connecter à chaque service qui le demande.

Le portail vous permet également de vous organiser en équipe, et regroupes les inforamtions nécessaire pour la connexion de vos entraînement avec mlflow.

## Organisation du travail

Chaque squad travaille dans **son propre fork** de ce dépôt : le fork est votre espace d'équipe.

```
flexistaff
├── docs/          # Documents du cas pratique (fournis par l'intervenant)
├── ressources/    # Ressources complémentaires publiées en cours de formation
├── static/        # Ressources statiques (images, logo…)
├── livrables/     # Vos livrables documentaires (guide de labellisation, comptes rendus…)
├── ai_training/   # Votre code d'entraînement (notebooks, scripts…)
├── dataset/       # Votre dataset local (non versionné, voir .gitignore)
├── OCR/           # Votre code OCR et parsing
├── .gitignore           
└── README.md
```

> **Ne modifiez pas** `docs/`, `ressources/` ni `static/` : ces dossiers sont mis à jour par l'intervenant. Vous pourrez ainsi récupérer les mises à jour sans conflit.

Les modèles (`best.pt`, `best.onnx`) ne sont pas versionnés dans Git : ils sont **enregistrés dans MLflow**. Le dataset reste disponible sur CVAT.

### Récupérer les mises à jour

Des ressources pourront être ajoutées sur la branche `main` du dépôt d'origine pendant la formation. Pour les récupérer dans votre fork :

- **Depuis GitHub** : bouton **Sync fork** sur la page de votre fork, puis `git pull` en local.
- **En ligne de commande** :

```bash
git remote add upstream <URL_DU_DEPOT_ORIGINAL>   # une seule fois
git pull upstream main
git push
```

## Aide-mémoire

### Jour 1 : labelliser, entraîner, tracer

- CVAT : choisir l'organisation **FlexiStaff** (en haut à droite), pas l'espace personnel pour voir les tâches de labellisation.
- Décidez d'abord *ensemble* des classes et des règles de tracé, puis écrivez le guide de labellisation.
- Export CVAT au format YOLO, puis découpage `train` / `val` (≈ 80 / 20) :

```
dataset/
├── data.yaml
├── images/{train,val}/
└── labels/{train,val}/
```

```yaml
# data.yaml
path: /content/dataset
train: images/train
val: images/val
nc: 3
names:
  0: classe_0
  1: classe_1
  2: classe_2
```

**Premier entraînement, tracé dans MLflow dès le jour 1 :**

> **⚠️** Adapter les noms de squad (<squad_A>, <A>, etc...) en fonction de votre équipe, membre, etc ...

```python
import mlflow
from ultralytics import YOLO

mlflow.set_tracking_uri("http://<IP_VM>:5000")
mlflow.set_experiment("flexistaff_<squad_A>")

model = YOLO("yolov8s.pt")

with mlflow.start_run("<membre>_<version>"):
    mlflow.set_tags({"squad": "<A>", "dataset": "<version_dataset>"})
    mlflow.log_params({"model": "yolov8s", "epochs": 50, "imgsz": 640})
    res = model.train(data="data.yaml", epochs=50, imgsz=640)
    mlflow.log_metric("mAP50", res.box.map50)
    mlflow.log_artifact("runs/detect/train/weights/best.pt")
```

Interface MLflow : `http://<IP_VM>:5000`. Nommez vos runs `<membre>_<version>` (ex. `toto_v1`). La ligne de commande `yolo detect train` ne trace rien dans MLflow : utilisez le code Python.


### Jour 2 : évaluer, améliorer, exporter

Partez de votre run `<membre>_v1` comme référence : chaque amélioration (`v2`, `v3`, ...) est un nouveau run, à comparer dans MLflow (*Compare*). Enregistrez aussi la précision et le rappel, pas seulement le mAP.

| Métrique | Question | Trop bas = |
|---|---|---|
| Précision | Parmi mes détections, combien sont justes ? | fausses alarmes |
| Rappel | Parmi les vrais objets, combien ai-je trouvés ? | oublis |
| mAP50 | Précision moyenne à IoU ≥ 0,5 | - |

Le **rappel** est la métrique qui compte pour nous : une cellule oubliée est un créneau perdu.

Enregistrez le modèle exporté dans le run MLflow correspondant :

```python
with mlflow.start_run(run_id="<RUN_ID>"):
    mlflow.log_artifact("runs/detect/train/weights/best.onnx")
```

### Jour 3 : OCR parsing et intégration

Interface : `http://localhost:8000` · API : `http://localhost:8000/docs` · Comparaison : `http://localhost:8000/compare-view`

Parsing défensif : corriger les confusions OCR (`O→0`, `I→1`, `Z→2`) **dans les chiffres uniquement**, tolérer plusieurs formats d'heure, croiser avec l'en-tête de date, baisser la confiance à chaque correction.

| Confiance | Comportement |
|---|---|
| ≥ 0,8 | pré-coché |
| 0,5 – 0,8 | à vérifier |
| < 0,5 | non proposé |

Événement CalDAV : `UID` unique, fuseau explicite (`TZID=Europe/Paris`), vérification anti-doublon avant insertion.

## Dépannage rapide

| Problème | Piste |
|---|---|
| CVAT n'affiche rien | Choisir l'organisation FlexiStaff ; vérifier qu'un job vous est assigné |
| MLflow injoignable depuis Colab | `curl http://<IP_VM>:5000/health` ; prévenir l'intervenant |
| `CUDA out of memory` | Réduire `batch` ou `imgsz`, ou prendre le modèle `n` |
| Bon score sur `train`, mauvais sur `val` | Surapprentissage : plus de données ou d'augmentation |
| CalDAV 401 | Identifiants incorrects, tester sur l'interface web de Radicale |
| Événements décalés d'1 h ou 2 h | Fuseau manquant : `TZID=Europe/Paris` |

## Ressources

[Ultralytics](https://docs.ultralytics.com) · [CVAT](https://docs.cvat.ai) · [MLflow](https://mlflow.org/docs/latest) · [EasyOCR](https://github.com/JaidedAI/EasyOCR) · [python-caldav](https://caldav.readthedocs.io) · [Radicale](https://radicale.org/v3.html)
