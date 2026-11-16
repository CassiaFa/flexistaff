# Générateur de plannings synthétiques

Outil du **jour 2** : il fabrique des images de plannings (grilles de mois, plannings hebdomadaires,
plannings à codes de poste) avec leurs annotations YOLO exactes, au même format que votre export CVAT
(*Ultralytics YOLO Detection 1.0* : `data.yaml`, `images/` et `labels/` par split). Vous les ajoutez à
votre dataset pour ré-entraîner dans Google Colab.

Le formateur fournit votre fichier de labels `labels/<votre-equipe>.yaml` (classes de votre équipe associées
aux rôles du générateur), à récupérer par `git pull` depuis le dépôt formateur.

## Utilisation (Colab, dossier courant = racine du dépôt)

```
!python ressources/generateur/generate_calendar_custom.py --labels ressources/generateur/labels/team-rocket.yaml --n 200 --prefix synth --mode merge --out dataset
```

`--out` doit pointer sur le dossier de votre export CVAT décompressé (celui qui contient `data.yaml`).

## Options

| Option | Rôle | Défaut |
|---|---|---|
| `--labels` | fichier de labels de l'équipe | `configs/labels/default.yaml` |
| `--n` | nombre d'images à générer | 200 |
| `--prefix` | préfixe des fichiers générés ; utilisez `synth` | `cal` |
| `--mode` | `merge` : ajout à un dataset existant ; `new` : nouveau dataset | arrêt si `--out` non vide |
| `--out` | dossier du dataset | |
| `--seed` | graine aléatoire (changez-la pour obtenir d'autres images) | 42 |
| `--val-split` | part des images envoyées en validation | 0.2 |
| `--month-ratio` | part de grilles mensuelles | 0.5 |
| `--roster-ratio` | part de plannings à codes de poste (`month-ratio + roster-ratio <= 1`, le reste : plannings hebdo) | 0 |
| `--clean-ratio` | part d'images sans dégradation (photo « propre ») | 0.1 |
| `--debug` | nombre d'images de contrôle avec boîtes dessinées (dossier `debug/`) | 0 |

## Dépendances

`opencv-python-headless` (ou `opencv-python`), `numpy`, `pyyaml`, `pillow` : déjà présents dans Colab.
Les polices libres nécessaires sont fournies dans `fonts/` (voir `fonts/LICENCES.md`).

## Ce que vérifie `--mode merge`

- l'**ordre des classes** de votre fichier de labels doit être identique à celui du `data.yaml` existant,
  sinon le script s'arrête sans rien écrire ;
- **aucun écrasement** : la numérotation reprend après la dernière image `synth_NNNN` ; vos images CVAT ne
  sont jamais modifiées ;
- les images vont dans `images/train` et `images/val` (selon `--val-split`), les labels dans `labels/`.

## Limites

- Types de documents : grilles de mois, plannings hebdomadaires et plannings à codes.
- Environ **300 images** au maximum conseillées.
- Ne jamais entraîner sur du synthétique seul : gardez vos images CVAT, et évaluez sur des **vraies** images.

## Conventions MLflow

- Tag `dataset` : `v1` = les 40 images CVAT, `v2` = v1 + images synthétiques, `v3` = v2 + photos du kit
  d'impression fournies par le formateur.
- Paramètres `n_images_cvat` et `n_images_synth` :

```python
imgs = list(Path("dataset/images/train").glob("*"))
n_synth = sum(p.name.startswith("synth_") for p in imgs)
mlflow.log_params({"n_images_synth": n_synth, "n_images_cvat": len(imgs) - n_synth})
mlflow.set_tag("dataset", "v2")
```
