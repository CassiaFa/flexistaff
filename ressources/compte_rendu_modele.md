# Compte rendu de la squad `<squad>`

> **Mode d'emploi.** Copiez ce fichier dans `livrables/compte_rendu.md` (ne modifiez pas `ressources/`).
> Complétez-le **à la fin de chaque journée**, en équipe : 15 minutes suffisent si vous le tenez à jour.
> Remplacez tout ce qui est entre `< >` ; supprimez les lignes d'aide en italique une fois remplies.
> Préférez des **chiffres** et des **exemples** (une capture, une ligne d'OCR) aux impressions générales.
> Ne mettez **jamais** de mot de passe ni le contenu de votre fichier d'identifiants dans ce document.

| | |
|---|---|
| Squad | `<team-rocket / team-magma / team-aqua>` |
| Membres | `<prénom (rôle principal)>`, … |
| Chef de squad | `<prénom>` |
| Modèle MLflow | `flexistaff_<squad>` |

---

## Jour 1 : labelliser, entraîner, premier champion

### 1.1 Nos classes

*Le détail est dans le guide de labellisation (`livrables/<fichier>`) ; ici, l'essentiel.*

| Classe | Ce qu'elle entoure | Pourquoi elle nous sert pour la suite (OCR, parsing) |
|---|---|---|
| `<classe>` | `<…>` | `<…>` |

Décision de labellisation la plus discutée dans la squad, et ce que vous avez tranché :

> `<…>`

### 1.2 Données

| | Valeur |
|---|---|
| Images labellisées (total) | `<n>` |
| Découpage train / val | `<n_train> / <n_val>` |
| Tag `dataset` dans MLflow | `v1` |
| Problèmes rencontrés (images floues, désaccords, boîtes oubliées…) | `<…>` |

### 1.3 Premiers entraînements

| Run | Modèle (`n`, `s`…) | epochs | imgsz | Précision | Rappel | mAP50 |
|---|---|---|---|---|---|---|
| `<membre>_v1` | | | | | | |
| `<membre>_v1` | | | | | | |

**Champion du jour 1 :** run `<membre>_v1`, version `<n>` de `flexistaff_<squad>`.
Pourquoi celui-là (rappel d'abord, mAP50 pour départager) :

> `<…>`

Une capture d'une prédiction sur une photo (fichier dans `livrables/`) : `<chemin>`

---

## Jour 2 : tester l'OCR, améliorer, promouvoir

### 2.1 Test OCR sur le champion

*Notebook `ressources/test_ocr_champion.ipynb`, jeu de test commun.*

| Classe | Texte lu : plutôt bon / partiel / illisible | Exemple de lecture (copier une ligne) |
|---|---|---|
| `<classe>` | | `<…>` |

Ce que l'OCR nous apprend sur nos boîtes (trop serrées ? trop grandes ? classe inutile ?) :

> `<…>`

### 2.2 Challengers

*Une ligne par tentative, y compris celles qui n'ont rien donné.*

| Run | Ce qui change par rapport au champion | Dataset | Rappel | mAP50 | Mieux / moins bien |
|---|---|---|---|---|---|
| `<membre>_v2` | `<ex. +200 images simulées>` | `v2` | | | |

### 2.3 Décision de promotion

| | Champion avant | Challenger retenu |
|---|---|---|
| Run / version | | |
| Rappel | | |
| mAP50 | | |
| Lecture OCR (impression) | | |

**Décision :** `<promu / pas promu>`. Justification en 5 lignes au plus (ce que vous avez changé, ce que ça a donné, pourquoi cette décision) :

> `<…>`

---

## Jour 3 : du texte lu à l'agenda

### 3.1 Notre parser

*Notebook `parsing_jour3.ipynb` complété.*

| | Valeur |
|---|---|
| Cas de test au vert | `<n> / <total>` (dont `<n>` ajoutés par nous) |
| Formats d'heure reconnus | `<ex. 14h-18h, 14:00-18:00, 9h30>` |
| Comment on trouve la date d'un créneau | `<ex. boîte de la case qui le contient + en-tête du mois>` |
| Corrections OCR appliquées | `<ex. O→0, I→1 dans les chiffres>` |

Un exemple du début à la fin, sur une vraie photo :

```text
Texte OCR      : <…>
Événement      : {"title": "<…>", "date": "<…>", "start_time": "<…>", "end_time": "<…>", "confidence": <…>}
```

### 3.2 Seuils de confiance

| Seuil | Notre valeur | Pourquoi |
|---|---|---|
| Pré-coché | `≥ <0,8>` | `<…>` |
| À vérifier | `<0,5> – <0,8>` | `<…>` |
| Non proposé | `< <0,5>` | `<…>` |

### 3.3 Publication dans l'agenda

| Photo | Événements attendus (comptés à la main) | Proposés par le parser | Corrects | Publiés | Doublons |
|---|---|---|---|---|---|
| `<photo>` | | | | | |

Rappel sur cette photo : `<corrects / attendus>`. Un créneau manqué et pourquoi :

> `<…>`

### 3.4 Bonus : le champion en production (API)

*Si vous avez fait `ressources/api_modele/`.*

| Mesure | Valeur |
|---|---|
| Latence moyenne / p95 (`mesurer_latence.py`, `-n 50`) | `<… ms> / <… ms>` |
| Mémoire du service | `<… Mo>` |
| Taille de l'image Docker (ou de l'environnement) | `<… Mo>` |
| Promotion puis retour arrière avec `/reload` testés | `<oui / non>` |

Pourquoi ONNX plutôt que `best.pt` pour ce service, avec vos chiffres :

> `<…>`

---

## Bilan

**Ce qui marche :**

- `<…>`

**Ce qui ne marche pas encore, et ce qu'on ferait avec une journée de plus :**

- `<…>`

**Si c'était un vrai produit pour FlexiStaff :** qu'est-ce qui manque avant de le confier aux managers d'agence ?

> `<…>`
