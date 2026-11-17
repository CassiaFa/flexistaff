# Tag `class_map` : vos classes → les rôles de CalEvent

CalEvent ne connaît pas vos noms de classes. Il lit des **rôles** : `calendar`, `month_header`,
`day_header_row`, `date_cell`, `event` et `resource_header`. Le tag `class_map`, posé sur votre modèle
enregistré dans MLflow, lui indique à quel rôle correspond chacune de vos classes.

## Qui le pose, et comment

- **Qui :** le chef de squad, une seule fois.
- **Comment :** dans la section « Fiche du modèle » de `entrainement_jour1.ipynb`. Remplir `CLASS_MAP` avec
  la ligne de votre équipe ci-dessous, puis exécuter la cellule.
- **Pour quels modèles :** le tag est posé sur le modèle enregistré `flexistaff_<squad>`. Il vaut donc pour
  le champion comme pour les futurs challengers. CalEvent le relit au rechargement (↻).

Format : `classe=rôle`, séparés par des virgules, sans espace. Une classe absente de la liste est ignorée par
le parser.

## Valeur par équipe

| Équipe | `CLASS_MAP` |
|---|---|
| Team Rocket | `Titre_calendrier=month_header,Date_case=date_cell,Ressource=resource_header,Evenement=event` |
| Team Magma | `Titre=month_header,Nom=resource_header,Date=date_cell,Cellule=event,Cellule_code=event` |
| Team Aqua | `Planning=calendar,ref_semaine_planning=day_header_row,employe_poste=resource_header,horaire_lieu_jour_tache=event` |

Les noms de classes doivent être écrits **exactement** comme dans votre `data.yaml` (majuscules comprises).

## Cas particuliers

- **Team Magma :** une classe n'a qu'un rôle, `Cellule` est donc lue comme `event`. En grille de mois, la
  case du jour entière est reconnue à son numéro en tête de texte : le jour vient de ce numéro, chaque ligne
  de la case devient un événement, et une case qui ne porte que son numéro est écartée.
- **Team Aqua :** `ref_semaine_planning` encadre la rangée des jours avec leurs dates (« LUNDI 30/12 … »).
  Le rôle `day_header_row` fait découper cette rangée en un en-tête par date, et chaque créneau prend la
  date de sa colonne. Les cases sans texte sont écartées. **L'année vient du titre** (« Planning du 30
  décembre 2024 ») : encadrez-le aussi en `ref_semaine_planning`, dans la même boîte que les jours ou dans
  une boîte à part. Sans le titre, « 30/12 » donne le jour et le mois, mais la date reste « à revoir ».

## Vérifier

- **Dans MLflow :** onglet *Models*, puis `flexistaff_<squad>`. Le tag `class_map` apparaît dans la section
  *Tags* du modèle.
- **Dans CalEvent :** après un rechargement (↻), la route `GET /api/models` de l'API donne, pour chaque modèle,
  la correspondance appliquée (`class_map`) et sa source (`class_map_source`) :
  - `mlflow` : votre tag est lu ;
  - `identité` : aucun tag n'a été trouvé, et vos classes sont prises telles quelles.
