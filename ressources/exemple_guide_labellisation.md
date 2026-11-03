# Exemple de guide de labellisation

> **Document fictif, fourni à titre d'exemple.** Il porte sur un cas d'usage sans rapport avec FlexiStaff : la détection de déchets sur un tapis de tri. Inspirez-vous de **sa structure et du type d'informations qu'il contient**, pas de son contenu. Les classes et les règles de votre guide sont à définir **en équipe**, à partir de vos propres images.

---

## Projet « TriVision » : guide de labellisation

| | |
|---|---|
| **Version** | 1.2 |
| **Date** | 14/03/2026 |
| **Rédigé par** | Squad Z (Alice, Bilal, Chloé, David) |
| **Outil** | CVAT, organisation *TriVision*, projet `tapis_tri` |
| **Format d'export** | YOLO 1.1 |

### 1. Objectif

Un centre de tri veut détecter automatiquement les déchets qui défilent sur son tapis roulant, pour piloter des buses à air qui les éjectent dans le bon bac.

Le modèle doit **détecter tous les objets recyclables** : un objet oublié part à l'enfouissement. En cas de doute sur la présence d'un objet, **on l'annote** (le rappel prime sur la précision). En revanche, en cas de doute sur sa **classe**, on suit les règles de la section 6.

### 2. Données

- **Source** : 300 images extraites des vidéos de la caméra placée au-dessus du tapis (1 image toutes les 2 secondes, pour éviter les doublons).
- **Conditions variées** : tapis chargé ou presque vide, objets écrasés, sales ou mouillés, éclairage qui varie selon l'heure.
- **Taille** : 1920 × 1080 px. Aucun redimensionnement avant l'annotation.

### 3. Les classes

| ID | Classe | Définition | Exemples | Contre-exemples |
|---|---|---|---|---|
| 0 | `plastique` | Bouteille ou flacon en plastique rigide. | Bouteille d'eau, flacon de shampoing, bidon de lessive | Sac plastique, film étirable, pot de yaourt |
| 1 | `metal` | Emballage en acier ou en aluminium. | Canette, boîte de conserve, aérosol | Papier aluminium froissé, couvercle seul |
| 2 | `carton` | Carton ou papier épais, plat ou plié. | Boîte de céréales, carton de livraison aplati | Papier journal, mouchoir, brique alimentaire |
| 3 | `verre` | Bouteille ou bocal en verre, entier ou cassé en gros morceaux. | Bouteille de vin, pot de confiture | Vaisselle, miroir, éclats de moins de 3 cm |

> **Pourquoi ces classes ?** Elles correspondent aux 4 bacs de sortie du tapis. Une classe qui n'aboutit à aucune action (par exemple « sac plastique ») ne servirait à rien au modèle : ces objets ne sont pas annotés.

### 4. Règles générales de tracé

1. **Rectangles uniquement** (pas de polygones), alignés sur les axes de l'image.
2. **Boîte serrée** : le rectangle englobe l'objet en entier, avec une **marge de 2 à 5 px** au maximum. Il n'inclut pas l'ombre de l'objet.
3. **Un objet = une boîte.** Deux canettes collées : deux boîtes.
4. **Objet partiellement visible** (coupé par le bord de l'image ou caché sous un autre déchet) : on l'annote si **au moins un tiers de l'objet est visible** et qu'on reconnaît sa classe. La boîte couvre **uniquement la partie visible**.
5. **Flou de mouvement** : on annote si un humain reconnaît l'objet et sa classe. Sinon, on n'annote pas.
6. **Chevauchements** : des boîtes peuvent se recouvrir quand les objets se superposent. C'est normal sur un tapis chargé.

### 5. Règles par classe

#### `plastique`
- Le bouchon et l'étiquette font partie de l'objet : ils sont **inclus** dans la boîte.
- Bouteille écrasée à plat : toujours `plastique`.

#### `metal`
- Canette écrasée : toujours `metal`, la boîte suit la forme écrasée.
- Aérosol : `metal`, même si le bouchon en plastique est encore en place.

#### `carton`
- Carton plié ou déchiré en plusieurs morceaux : **une boîte par morceau** de plus de 10 × 10 cm (environ 80 × 80 px sur l'image).
- Carton détrempé et déformé : on l'annote s'il est reconnaissable.

#### `verre`
- Bouteille cassée : on annote chaque morceau de **plus de 3 cm** (environ 25 px). Les éclats plus petits ne sont pas annotés.

### 6. Cas limites (décisions prises en équipe)

| Situation | Décision | Date |
|---|---|---|
| Brique alimentaire (carton + plastique + aluminium) | **Non annotée** : aucun bac ne lui correspond. | 12/03 |
| Bouteille en plastique contenue dans un carton ouvert | Deux boîtes : `carton` et `plastique` (chevauchement accepté). | 12/03 |
| Objet dont on ne sait pas s'il est en verre ou en plastique (bouteille transparente vue de loin) | On zoome. Si le doute persiste : `plastique` (erreur la moins coûteuse pour la machine). | 13/03 |
| Tapis vide sur l'image | Image **conservée** sans annotation : le modèle doit aussi apprendre qu'il n'y a parfois rien à détecter. | 13/03 |
| Image entièrement floue ou surexposée | Image **rejetée** (tag `rejetee` dans CVAT). | 14/03 |

> Toute nouvelle situation ambiguë est signalée sur le canal de l'équipe. **On ne décide pas seul** : la décision est ajoutée à ce tableau et s'applique à tout le monde.

### 7. Ce qu'on n'annote pas

- Les déchets qui ne vont dans aucun bac : sacs et films plastiques, briques alimentaires, textiles, déchets organiques.
- Les éléments du tapis : bords, rouleaux, marquages au sol.
- Les mains ou outils des opérateurs.
- Les objets visibles à moins d'un tiers, ou méconnaissables.

### 8. Exemple annoté

```
┌───────────────────────────────────────────────────────┐
│                                                       │
│ ▓▓ bord du tapis (non annoté) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│                                                       │
│  ┌─────────────┐        ┌───────┐                     │
│  │ [plastique] │        │[metal]│                     │
│  │  bouteille  │        └───────┘     ~ sac plastique │
│  └─────────────┘                        (non annoté)  │
│             ┌──────────────────────┐                  │
│             │ [carton]   ┌───────┐ │           ┌───────│
│             │            │[metal]│ │           │[verre]│
│             │            └───────┘ │           │ coupé │
│             └──────────────────────┘           └───────│
│                                                       │
│ ▓▓ bord du tapis (non annoté) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│                                                       │
└───────────────────────────────────────────────────────┘
  Canette posée sur un carton : deux boîtes qui se chevauchent.
  Bouteille en verre coupée par le bord : boîte sur la partie visible.
```

### 9. Contrôle qualité

- **Double annotation** : 10 % des images (30 images tirées au hasard) sont annotées par deux personnes. On compare les annotations : chaque désaccord est discuté et, si besoin, ajouté aux cas limites (section 6).
- **Revue croisée** : chaque job terminé est relu par un autre membre de l'équipe avant validation dans CVAT (statut *validation* → *acceptation*).
- **Vérifications avant export** :
  - aucune image oubliée (les images sans objet sont volontaires, les rejetées sont taguées) ;
  - aucune boîte minuscule (moins de 5 px de côté) ;
  - répartition des classes vérifiée. Si une classe est très rare (par exemple `verre` < 5 % des objets), on le signale : il faudra peut-être ajouter des images.

### 10. Répartition du travail

| Membre | Images | Relecture de |
|---|---|---|
| Alice | 1 à 75 | David |
| Bilal | 76 à 150 | Alice |
| Chloé | 151 à 225 | Bilal |
| David | 226 à 300 | Chloé |

### 11. Historique des versions

| Version | Date | Modification |
|---|---|---|
| 1.0 | 12/03/2026 | Première version : 4 classes, règles générales. |
| 1.1 | 13/03/2026 | Ajout des règles sur les images vides et le doute verre / plastique. |
| 1.2 | 14/03/2026 | Seuil de visibilité passé de « la moitié » à « un tiers » (trop d'objets oubliés relevés en double annotation). |

---

### Checklist pour rédiger votre propre guide

- [ ] L'objectif est formulé, avec la métrique qui compte (rappel ou précision) et ce que coûte une erreur.
- [ ] Les données sont décrites (source, quantité, conditions).
- [ ] Chaque classe a une définition, des exemples **et des contre-exemples**, et le choix des classes est justifié.
- [ ] Les règles de tracé sont assez précises pour que deux personnes tracent la même boîte.
- [ ] Les objets partiels, flous ou masqués sont traités, avec un seuil chiffré.
- [ ] Ce qu'on n'annote pas est listé explicitement.
- [ ] Les cas limites et les décisions prises sont tracés.
- [ ] Un contrôle qualité est prévu (double annotation, relecture, vérifications avant export).
- [ ] Le document est versionné.
