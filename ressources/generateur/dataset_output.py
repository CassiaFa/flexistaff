"""
Écriture d'un dataset YOLO généré : nouveau dataset ou fusion dans un existant.

Partagé par generate_calendar_dataset.py et generate_calendar_custom.py.

Nommage : <prefix>_<index> avec un index complété de zéros (cal_0000, cal_0001…).
CVAT et l'explorateur trient les fichiers par ordre lexicographique : l'index
doit garder la même largeur et former une suite continue.

Deux modes (--mode) :
    new    dataset neuf : le dossier --out doit être vide ou inexistant.
    merge  ajout à un dataset existant (export CVAT, run précédent du
           générateur) : la numérotation reprend après le dernier
           <prefix>_NNNN trouvé (tous splits confondus), manifest.json est
           complété, data.yaml est vérifié (mêmes classes, même ordre).
Sans --mode : nouveau dataset si --out est vide, sinon arrêt avec un message
(on n'écrase jamais un dataset existant).

Reproductibilité : l'image d'index i est tirée avec la graine f"{seed}-{i}".
Comme la fusion reprend l'index global, une fusion avec la même --seed produit
des images différentes du premier lot, et la suite des commandes
(new puis merge…) régénère exactement le même dataset.
"""

import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic"}
SPLITS = ("train", "val", "test")
MIN_WIDTH = 4


def add_output_args(ap, default_out):
    """Options communes de sortie (--out, --mode, --prefix, --start)."""
    g = ap.add_argument_group("sortie / fusion de datasets")
    g.add_argument("--out", type=Path, default=Path(default_out),
                   help=f"dossier du dataset (défaut : {default_out})")
    g.add_argument("--mode", choices=["new", "merge"], default=None,
                   help="new : nouveau dataset (dossier vide ou inexistant) ; "
                        "merge : ajout à un dataset existant, la numérotation reprend "
                        "après la dernière image <prefix>_NNNN. Sans --mode : new si "
                        "--out est vide, sinon arrêt (jamais d'écrasement)")
    g.add_argument("--prefix", default="cal",
                   help="préfixe des fichiers générés (défaut : cal). Ex. --prefix synth "
                        "pour distinguer les images synthétiques des photos d'un export "
                        "CVAT ; la numérotation reprend par préfixe")
    g.add_argument("--start", type=int, default=None,
                   help="index de la première image (défaut : 0 en mode new, "
                        "dernier index + 1 en mode merge)")


# --------------------------------------------------------------------------
# Lecture de l'existant
# --------------------------------------------------------------------------
def _existing_files(out):
    """Images (hors debug/) et labels .txt déjà présents dans out."""
    images, labels = [], []
    if not out.is_dir():
        return images, labels
    for p in out.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(out).parts
        if rel[0] == "debug":
            continue
        if p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
        elif p.suffix == ".txt" and "labels" in rel[:-1]:
            labels.append(p)
    return images, labels


def _names_from_yaml(data, path):
    """`names` d'un data.yaml (liste ou dict {0: a, 1: b}) -> liste ordonnée."""
    names = data.get("names")
    if isinstance(names, list):
        return [str(n) for n in names]
    if isinstance(names, dict):
        try:
            items = sorted((int(k), str(v)) for k, v in names.items())
        except (TypeError, ValueError):
            raise SystemExit(f"{path} : clés de 'names' non numériques : {list(names)}")
        if [k for k, _ in items] != list(range(len(items))):
            raise SystemExit(f"{path} : ids de classes non contigus : {[k for k, _ in items]}")
        return [v for _, v in items]
    raise SystemExit(f"{path} : clé 'names' absente ou illisible")


def _label_path(img_path):
    """Même règle qu'Ultralytics : dernier /images/ -> /labels/, extension .txt."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return Path(sb.join(str(img_path).rsplit(sa, 1))).with_suffix(".txt")


@dataclass
class SplitTarget:
    img_dir: Path                 # dossier où écrire les images du split
    list_file: Path | None = None  # train.txt… si data.yaml pointe vers une liste
    line_prefix: str = ""          # préfixe des lignes ajoutées à list_file
    yaml_entry: str | None = None  # entrée à ajouter à data.yaml si absente


@dataclass
class OutputPlan:
    out: Path
    merge: bool
    prefix: str
    start: int
    width: int
    seed: int
    splits: list                   # split de chaque nouvelle image
    targets: dict                  # split -> SplitTarget
    names: list
    data_yaml: dict | None = None  # data.yaml existant (mode merge)
    manifest: list = field(default_factory=list)
    n_other: int = 0               # images existantes d'un autre préfixe
    n_same: int = 0                # images existantes du même préfixe

    def name(self, i):
        return f"{self.prefix}_{i:0{self.width}d}"

    def image_path(self, split, name):
        return self.targets[split].img_dir / f"{name}.jpg"

    def label_path(self, split, name):
        return _label_path(self.image_path(split, name))

    def rel(self, p):
        return Path(os.path.relpath(p, self.out)).as_posix()


def _split_target(out, data, split):
    """Où écrire les images d'un split d'après data.yaml (dossier ou liste .txt)."""
    entry = (data or {}).get(split)
    if isinstance(entry, list):
        entry = entry[0] if entry else None
    if not entry:
        return SplitTarget(out / "images" / split, yaml_entry=f"images/{split}")
    p = Path(entry) if Path(entry).is_absolute() else out / entry
    if p.suffix != ".txt":
        return SplitTarget(p)
    # Liste d'images (export CVAT « Ultralytics YOLO ») : on imite les lignes existantes.
    lines = [l.strip() for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    if lines:
        prefix = lines[0].rsplit("/", 1)[0] + "/" if "/" in lines[0] else ""
        base = p.parent if prefix.startswith("./") else out
        img_dir = Path(prefix) if Path(prefix).is_absolute() else base / prefix.removeprefix("./")
        return SplitTarget(img_dir, list_file=p, line_prefix=prefix)
    img_dir = out / "images" / split
    return SplitTarget(img_dir, list_file=p,
                       line_prefix="./" + Path(os.path.relpath(img_dir, p.parent)).as_posix() + "/")


def prepare_output(args, names):
    """Vérifie le dossier de sortie et prépare la numérotation. N'écrit rien."""
    out, prefix, n = args.out, args.prefix, args.n
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", prefix):
        raise SystemExit(f"--prefix '{prefix}' invalide : lettres, chiffres, - et _ uniquement")
    images, labels = _existing_files(out)
    has_yaml, has_manifest = (out / "data.yaml").exists(), (out / "manifest.json").exists()
    existing = bool(images or labels or has_yaml or has_manifest)
    mode = args.mode

    if existing and mode is None:
        raise SystemExit(
            f"Le dossier {out} contient déjà un dataset ({len(images)} images"
            f"{', data.yaml' if has_yaml else ''}{', manifest.json' if has_manifest else ''}).\n"
            f"Rien n'a été écrit. Choisissez :\n"
            f"  --mode merge        ajouter les nouvelles images à ce dataset "
            f"(numérotation reprise après la dernière image {prefix}_NNNN)\n"
            f"  --out <autre>       créer un nouveau dataset dans un autre dossier")
    if existing and mode == "new":
        raise SystemExit(
            f"--mode new : le dossier {out} n'est pas vide ({len(images)} images"
            f"{', data.yaml' if has_yaml else ''}{', manifest.json' if has_manifest else ''}).\n"
            f"Rien n'a été écrit. Utilisez --mode merge ou un autre --out.")
    merge = mode == "merge" and existing
    if mode == "merge" and not existing:
        print(f"[info] {out} est vide : --mode merge crée un nouveau dataset.")

    # Numérotation : dernier index du préfixe (images et labels, tous splits)
    pat = re.compile(rf"{re.escape(prefix)}_(\d+)")
    found = [m.group(1) for p in images + labels if (m := pat.fullmatch(p.stem))]
    same_imgs = [p for p in images if pat.fullmatch(p.stem)]
    last = max((int(d) for d in found), default=-1)
    width = max([MIN_WIDTH] + [len(d) for d in found])
    if args.start is None:
        start = last + 1
    else:
        start = args.start
        if start < 0:
            raise SystemExit("--start doit être >= 0")
        if start <= last:
            raise SystemExit(
                f"--start {start} : le dossier contient déjà {prefix}_{last:0{width}d}. "
                f"Rien n'a été écrit. Utilisez --start {last + 1} ou plus (ou omettez --start).")
    end = start + n - 1
    if not found:  # nouveau préfixe : largeur suffisante d'emblée
        width = max(MIN_WIDTH, len(str(end)))
    elif len(str(end)) > width:
        print(f"[attention] l'index {end} dépasse {10 ** width - 1} : les nouveaux noms auront "
              f"{len(str(end))} chiffres, l'ordre lexicographique (CVAT) ne sera plus continu. "
              f"Envisagez un nouveau --prefix ou un nouveau dataset.")

    # data.yaml existant : mêmes classes, même ordre
    data = None
    if merge and has_yaml:
        data = yaml.safe_load((out / "data.yaml").read_text()) or {}
        if not isinstance(data, dict):
            raise SystemExit(f"{out / 'data.yaml'} : contenu illisible")
        theirs = _names_from_yaml(data, out / "data.yaml")
        if theirs != list(names):
            raise SystemExit(
                f"Classes incompatibles, rien n'a été écrit.\n"
                f"  data.yaml existant : {theirs}\n"
                f"  labels générés     : {list(names)}\n"
                f"Les ids de classes doivent être identiques (mêmes noms, même ordre). "
                f"Adaptez le fichier de labels (--labels) ou utilisez un autre --out.")
        if "nc" in data and int(data["nc"]) != len(names):
            raise SystemExit(f"{out / 'data.yaml'} : nc = {data['nc']} mais {len(names)} classes")
        root = data.get("path")
        if root and Path(str(root)).is_absolute() and Path(str(root)).resolve() != out.resolve():
            print(f"[attention] data.yaml : path = {root} différent de {out} ; "
                  f"les images sont ajoutées sous {out}.")

    # manifest.json existant
    manifest = []
    if merge and has_manifest:
        try:
            manifest = json.loads((out / "manifest.json").read_text())
        except json.JSONDecodeError as e:
            raise SystemExit(f"{out / 'manifest.json'} illisible ({e}). Rien n'a été écrit.")
        if not isinstance(manifest, list):
            raise SystemExit(f"{out / 'manifest.json'} : une liste JSON est attendue")
        done = {(m.get("seed"), m.get("index")) for m in manifest if "seed" in m}
        dup = [i for i in range(start, end + 1) if (args.seed, i) in done]
        if dup:
            raise SystemExit(
                f"manifest.json contient déjà des images générées avec --seed {args.seed} "
                f"et les index {dup[0]}…{dup[-1]} (autre préfixe ?) : elles seraient identiques. "
                f"Rien n'a été écrit. Changez --seed ou --start.")

    # Répartition des nouvelles images. start = 0 : même tirage qu'avant la fusion.
    n_val, n_test = round(n * args.val_split), round(n * args.test_split)
    if n_val + n_test > n:
        raise SystemExit("--val-split + --test-split doit être <= 1")
    splits = ["val"] * n_val + ["test"] * n_test + ["train"] * (n - n_val - n_test)
    random.Random(args.seed if start == 0 else f"{args.seed}-splits-{start}").shuffle(splits)
    targets = {s: _split_target(out, data, s) for s in sorted(set(splits))}

    plan = OutputPlan(out=out, merge=merge, prefix=prefix, start=start, width=width,
                      seed=args.seed, splits=splits, targets=targets, names=list(names),
                      data_yaml=data, manifest=manifest,
                      n_other=len(images) - len(same_imgs), n_same=len(same_imgs))
    # Garde-fou : aucun fichier existant ne doit être écrasé.
    for k, s in enumerate(splits):
        nm = plan.name(start + k)
        for p in (plan.image_path(s, nm), plan.label_path(s, nm)):
            if p.exists():
                raise SystemExit(f"{p} existe déjà : rien n'a été écrit.")
    return plan


def make_dirs(plan, debug):
    for s, t in plan.targets.items():
        t.img_dir.mkdir(parents=True, exist_ok=True)
        _label_path(t.img_dir / "x.jpg").parent.mkdir(parents=True, exist_ok=True)
    if debug:
        (plan.out / "debug").mkdir(parents=True, exist_ok=True)


def manifest_entry(plan, i, split, name, **fields):
    return {"image": plan.rel(plan.image_path(split, name)), "split": split,
            "seed": plan.seed, "index": i, **fields}


# --------------------------------------------------------------------------
# Écriture finale
# --------------------------------------------------------------------------
def _save_side_file(plan, fname, text):
    """labels_config.yaml / roster_codes.yaml : jamais d'écrasement silencieux."""
    p = plan.out / fname
    if not p.exists() or p.read_text() == text:
        p.write_text(text)
        return
    alt = p.with_name(f"{p.stem}_{plan.name(plan.start)}{p.suffix}")
    alt.write_text(text)
    print(f"[attention] {fname} existant différent : configuration de ce lot "
          f"enregistrée dans {alt.name}")


def finalize_output(plan, new_entries, side_files=None):
    """Écrit data.yaml, manifest.json, listes .txt, fichiers annexes, puis le bilan."""
    out = plan.out
    added = {s: plan.splits.count(s) for s in plan.targets}

    # Listes d'images (data.yaml qui pointe vers train.txt…)
    for s, t in plan.targets.items():
        if t.list_file is None:
            continue
        new = [t.line_prefix + Path(e["image"]).name for e in new_entries if e["split"] == s]
        old = t.list_file.read_text() if t.list_file.exists() else ""
        sep = "" if not old or old.endswith("\n") else "\n"
        t.list_file.write_text(old + sep + "\n".join(new) + "\n")

    yaml_path = out / "data.yaml"
    if plan.data_yaml is None:
        if not plan.merge or not yaml_path.exists():
            if plan.merge:
                print("[info] pas de data.yaml dans le dataset existant : création.")
            # Pas de clé `path` : Ultralytics prend alors le dossier du yaml comme racine
            # (un `path: .` serait résolu par rapport au répertoire courant, pas au yaml).
            entries = {s: plan.targets[s].yaml_entry or f"images/{s}" for s in plan.targets}
            lines = [f"train: {entries.get('train', 'images/train')}",
                     f"val: {entries.get('val', 'images/val')}"]
            if "test" in plan.targets:
                lines.append(f"test: {entries['test']}")
            lines += ["", f"nc: {len(plan.names)}", "names:"]
            lines += [f"  {k}: {n}" for k, n in enumerate(plan.names)]
            yaml_path.write_text("\n".join(lines) + "\n")
    else:
        # data.yaml existant : on ne touche qu'aux entrées de splits manquantes.
        data, changed = plan.data_yaml, []
        for s in ("train", "val") + (("test",) if "test" in plan.targets else ()):
            if not data.get(s):
                t = plan.targets.get(s)
                data[s] = t.yaml_entry if t and t.yaml_entry else f"images/{s}"
                changed.append(s)
        if changed:
            yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
            print(f"[info] data.yaml : entrée(s) ajoutée(s) : {', '.join(changed)} "
                  f"(autres clés conservées)")
        else:
            print("[info] data.yaml existant conservé tel quel (classes vérifiées).")

    (out / "manifest.json").write_text(
        json.dumps(plan.manifest + new_entries, ensure_ascii=False, indent=1))
    for fname, text in (side_files or {}).items():
        if text is not None:
            _save_side_file(plan, fname, text)

    # Bilan
    end = plan.start + len(plan.splits) - 1
    n, s = len(plan.splits), "s" if len(plan.splits) > 1 else ""
    verb = f"ajoutée{s} à" if plan.merge else f"générée{s} dans"
    print(f"{n} image{s} {verb} {out} : {plan.name(plan.start)} → {plan.name(end)}")
    print("  ajoutées : " + " / ".join(f"{s} {added[s]}" for s in SPLITS if s in added))
    totals = {}
    for s, t in plan.targets.items():
        totals[s] = sum(1 for p in t.img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    print("  total dans les dossiers concernés : "
          + " / ".join(f"{s} {totals[s]}" for s in SPLITS if s in totals))
    if plan.merge:
        print(f"  existant avant fusion : {plan.n_same} images {plan.prefix}_*, "
              f"{plan.n_other} autres images")
