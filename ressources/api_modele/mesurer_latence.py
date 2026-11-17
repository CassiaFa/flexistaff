"""Mesure la latence de POST /predict : script FOURNI.

Envoie N fois la même image au service et affiche la moyenne, la médiane et le p95
(« 95 % des requêtes sont plus rapides que cette valeur »), côté client et côté serveur.

Exemple :  python mesurer_latence.py planning.jpg -n 50
           python mesurer_latence.py planning.jpg --url http://localhost:8001 --conf 0.25
"""
import argparse
import statistics
import time
from pathlib import Path

import requests


def percentile(valeurs: list[float], p: float) -> float:
    ordonnees = sorted(valeurs)
    return ordonnees[min(len(ordonnees) - 1, round(p / 100 * (len(ordonnees) - 1)))]


def resume(nom: str, valeurs: list[float]) -> str:
    return (f"{nom:<22} moyenne {statistics.mean(valeurs):7.1f} ms | médiane {statistics.median(valeurs):7.1f} ms"
            f" | p95 {percentile(valeurs, 95):7.1f} ms | min {min(valeurs):7.1f} | max {max(valeurs):7.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="chemin d'une image (jpg/png)")
    ap.add_argument("-n", type=int, default=30, help="nombre de requêtes mesurées (défaut 30)")
    ap.add_argument("--warmup", type=int, default=3, help="requêtes ignorées au début (défaut 3)")
    ap.add_argument("--url", default="http://localhost:8001", help="adresse du service")
    ap.add_argument("--conf", type=float, default=None, help="seuil de confiance (sinon celui du service)")
    args = ap.parse_args()

    octets = Path(args.image).read_bytes()
    params = {} if args.conf is None else {"conf": args.conf}
    session = requests.Session()
    totaux, serveur, derniere = [], [], None

    for i in range(args.warmup + args.n):
        debut = time.perf_counter()
        r = session.post(f"{args.url}/predict", params=params,
                         files={"fichier": (Path(args.image).name, octets, "image/jpeg")}, timeout=60)
        total_ms = (time.perf_counter() - debut) * 1000
        r.raise_for_status()
        derniere = r.json()
        if i >= args.warmup:  # les premières requêtes « chauffent » le modèle : on ne les compte pas
            totaux.append(total_ms)
            serveur.append(derniere["duree_ms"])

    print(f"Modèle {derniere['modele']} version {derniere['version']} | "
          f"image {derniere['largeur']}x{derniere['hauteur']} | {len(derniere['detections'])} détection(s)")
    print(resume("Total (client)", totaux))
    print(resume("Inférence (serveur)", serveur))
    print(f"Débit séquentiel : {1000 / statistics.mean(totaux):.1f} images/s")


if __name__ == "__main__":
    main()
