#!/usr/bin/env python3
"""
Générateur de dataset synthétique CalEvent (détection YOLO).

Produit des images de calendriers / plannings (grille mensuelle type calendrier
mural, planning hebdomadaire type Excel imprimé) avec les annotations YOLO
exactes et une vérité terrain structurée (year / month / events) pour valider
la suite du pipeline (OCR + parsing).

Classes (celles du CLAUDE.md) :
    0 calendar        le calendrier entier (bandeau du mois + grille)
    1 month_header    le titre du mois ("AVRIL 2026", "SEMAINE 14 — AVRIL 2026")
    2 day_header_row  la rangée (ou colonne si planning transposé) des jours de
                      la semaine : Lun, Mar, ...
    3 date_cell       une cellule portant un numéro de jour
    4 event           un événement (pavé coloré, ligne de texte ou cellule remplie)

Trois types de documents (--month-ratio, --roster-ratio, le reste = week_planning) :
    month_grid     grille 7 colonnes, date_cell = case du jour, event = pavés dedans
    week_planning  planning hebdo, ligne des dates (date_cell) + cellules event ;
                   mise en page normale ou transposée (jours en lignes)
    code_roster    planning d'équipe à codes (une ligne par agent, une colonne par
                   jour, cellules "G03", "RH", "CA"... éventuellement surlignées) ;
                   date_cell = en-tête de colonne du jour, event = case avec code.
                   Les codes viennent de configs/roster_codes.yaml (--roster-codes).

Avec --roster-ratio 0 (défaut), la sortie est identique à celle du dataset v2.

Usage :
    uv run python src/data/generate_calendar_dataset.py --n 300 --out data/datasets/v2
    uv run python src/data/generate_calendar_dataset.py --n 300 --roster-ratio 0.3 --out data/datasets/v3
    uv run python src/data/generate_calendar_dataset.py --n 20 --debug 10 --out /tmp/preview
    # ajout de 100 images à un dataset existant (numérotation reprise, cf. --mode)
    uv run python src/data/generate_calendar_dataset.py --n 100 --out data/datasets/v2 --mode merge

Sans --mode, le script refuse d'écrire dans un dossier qui contient déjà un
dataset ; --mode merge reprend la numérotation après la dernière image
<prefix>_NNNN (voir dataset_output.py).
"""

import argparse
import calendar
import datetime as dt
import math
import os
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from dataset_output import (
    add_output_args, finalize_output, make_dirs, manifest_entry, prepare_output,
)

CLASSES = ["calendar", "month_header", "day_header_row", "date_cell", "event"]
CAL, MONTH, DAYROW, DATE, EVENT = range(len(CLASSES))
_HERE = Path(__file__).resolve().parent
# Codes du planning à codes : configs/ du projet ai_training, sinon copie à côté des scripts (kit étudiant)
DEFAULT_ROSTER_CODES = _HERE.parents[1] / "configs" / "roster_codes.yaml"
if not DEFAULT_ROSTER_CODES.exists():
    DEFAULT_ROSTER_CODES = _HERE / "roster_codes.yaml"

# --------------------------------------------------------------------------
# Polices (la première famille disponible sur la machine est utilisée)
# --------------------------------------------------------------------------
# Ordre de recherche : $GENERATEUR_FONT_DIR (si défini), dossier fonts/ à côté des scripts, polices système
FONT_DIRS = [d for d in (
    Path(os.environ["GENERATEUR_FONT_DIR"]) if os.environ.get("GENERATEUR_FONT_DIR") else None,
    _HERE / "fonts",
    Path("/usr/share/fonts/truetype"),
) if d is not None]
FONT_FAMILIES = {
    "sans": ("liberation/LiberationSans-Regular.ttf", "liberation/LiberationSans-Bold.ttf"),
    "serif": ("liberation/LiberationSerif-Regular.ttf", "liberation/LiberationSerif-Bold.ttf"),
    "carlito": ("crosextra/Carlito-Regular.ttf", "crosextra/Carlito-Bold.ttf"),
    "dejavu": ("dejavu/DejaVuSans.ttf", "dejavu/DejaVuSans-Bold.ttf"),
}


def _find_font(rel):
    """Chemin complet d'une police (premier dossier de FONT_DIRS qui la contient), sinon None."""
    for d in FONT_DIRS:
        if (d / rel).exists():
            return d / rel
    return None


AVAILABLE_FAMILIES = [
    k for k, (r, b) in FONT_FAMILIES.items() if _find_font(r) and _find_font(b)
]
if not AVAILABLE_FAMILIES:
    print("ATTENTION : aucune police trouvée (dossiers testés : "
          + ", ".join(str(d) for d in FONT_DIRS)
          + "). Police par défaut de Pillow utilisée : les images seront moins réalistes. "
          "Vérifiez le dossier fonts/ à côté des scripts.")
    AVAILABLE_FAMILIES = [None]  # None = police par défaut de Pillow


@lru_cache(maxsize=None)
def _font(path, size):
    if path is None:
        return ImageFont.load_default(size)
    return ImageFont.truetype(str(_find_font(path)), size)


class Fonts:
    def __init__(self, family):
        self.reg, self.bold = FONT_FAMILIES[family] if family else (None, None)

    def r(self, size):
        return _font(self.reg, int(size))

    def b(self, size):
        return _font(self.bold, int(size))


# --------------------------------------------------------------------------
# Contenu
# --------------------------------------------------------------------------
JOURS = ["LUNDI", "MARDI", "MERCREDI", "JEUDI", "VENDREDI", "SAMEDI", "DIMANCHE"]
JOURS_ABBR = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
        "août", "septembre", "octobre", "novembre", "décembre"]
MOIS_ABBR = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août",
             "sept.", "oct.", "nov.", "déc."]

AGENDA_TITLES = [
    "Réunion équipe", "RDV médecin", "Dentiste", "Anniversaire Paul", "Livraison",
    "Formation IA", "Point client", "Appel fournisseur", "Atelier", "Soutenance",
    "Cinéma", "Match de foot", "Déjeuner Marie", "Conseil de classe", "Inventaire",
    "Maintenance", "Entretien", "Démo produit", "Visite chantier", "Yoga",
    "Garage", "Séminaire", "Audit qualité", "Coiffeur",
]
POSTES = [
    "Cariste", "Préparateur cde", "Manutentionnaire", "Agent de quai",
    "Chef d'équipe", "Magasinier", "Coffreur", "Maçon VRD", "Conducteur engin",
    "Électricien", "Peintre", "Soudeur", "Agent de tri", "Opérateur ligne",
    "Chauffeur PL", "Grutier",
]
SITES = [
    "Entrepôt Nord", "Quai B", "Site Vénissieux", "Chantier A7", "Plateforme Est",
    "Dépôt Corbas", "Zone logistique 3", "Atelier 2", "Hangar C",
    "Chantier Part-Dieu", "Site Saint-Priest", "Quai 12",
]
NOMS = [
    "M. Bernard", "Mme Dubois", "M. Rossi", "Mme Nguyen", "M. Lefèvre", "M. Garcia",
    "Mme Moreau", "M. Petit", "Mme Fontaine", "M. Silva", "M. Durand", "Mme Leroy",
    "M. Chevalier", "Mme Marchand", "M. Ferreira",
]
ENTREPRISES = [
    "TRANSLOG SAS", "BÂTI-CONSTRUCT", "LOGIPARC", "GROUPE MERCIER", "DISTRI-EST",
    "SOGEBAT", "FRET EXPRESS", "CONSTRUCT'IF",
]
CRENEAUX = [
    ("06:00", "14:00"), ("08:00", "16:00"), ("14:00", "22:00"), ("22:00", "06:00"),
    ("07:30", "12:30"), ("13:00", "17:00"), ("09:00", "18:00"), ("05:00", "13:00"),
    ("08:30", "17:30"),
]


def fmt_time(t, style):
    """'08:30' -> '08:30' | '8h30' | '08h30' selon le style du document."""
    if style == 0:
        return t
    h, m = t.split(":")
    if style == 1:
        return f"{int(h)}h{m}" if m != "00" else f"{int(h)}h"
    return f"{h}h{m}"


def fmt_range(t1, t2, style, sep_style):
    a, b = fmt_time(t1, style), fmt_time(t2, style)
    return [f"{a}-{b}", f"{a} à {b}", f"{a}/{b}", f"de {a} à {b}"][sep_style]


# --------------------------------------------------------------------------
# Palettes (fond, en-tête bg, en-tête fg, grille, texte, accent)
# --------------------------------------------------------------------------
PALETTES = [
    ((255, 255, 255), (68, 114, 196), (255, 255, 255), (180, 180, 180), (0, 0, 0), (222, 235, 247)),
    ((255, 255, 255), (84, 130, 53), (255, 255, 255), (166, 166, 166), (0, 0, 0), (226, 239, 218)),
    ((252, 252, 250), (192, 80, 77), (255, 255, 255), (191, 191, 191), (20, 20, 20), (252, 228, 214)),
    ((255, 255, 255), (64, 64, 64), (255, 255, 255), (150, 150, 150), (0, 0, 0), (233, 233, 233)),
    ((255, 253, 245), (191, 143, 0), (255, 255, 255), (175, 171, 160), (30, 25, 10), (255, 242, 204)),
    ((255, 255, 255), (112, 48, 160), (255, 255, 255), (170, 170, 170), (0, 0, 0), (228, 223, 236)),
]
EVENT_COLORS = [
    (189, 215, 238), (198, 224, 180), (255, 217, 102), (248, 203, 173),
    (214, 198, 232), (255, 192, 203), (178, 223, 219), (225, 225, 140),
]


# --------------------------------------------------------------------------
# Helpers de dessin
# --------------------------------------------------------------------------
def fit_text(d, text, font_fn, size, max_w, min_size=10):
    """Réduit la police puis tronque avec '…' pour que le texte tienne dans max_w."""
    while size > min_size:
        f = font_fn(size)
        if d.textlength(text, font=f) <= max_w:
            return text, f
        size -= 1
    f = font_fn(min_size)
    while text and d.textlength(text + "…", font=f) > max_w:
        text = text[:-1]
    return text.rstrip() + "…", f


def text_box(d, xy, text, font, anchor="lm", pad=3):
    x1, y1, x2, y2 = d.textbbox(xy, text, font=font, anchor=anchor)
    return (x1 - pad, y1 - pad, x2 + pad, y2 + pad)


def draw_banner(d, rng, fonts, text, x1, x2, y, h, hdr_bg, hdr_fg, txt, rec=None):
    """Titre du mois. Bandeau coloré ou texte nu ; retourne la boîte month_header.
    Avec rec (Recorder), le texte du titre est aussi enregistré (rôle text, kind title)."""
    size = int(h * rng.uniform(0.42, 0.56))
    text, font = fit_text(d, text, fonts.b, size, (x2 - x1) * 0.9)
    cy = y + h / 2
    if rng.random() < 0.65:
        d.rectangle([x1, y, x2, y + h], fill=hdr_bg)
        xy, anchor, fill, box = ((x1 + x2) / 2, cy), "mm", hdr_fg, (x1, y, x2, y + h)
    else:
        if rng.random() < 0.5:
            xy, anchor = ((x1 + x2) / 2, cy), "mm"
        else:
            xy, anchor = (x1 + 10, cy), "lm"
        fill, box = txt, None
    d.text(xy, text, font=font, fill=fill, anchor=anchor)
    if rec is not None and text:
        rec.add("text", text_box(d, xy, text, font, anchor, pad=2), kind="title", parent="month_header")
    return box or text_box(d, xy, text, font, anchor, pad=6)


def mix(c1, c2, t):
    return tuple(int(a * (1 - t) + b * t) for a, b in zip(c1, c2))


def pick_time(rng):
    h = rng.randint(7, 20)
    start = f"{h:02d}:{rng.choice(['00', '00', '30', '15', '45'])}"
    if rng.random() < 0.4:
        sh, sm = map(int, start.split(":"))
        end_min = min(sh * 60 + sm + rng.choice([30, 45, 60, 90, 120]), 23 * 60 + 45)
        return start, f"{end_min // 60:02d}:{end_min % 60:02d}"
    return start, None


def event_truth(date, title, t1, t2, **extra):
    return {"day": date.day, "date_iso": date.isoformat(), "title": title,
            "time_start": t1, "time_end": t2, **extra}


# --------------------------------------------------------------------------
# Type 1 : grille mensuelle (calendrier mural)
# --------------------------------------------------------------------------
def make_month_grid(rng):
    W = rng.choice([1200, 1400, 1600])
    H = rng.choice([1000, 1100, 1200])
    bg, hdr_bg, hdr_fg, grid, txt, accent = rng.choice(PALETTES)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    fonts = Fonts(rng.choice(AVAILABLE_FAMILIES))

    year, month = rng.randint(2024, 2027), rng.randint(1, 12)
    first_wd = rng.choice([0, 0, 0, 6])  # lundi en tête (ou dimanche)
    weeks = calendar.Calendar(firstweekday=first_wd).monthdatescalendar(year, month)
    n_rows = len(weeks)

    mx = rng.randint(40, 80)
    top = rng.randint(35, 70)
    banner_h = rng.randint(70, 100)
    wd_h = rng.randint(38, 52)
    x1, x2 = mx, W - mx
    gy1 = top + banner_h + wd_h
    gy2 = H - rng.randint(40, 90)
    col_w = (x2 - x1) / 7
    row_h = (gy2 - gy1) / n_rows

    boxes, events = [], []
    boxes.append((CAL, x1, top, x2, gy2))

    # Titre du mois
    title = f"{MOIS[month - 1].upper()} {year}" if rng.random() < 0.6 else f"{MOIS[month - 1].capitalize()} {year}"
    boxes.append((MONTH, *draw_banner(d, rng, fonts, title, x1, x2, top, banner_h, hdr_bg, hdr_fg, txt)))

    # Rangée des jours de la semaine
    wd_y1 = top + banner_h
    abbr = rng.random() < 0.5
    dark_hdr = rng.random() < 0.5
    for c in range(7):
        cx1 = x1 + c * col_w
        wd = (first_wd + c) % 7
        d.rectangle([cx1, wd_y1, cx1 + col_w, gy1], fill=hdr_bg if dark_hdr else accent, outline=grid)
        label, f = fit_text(d, JOURS_ABBR[wd] if abbr else JOURS[wd], fonts.b, 20, col_w - 10)
        d.text((cx1 + col_w / 2, wd_y1 + wd_h / 2), label, font=f,
               fill=hdr_fg if dark_hdr else txt, anchor="mm")
    boxes.append((DAYROW, x1, wd_y1, x2, gy1))

    # Grille
    num_sz = rng.randint(16, 22)
    num_h = num_sz + 14
    num_right = rng.random() < 0.4
    ev_sz = rng.randint(12, 16)
    ev_h = ev_sz + 10
    ev_style = rng.choice(["strip", "strip", "text"])
    tstyle = rng.randint(0, 2)
    max_ev = max(0, int((row_h - num_h - 4) // (ev_h + 3)))
    ev_density = rng.uniform(0.25, 0.7)

    for r, week in enumerate(weeks):
        cy1 = gy1 + r * row_h
        cy2 = cy1 + row_h
        for c, date in enumerate(week):
            cx1 = x1 + c * col_w
            cx2 = cx1 + col_w
            in_month = date.month == month
            if not in_month:
                d.rectangle([cx1, cy1, cx2, cy2], fill=mix(bg, (200, 200, 200), 0.35), outline=grid)
                continue
            weekend = date.weekday() >= 5
            d.rectangle([cx1, cy1, cx2, cy2], fill=mix(bg, accent, 0.5) if weekend else bg, outline=grid)
            nxy = (cx2 - 8, cy1 + num_h / 2) if num_right else (cx1 + 8, cy1 + num_h / 2)
            d.text(nxy, str(date.day), font=fonts.b(num_sz), fill=txt, anchor="rm" if num_right else "lm")
            boxes.append((DATE, cx1, cy1, cx2, cy2))

            n_ev = 0
            if max_ev and rng.random() < ev_density:
                n_ev = min(max_ev, rng.choices([1, 2, 3], [6, 3, 1])[0])
            for k in range(n_ev):
                ey = cy1 + num_h + k * (ev_h + 3)
                t1, t2 = pick_time(rng) if rng.random() > 0.2 else (None, None)
                name = rng.choice(AGENDA_TITLES)
                label = f"{fmt_time(t1, tstyle)} {name}" if t1 else name
                color = rng.choice(EVENT_COLORS)
                if ev_style == "strip":
                    d.rounded_rectangle([cx1 + 4, ey, cx2 - 4, ey + ev_h], radius=4, fill=color)
                    label, f = fit_text(d, label, fonts.r, ev_sz, col_w - 18)
                    d.text((cx1 + 9, ey + ev_h / 2), label, font=f, fill=(20, 20, 20), anchor="lm")
                    ebox = (cx1 + 4, ey, cx2 - 4, ey + ev_h)
                else:
                    label, f = fit_text(d, label, fonts.r, ev_sz, col_w - 28)
                    d.ellipse([cx1 + 7, ey + ev_h / 2 - 4, cx1 + 15, ey + ev_h / 2 + 4], fill=mix(color, (0, 0, 0), 0.35))
                    d.text((cx1 + 19, ey + ev_h / 2), label, font=f, fill=txt, anchor="lm")
                    tb = text_box(d, (cx1 + 19, ey + ev_h / 2), label, f, "lm", pad=3)
                    ebox = (cx1 + 4, tb[1], tb[2], tb[3])
                boxes.append((EVENT, *ebox))
                events.append(event_truth(date, name, t1, t2))

    d.text((x1, H - rng.randint(25, 40)), rng.choice([
        "Calendrier familial", "Planning de l'équipe — usage interne",
        f"Édité le {rng.randint(1, 28):02d}/{month:02d}/{year}"]),
        font=fonts.r(13), fill=(120, 120, 120), anchor="lm")

    truth = {"doc_type": "month_grid", "year": year, "month": month, "events": events}
    return img, boxes, truth


# --------------------------------------------------------------------------
# Type 2 : planning hebdomadaire (Excel imprimé)
# --------------------------------------------------------------------------
def make_week_planning(rng):
    W = rng.choice([1400, 1600, 1700])
    H = rng.choice([900, 1000, 1100])
    bg, hdr_bg, hdr_fg, grid, txt, accent = rng.choice(PALETTES)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    fonts = Fonts(rng.choice(AVAILABLE_FAMILIES))

    # Semaine réelle (les dates peuvent chevaucher deux mois : 30, 31, 1, 2...)
    d0 = dt.date(rng.randint(2024, 2027), 1, 1) + dt.timedelta(days=rng.randint(0, 364))
    monday = d0 - dt.timedelta(days=d0.weekday())
    n_days = rng.choice([5, 6, 6, 7])
    days = [monday + dt.timedelta(i) for i in range(n_days)]
    transposed = rng.random() < 0.25
    abbr = rng.random() < 0.5
    date_fmt = rng.randint(0, 2)  # 3 | 03/04 | 03 avr
    tstyle, sep_style = rng.randint(0, 2), rng.randint(0, 3)
    color_cells = rng.random() < 0.6
    use_names = rng.random() < 0.5
    hdr2 = mix(hdr_bg, (255, 255, 255), 0.25)  # en-têtes de tableau distincts du bandeau

    def date_label(day):
        return [str(day.day), f"{day.day:02d}/{day.month:02d}",
                f"{day.day:02d} {MOIS_ABBR[day.month - 1]}"][date_fmt]

    def day_label(day):
        return JOURS_ABBR[day.weekday()] if abbr else JOURS[day.weekday()]

    mx = rng.randint(45, 75)
    top = rng.randint(28, 50)
    x1, x2 = mx, W - mx
    table_w = x2 - x1
    bottom = H - rng.randint(60, 100)

    ent = rng.choice(ENTREPRISES)
    d.text((x1, top), f"{ent}  —  Personnel intérimaire", font=fonts.r(15),
           fill=(90, 90, 90), anchor="lm")

    iso = monday.isocalendar()
    title = rng.choice([
        f"SEMAINE {iso.week:02d} — {MOIS[monday.month - 1].upper()} {monday.year}",
        f"PLANNING SEMAINE {iso.week:02d}",
        f"Planning du {monday.day:02d} {MOIS[monday.month - 1]} {monday.year}",
    ])
    banner_y, banner_h = top + 22, rng.randint(52, 70)
    banner_y2 = banner_y + banner_h
    boxes, events = [], []
    boxes.append((MONTH, *draw_banner(d, rng, fonts, title, x1, x2, banner_y, banner_h, hdr_bg, hdr_fg, txt)))

    def draw_event(cx1, cy1, cx2, cy2, day, resource, slot, site):
        fill = rng.choice(EVENT_COLORS) if color_cells else None
        d.rectangle([cx1, cy1, cx2, cy2], fill=fill or bg, outline=grid)
        t_txt = fmt_range(slot[0], slot[1], tstyle, sep_style)
        l1, f1 = fit_text(d, t_txt, fonts.r, 17, cx2 - cx1 - 10)
        l2, f2 = fit_text(d, site, fonts.r, 14, cx2 - cx1 - 10)
        mid = (cy1 + cy2) / 2
        d.text(((cx1 + cx2) / 2, mid - 9), l1, font=f1, fill=txt, anchor="mm")
        d.text(((cx1 + cx2) / 2, mid + 11), l2, font=f2, fill=(70, 70, 70), anchor="mm")
        boxes.append((EVENT, cx1, cy1, cx2, cy2))
        events.append(event_truth(day, site, slot[0], slot[1], resource=resource))

    def resources(n):
        return rng.sample(NOMS if use_names else POSTES, n)

    if not transposed:
        label_w = int(table_w * rng.uniform(0.12, 0.18))
        col_w = (table_w - label_w) / n_days
        wd_h, dt_h = rng.randint(38, 50), rng.randint(30, 40)
        y_dt = banner_y2 + wd_h
        y_body = y_dt + dt_h
        n_rows = max(3, min(rng.randint(4, 9), int((bottom - y_body) // 52)))
        row_h = (bottom - y_body) / n_rows
        res = resources(n_rows)
        bx1 = x1 + label_w

        d.rectangle([x1, banner_y2, bx1, y_dt], fill=hdr2, outline=grid)
        d.rectangle([x1, y_dt, bx1, y_body], fill=accent, outline=grid)
        if rng.random() < 0.6:
            d.text((x1 + 8, banner_y2 + wd_h / 2), "Équipe" if use_names else "Poste",
                   font=fonts.b(16), fill=hdr_fg, anchor="lm")
        for j, day in enumerate(days):
            cx1 = bx1 + j * col_w
            d.rectangle([cx1, banner_y2, cx1 + col_w, y_dt], fill=hdr2, outline=grid)
            lab, f = fit_text(d, day_label(day), fonts.b, 19, col_w - 8)
            d.text((cx1 + col_w / 2, banner_y2 + wd_h / 2), lab, font=f, fill=hdr_fg, anchor="mm")
            d.rectangle([cx1, y_dt, cx1 + col_w, y_body], fill=accent, outline=grid)
            lab, f = fit_text(d, date_label(day), fonts.r, 17, col_w - 8)
            d.text((cx1 + col_w / 2, y_dt + dt_h / 2), lab, font=f, fill=txt, anchor="mm")
            boxes.append((DATE, cx1, y_dt, cx1 + col_w, y_body))
        boxes.append((DAYROW, bx1, banner_y2, x2, y_dt))

        p_fill = rng.uniform(0.5, 0.8)
        for i in range(n_rows):
            ry1 = y_body + i * row_h
            ry2 = ry1 + row_h
            d.rectangle([x1, ry1, bx1, ry2], fill=accent if i % 2 == 0 else bg, outline=grid)
            lab, f = fit_text(d, res[i], fonts.r, 16, label_w - 14)
            d.text((x1 + 8, (ry1 + ry2) / 2), lab, font=f, fill=txt, anchor="lm")
            for j, day in enumerate(days):
                cx1 = bx1 + j * col_w
                if rng.random() < p_fill:
                    draw_event(cx1, ry1, cx1 + col_w, ry2, day, res[i],
                               rng.choice(CRENEAUX), rng.choice(SITES))
                else:
                    d.rectangle([cx1, ry1, cx1 + col_w, ry2], fill=bg, outline=grid)
        gy2 = y_body + n_rows * row_h
    else:
        wd_w = int(table_w * rng.uniform(0.13, 0.18))
        dt_w = int(table_w * rng.uniform(0.07, 0.11))
        n_cols = rng.randint(3, 5)
        col_w = (table_w - wd_w - dt_w) / n_cols
        hdr_h = rng.randint(40, 52)
        y_body = banner_y2 + hdr_h
        row_h = (bottom - y_body) / n_days
        res = resources(n_cols)
        bx_wd2 = x1 + wd_w
        bx1 = bx_wd2 + dt_w

        d.rectangle([x1, banner_y2, bx1, y_body], fill=hdr2, outline=grid)
        d.text((x1 + 8, banner_y2 + hdr_h / 2), "Jour", font=fonts.b(17), fill=hdr_fg, anchor="lm")
        for j in range(n_cols):
            cx1 = bx1 + j * col_w
            d.rectangle([cx1, banner_y2, cx1 + col_w, y_body], fill=hdr2, outline=grid)
            lab, f = fit_text(d, res[j], fonts.b, 17, col_w - 10)
            d.text((cx1 + col_w / 2, banner_y2 + hdr_h / 2), lab, font=f, fill=hdr_fg, anchor="mm")

        p_fill = rng.uniform(0.5, 0.8)
        for i, day in enumerate(days):
            ry1 = y_body + i * row_h
            ry2 = ry1 + row_h
            d.rectangle([x1, ry1, bx_wd2, ry2], fill=accent, outline=grid)
            lab, f = fit_text(d, day_label(day), fonts.b, 18, wd_w - 12)
            d.text((x1 + 8, (ry1 + ry2) / 2), lab, font=f, fill=txt, anchor="lm")
            d.rectangle([bx_wd2, ry1, bx1, ry2], fill=accent, outline=grid)
            lab, f = fit_text(d, date_label(day), fonts.r, 17, dt_w - 8)
            d.text(((bx_wd2 + bx1) / 2, (ry1 + ry2) / 2), lab, font=f, fill=txt, anchor="mm")
            boxes.append((DATE, bx_wd2, ry1, bx1, ry2))
            for j in range(n_cols):
                cx1 = bx1 + j * col_w
                if rng.random() < p_fill:
                    draw_event(cx1, ry1, cx1 + col_w, ry2, day, res[j],
                               rng.choice(CRENEAUX), rng.choice(SITES))
                else:
                    d.rectangle([cx1, ry1, cx1 + col_w, ry2], fill=bg, outline=grid)
        boxes.append((DAYROW, x1, y_body, bx_wd2, y_body + n_days * row_h))
        gy2 = y_body + n_days * row_h

    boxes.append((CAL, x1, banner_y, x2, gy2))
    d.text((x1, H - rng.randint(35, 55)), rng.choice([
        f"Édité le {rng.randint(1, 28):02d}/{monday.month:02d} — Document interne",
        f"Contact agence : 04.7{rng.randint(0, 9)}.{rng.randint(10, 99)}.{rng.randint(10, 99)}.{rng.randint(10, 99)}",
        "Toute modification doit être signalée avant 16h la veille."]),
        font=fonts.r(14), fill=(120, 120, 120), anchor="lm")

    truth = {"doc_type": "week_planning", "year": monday.year, "month": monday.month,
             "events": events}
    return img, boxes, truth


# --------------------------------------------------------------------------
# Enregistrement d'éléments par rôle (utilisé par code_roster et par
# generate_calendar_custom.py pour la labélisation personnalisable)
# --------------------------------------------------------------------------
class Recorder:
    """Accumule les éléments dessinés : {"role", "box", **attributs}."""

    def __init__(self, d, **common):
        self.d, self.common, self.elements = d, common, []

    def add(self, role, box, **attrs):
        self.elements.append({"role": role, "box": tuple(float(v) for v in box),
                              **self.common, **attrs})

    def text(self, xy, s, font, fill, kind, anchor="la", parent=None, **attrs):
        self.d.text(xy, s, font=font, fill=fill, anchor=anchor)
        if s:
            self.add("text", text_box(self.d, xy, s, font, anchor, pad=2),
                     kind=kind, parent=parent, **attrs)


def date_attrs(date):
    return {"date": date.isoformat(), "day": date.day,
            "weekday": date.weekday(), "weekend": date.weekday() >= 5}


def time_attrs(t1, t2):
    return {"start": t1, "end": t2, "has_time": t1 is not None,
            "night": bool(t1 and t2 and t2 < t1)}


# Rôles -> 5 classes historiques (les autres rôles ne sont pas annotés)
ROLE_TO_CLASS = {"calendar": CAL, "month_header": MONTH, "day_header_row": DAYROW,
                 "date_cell": DATE, "event": EVENT}


# --------------------------------------------------------------------------
# Type 3 : planning d'équipe à codes (export Excel type hôpital / EHPAD)
# --------------------------------------------------------------------------
ROSTER_CATEGORIES = ("work", "training", "rest", "leave")
NOMS_FAMILLE = [
    "Le Bian", "Le Grand", "Le Juste", "Martin", "Bernard", "Dubois", "Moreau",
    "Laurent", "Simon", "Michel", "Lefebvre", "Leroy", "Roux", "David", "Bertrand",
    "Morel", "Fournier", "Girard", "Bonnet", "Dupont", "Lambert", "Fontaine",
    "Rousseau", "Vincent", "Muller", "Le Gall", "Faure", "André", "Mercier", "Blanc",
]
PRENOMS = [
    "Chloé", "Marcel", "Stéphane", "Julie", "Nicolas", "Sophie", "Thomas", "Camille",
    "Karim", "Léa", "Mathieu", "Inès", "Laurent", "Emma", "Yannick", "Sarah",
    "Pierre", "Nadia", "Hugo", "Élodie", "Fatou", "Antoine", "Manon", "Kevin",
]
SERVICES = [
    "Service Urgences", "Réanimation", "Bloc opératoire", "EHPAD Les Tilleuls",
    "Pôle gériatrie", "Équipe de nuit", "Médecine interne", "Cardiologie",
]
JOURS_LETTRE = ["L", "M", "M", "J", "V", "S", "D"]
MOIS_3 = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun", "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]


def parse_days_range(s):
    """Option --roster-days : "15" -> (15, 15), "10-20" -> (10, 20)."""
    try:
        a, _, b = s.partition("-")
        a, b = int(a), int(b or a)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{s}' : attendu un nombre (15) ou une plage (10-20)")
    if not 1 <= a <= b <= 62:
        raise argparse.ArgumentTypeError(f"'{s}' : nombre de jours entre 1 et 62, min <= max")
    return a, b


def load_roster_codes(path=DEFAULT_ROSTER_CODES):
    """Charge le fichier de codes : {code: {label, category, start, end, color, weight}}."""
    raw = yaml.safe_load(Path(path).read_text())
    codes = {}
    for code, spec in (raw or {}).get("codes", {}).items():
        cat = spec.get("category")
        if cat not in ROSTER_CATEGORIES:
            raise SystemExit(f"{path} : code '{code}', category doit être parmi {ROSTER_CATEGORIES}")
        color = spec.get("color")
        if color:
            color = tuple(int(color.lstrip("#")[k:k + 2], 16) for k in (0, 2, 4))
        codes[str(code)] = {"label": spec.get("label", str(code)), "category": cat,
                            "start": spec.get("start"), "end": spec.get("end"),
                            "color": color, "weight": float(spec.get("weight", 1))}
    if not any(c["category"] != "leave" and c["weight"] > 0 for c in codes.values()):
        raise SystemExit(f"{path} : il faut au moins un code non 'leave' avec weight > 0")
    return codes


def roster_sequence(rng, codes, days):
    """Suite de codes réaliste pour un agent : séries, repos le week-end, congés en blocs."""
    pool = [k for k, c in codes.items() if c["category"] != "leave" and c["weight"] > 0]
    leaves = [k for k, c in codes.items() if c["category"] == "leave"]
    seq, prev, run, leave_left = [], None, 0, 0
    for day in days:
        if leave_left:
            code, leave_left = prev, leave_left - 1
        elif leaves and rng.random() < 0.03:
            code, leave_left = rng.choice(leaves), rng.randint(2, 6)
        elif prev in pool and run < 3 and rng.random() < 0.5:
            code = prev
        else:
            boost = 3 if day.weekday() >= 5 else 1
            w = [codes[k]["weight"] * (boost if codes[k]["category"] == "rest" else 1) for k in pool]
            code = rng.choices(pool, w)[0]
        if rng.random() < 0.03:
            code = None  # case laissée vide
        run = run + 1 if code == prev else 1
        prev = code
        seq.append(code)
    return seq


def luminance(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def make_code_roster(rng, codes, days=None):
    """Planning à codes. Retourne (img, elements, truth) : des éléments par rôle
    (voir Recorder), convertis en classes via ROLE_TO_CLASS ou un fichier de labels.
    days = (min, max) nombre de jours affichés (None = 7 à 31 jours)."""
    fonts = Fonts(rng.choice(AVAILABLE_FAMILIES))
    n_days = rng.randint(*days) if days else rng.choice([7, 14, 14, 21, 28, 31])
    n_people = rng.randint(3, 12)
    d0 = dt.date(rng.randint(2024, 2027), 1, 1) + dt.timedelta(days=rng.randint(0, 364))
    if rng.random() < 0.6:
        d0 -= dt.timedelta(days=d0.weekday())  # commence un lundi
    days = [d0 + dt.timedelta(i) for i in range(n_days)]

    mx, top = rng.randint(30, 60), rng.randint(25, 45)
    col_w = rng.randint(50, 72)
    name_w = rng.randint(130, 190)
    hdr_h = rng.randint(44, 58)
    row_h = rng.randint(40, 56)
    has_title = rng.random() < 0.6
    title_h = rng.randint(50, 66) if has_title else 0
    has_legend = rng.random() < 0.4
    W = 2 * mx + name_w + n_days * col_w
    table_y = top + title_h
    body_y = table_y + hdr_h
    table_y2 = body_y + n_people * row_h
    legend_y = table_y2 + 25
    H = legend_y + (110 if has_legend else 0) + 60

    plain = rng.random() < 0.5  # en-tête sans couleur (export brut) ou palette
    bg, hdr_bg, hdr_fg, _, txt, accent = rng.choice(PALETTES)
    if plain:
        bg, hdr_bg, hdr_fg, txt = (255, 255, 255), (255, 255, 255), (0, 0, 0), (0, 0, 0)
    grid = rng.choice([(208, 208, 208), (190, 190, 190), (150, 150, 150)])
    shade_weekend = rng.random() < 0.4
    weekend_bg = mix(bg, (200, 200, 200), 0.35)
    highlight = rng.choices(["none", "cell", "marker"], [1, 2, 1.3])[0]
    hdr_style = rng.randint(0, 2)
    two_line_names = rng.random() < 0.6
    code_sz = rng.randint(14, 18)

    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    rec = Recorder(d, doc_type="code_roster", layout="roster")
    x1, bx1, x2 = mx, mx + name_w, mx + name_w + n_days * col_w

    if has_title:
        month_names = sorted({(day.year, day.month) for day in days})
        period = " / ".join(f"{MOIS[m - 1].capitalize()} {y}" for y, m in month_names)
        title = rng.choice([f"Planning {rng.choice(SERVICES)} — {period}",
                            f"PLANNING {period.upper()}", f"{rng.choice(SERVICES)}  {period}"])
        rec.add("month_header", draw_banner(d, rng, fonts, title, x1, min(x2, x1 + 900),
                                            top, title_h - 8, hdr_bg, hdr_fg, txt, rec=rec))

    # En-tête : coin + une colonne par jour (2 lignes, cf. "L27 / S18")
    d.rectangle([x1, table_y, bx1, body_y], fill=hdr_bg, outline=grid)
    rec.add("corner", (x1, table_y, bx1, body_y))
    corner_lbl = rng.choice(["Nom /Prénom", "Nom Prénom", "Agent", "NOM"])
    rec.text((x1 + 6, table_y + hdr_h / 2), corner_lbl, fonts.r(15), hdr_fg,
             kind="corner", anchor="lm", parent="corner")
    hdr_s1, hdr_s2 = rng.randint(14, 16), rng.randint(13, 15)
    for j, day in enumerate(days):
        cx1 = x1 + name_w + j * col_w
        da = {"col": j, **date_attrs(day), "week": day.isocalendar().week}
        weekend_col = shade_weekend and day.weekday() >= 5
        d.rectangle([cx1, table_y, cx1 + col_w, body_y],
                    fill=weekend_bg if weekend_col and plain else hdr_bg, outline=grid)
        if hdr_style == 0:
            l1 = f"{JOURS_LETTRE[day.weekday()]}{day.day:02d}"
            l2, k2 = (f"S{da['week']:02d}", "week") if day.weekday() == 0 else (MOIS_3[day.month - 1], "month")
        elif hdr_style == 1:
            l1, l2, k2 = JOURS_ABBR[day.weekday()], f"{day.day:02d}", "date"
        else:
            l1, l2, k2 = JOURS_LETTRE[day.weekday()], f"{day.day:02d}/{day.month:02d}", "date"
        cx = cx1 + col_w / 2
        l1, f1 = fit_text(d, l1, fonts.r, hdr_s1, col_w - 6)
        l2, f2 = fit_text(d, l2, fonts.r, hdr_s2, col_w - 6)
        rec.text((cx, table_y + hdr_h * 0.3), l1, f1, hdr_fg, kind="day_name" if hdr_style else "date",
                 anchor="mm", parent="date_cell", **da)
        rec.text((cx, table_y + hdr_h * 0.72), l2, f2, hdr_fg, kind=k2,
                 anchor="mm", parent="date_cell", **da)
        rec.add("date_cell", (cx1, table_y, cx1 + col_w, body_y), **da)
    rec.add("day_header_row", (bx1, table_y, x2, body_y))

    # Corps : une ligne par agent
    people = rng.sample([(n, p) for n in NOMS_FAMILLE for p in PRENOMS], n_people)
    events = []
    f_name = fonts.r(15)
    for i, (nom, prenom) in enumerate(people):
        ry1 = body_y + i * row_h
        ry2 = ry1 + row_h
        resource = f"{nom} {prenom}"
        d.rectangle([x1, ry1, bx1, ry2], fill=bg, outline=grid)
        ra = {"row": i, "resource": resource}
        if two_line_names:
            rec.text((x1 + 6, ry1 + row_h * 0.3), nom, f_name, txt, kind="resource",
                     anchor="lm", parent="resource_header", **ra)
            rec.text((x1 + 6, ry1 + row_h * 0.72), prenom, f_name, txt, kind="resource",
                     anchor="lm", parent="resource_header", **ra)
        else:
            lab, f = fit_text(d, f"{nom.upper()} {prenom}", fonts.r, 15, name_w - 10)
            rec.text((x1 + 6, (ry1 + ry2) / 2), lab, f, txt, kind="resource",
                     anchor="lm", parent="resource_header", **ra)
        rec.add("resource_header", (x1, ry1, bx1, ry2), **ra)

        for j, (day, code) in enumerate(zip(days, roster_sequence(rng, codes, days))):
            cx1 = x1 + name_w + j * col_w
            cell = (cx1, ry1, cx1 + col_w, ry2)
            cell_bg = weekend_bg if shade_weekend and day.weekday() >= 5 else bg
            pos = {"row": i, "col": j, **date_attrs(day), "resource": resource}
            if code is None:
                d.rectangle(cell, fill=cell_bg, outline=grid)
                rec.add("empty_cell", cell, **pos)
                continue
            c = codes[code]
            color = c["color"] if highlight != "none" else None
            d.rectangle(cell, fill=color if highlight == "cell" and color else cell_bg, outline=grid)
            fg = txt
            lab, f = fit_text(d, code, fonts.r, code_sz, col_w - 8)
            cxy = (cx1 + col_w / 2, (ry1 + ry2) / 2)
            ea = {**pos, **time_attrs(c["start"], c["end"]), "code": code,
                  "category": c["category"], "title": c["label"]}
            if color and highlight == "marker":
                tb = text_box(d, cxy, lab, f, "mm", pad=4)
                d.rectangle(tb, fill=color)
                rec.add("highlight", tb, **ea)
            if color and luminance(color) < 140:
                fg = (255, 255, 255)
            rec.text(cxy, lab, f, fg, kind="code", anchor="mm", parent="event", **ea)
            rec.add("event", cell, **ea)
            events.append(event_truth(day, c["label"], c["start"], c["end"], resource=resource,
                                      code=code, category=c["category"]))

    # Légende (optionnelle) : les codes présents, avec leur couleur
    if has_legend:
        used = [k for k in codes if any(e["code"] == k for e in events)]
        f_leg = fonts.r(13)
        lx, ly, ly0 = x1, legend_y, legend_y
        rec.text((lx, ly), "Légende :", fonts.b(13), txt, kind="legend", anchor="lm", parent="legend")
        lx += 80
        for k in used:
            c = codes[k]
            desc = (fmt_range(c["start"], c["end"], 0, 0) if c["start"] else c["label"])
            item = f"{k} = {desc}"
            iw = 22 + d.textlength(item, font=f_leg) + 18
            if lx + iw > W - mx:
                lx, ly = x1 + 80, ly + 24
            if c["color"]:
                d.rectangle([lx, ly - 7, lx + 14, ly + 7], fill=c["color"], outline=grid)
            rec.text((lx + 20, ly), item, f_leg, txt, kind="legend", anchor="lm",
                     parent="legend", code=k)
            lx += iw
        rec.add("legend", (x1 - 4, ly0 - 12, W - mx, ly + 12))

    rec.add("calendar", (x1, top if has_title else table_y, x2, table_y2))
    if rng.random() < 0.5:
        rec.text((x1, H - 30), f"Édité le {rng.randint(1, 28):02d}/{d0.month:02d}/{d0.year}",
                 fonts.r(13), (120, 120, 120), kind="footer", anchor="lm")

    truth = {"doc_type": "code_roster", "year": d0.year, "month": d0.month, "events": events}
    return img, rec.elements, truth


# --------------------------------------------------------------------------
# Dégradation "photo prise au téléphone"
# --------------------------------------------------------------------------
def transform_boxes(boxes, M):
    """Applique une matrice 3x3 (ou 2x3) aux 4 coins et reprend l'englobante."""
    M = np.vstack([M, [0, 0, 1]]) if M.shape[0] == 2 else M
    out = []
    for (c, x1, y1, x2, y2) in boxes:
        pts = np.float32([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]])
        tp = pts @ M.T
        tp = tp[:, :2] / tp[:, 2:3]
        out.append((c, tp[:, 0].min(), tp[:, 1].min(), tp[:, 0].max(), tp[:, 1].max()))
    return out


def warp_perspective(img, boxes, rng, strength):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    m = strength
    dst = np.float32([
        [rng.uniform(0, m) * w, rng.uniform(0, m) * h],
        [w - rng.uniform(0, m) * w, rng.uniform(0, m) * h],
        [w - rng.uniform(0, m) * w, h - rng.uniform(0, m) * h],
        [rng.uniform(0, m) * w, h - rng.uniform(0, m) * h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out, transform_boxes(boxes, M)


def add_background(img, boxes, rng, nrng):
    """Pose la feuille sur un fond de bureau (ombre portée) puis rotation globale."""
    h, w = img.shape[:2]
    pad_x, pad_y = int(w * rng.uniform(0.04, 0.13)), int(h * rng.uniform(0.04, 0.13))
    W, H = w + 2 * pad_x, h + 2 * pad_y
    base = rng.choice([(95, 82, 70), (60, 62, 66), (120, 110, 96),
                       (40, 42, 48), (140, 132, 120), (78, 86, 82)])
    canvas = np.full((H, W, 3), base, np.float32) + nrng.normal(0, 8, (H, W, 3))

    sh = np.zeros((H, W), np.float32)
    off = int(min(w, h) * 0.012)
    sh[pad_y + off:pad_y + h + off, pad_x + off:pad_x + w + off] = 1.0
    sh = cv2.GaussianBlur(sh, (0, 0), sigmaX=min(w, h) * 0.012)
    canvas = np.clip(canvas * (1 - 0.45 * sh[..., None]), 0, 255).astype(np.uint8)
    canvas[pad_y:pad_y + h, pad_x:pad_x + w] = img
    boxes = [(c, x1 + pad_x, y1 + pad_y, x2 + pad_x, y2 + pad_y) for (c, x1, y1, x2, y2) in boxes]

    M = cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-3, 3), 1.0)
    canvas = cv2.warpAffine(canvas, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    return canvas, transform_boxes(boxes, M)


def add_lighting(img, rng):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    vign = 1.0 - rng.uniform(0.15, 0.42) * dist / dist.max()
    if rng.random() < 0.45:  # ombre linéaire (main, téléphone)
        ang = rng.uniform(0, math.pi)
        lin = xx * math.cos(ang) + yy * math.sin(ang)
        lin = (lin - lin.min()) / (lin.max() - lin.min())
        vign *= 1.0 - rng.uniform(0.08, 0.28) * lin
    out = img.astype(np.float32) * vign[..., None]
    out *= np.array([rng.uniform(0.94, 1.02), rng.uniform(0.95, 1.02),
                     rng.uniform(0.92, 1.05)], np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_paper_texture(img, rng, nrng):
    h, w = img.shape[:2]
    out = img.astype(np.float32) + nrng.normal(0, rng.uniform(2, 6), (h, w, 1))
    if rng.random() < 0.3:  # pli
        if rng.random() < 0.5:
            pos = rng.uniform(0.3, 0.7) * w
            band = np.exp(-((np.arange(w) - pos) ** 2) / (2 * (w * 0.012) ** 2))
            out *= (1 - 0.12 * band)[None, :, None]
        else:
            pos = rng.uniform(0.3, 0.7) * h
            band = np.exp(-((np.arange(h) - pos) ** 2) / (2 * (h * 0.012) ** 2))
            out *= (1 - 0.12 * band)[:, None, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def motion_blur(img, rng):
    ksz = rng.choice([7, 9])
    kern = np.zeros((ksz, ksz), np.float32)
    a = rng.uniform(0, math.pi)
    for t in range(ksz):
        x = int(ksz / 2 + (t - ksz / 2) * math.cos(a))
        y = int(ksz / 2 + (t - ksz / 2) * math.sin(a))
        if 0 <= x < ksz and 0 <= y < ksz:
            kern[y, x] = 1
    return cv2.filter2D(img, -1, kern / kern.sum())


def photo_degrade(img_pil, boxes, rng, nrng):
    img = np.array(img_pil)[:, :, ::-1].copy()  # RGB -> BGR
    img = add_paper_texture(img, rng, nrng)
    img, boxes = warp_perspective(img, boxes, rng, rng.uniform(0.01, 0.055))
    img, boxes = add_background(img, boxes, rng, nrng)
    img = add_lighting(img, rng)

    if rng.random() < 0.6:
        k = rng.choice([3, 3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.18:
        img = motion_blur(img, rng)
    img = np.clip(img.astype(np.float32) + nrng.normal(0, rng.uniform(1.5, 7), img.shape),
                  0, 255).astype(np.uint8)

    scale = rng.uniform(0.5, 0.9)
    nh, nw = int(img.shape[0] * scale), int(img.shape[1] * scale)
    sx, sy = nw / img.shape[1], nh / img.shape[0]
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    boxes = [(c, x1 * sx, y1 * sy, x2 * sx, y2 * sy) for (c, x1, y1, x2, y2) in boxes]
    return img, boxes


# --------------------------------------------------------------------------
# Sortie YOLO
# --------------------------------------------------------------------------
def yolo_lines(boxes, w, h, min_px=4):
    lines, counts = [], [0] * len(CLASSES)
    for (cls, x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < min_px or y2 - y1 < min_px:
            continue
        lines.append(f"{int(cls)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        counts[int(cls)] += 1
    return lines, counts


def draw_debug(img, boxes):
    colors = [(60, 60, 255), (200, 60, 200), (250, 170, 40), (60, 200, 60), (40, 120, 250)]
    out = img.copy()
    for (cls, x1, y1, x2, y2) in boxes:
        c = colors[int(cls)]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        cv2.putText(out, CLASSES[int(cls)], (int(x1) + 3, int(y1) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="nombre d'images à générer")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--test-split", type=float, default=0.0)
    ap.add_argument("--month-ratio", type=float, default=0.5,
                    help="part de grilles mensuelles")
    ap.add_argument("--roster-ratio", type=float, default=0.0,
                    help="part de plannings à codes (le reste = plannings hebdo)")
    ap.add_argument("--roster-codes", type=Path, default=DEFAULT_ROSTER_CODES,
                    help="fichier YAML des codes horaires (plannings à codes)")
    ap.add_argument("--roster-days", type=parse_days_range, default=None,
                    help="jours par planning à codes : 15 ou 10-20 (défaut : 7 à 31)")
    ap.add_argument("--clean-ratio", type=float, default=0.1,
                    help="part d'images sans dégradation photo (scan / export propre)")
    ap.add_argument("--clean", action="store_true", help="aucune dégradation (équivaut à --clean-ratio 1)")
    ap.add_argument("--debug", type=int, default=0, help="nb d'images de contrôle avec boîtes dessinées")
    add_output_args(ap, "data/datasets/v2")
    args = ap.parse_args()
    if args.clean:
        args.clean_ratio = 1.0
    if args.month_ratio + args.roster_ratio > 1:
        ap.error("--month-ratio + --roster-ratio doit être <= 1")
    codes = load_roster_codes(args.roster_codes) if args.roster_ratio > 0 else None

    plan = prepare_output(args, CLASSES)  # vérifie --out / --mode, n'écrit rien
    make_dirs(plan, args.debug)

    manifest, totals = [], [0] * len(CLASSES)
    for k, split in enumerate(plan.splits):
        i = plan.start + k  # index global : image i reproductible indépendamment des autres
        rng = random.Random(f"{args.seed}-{i}")
        nrng = np.random.default_rng(args.seed * 1_000_003 + i)

        u = rng.random()
        if u < args.month_ratio:
            img_pil, boxes, truth = make_month_grid(rng)
        elif u < args.month_ratio + args.roster_ratio:
            img_pil, elements, truth = make_code_roster(rng, codes, args.roster_days)
            boxes = [(ROLE_TO_CLASS[e["role"]], *e["box"]) for e in elements
                     if e["role"] in ROLE_TO_CLASS]
        else:
            img_pil, boxes, truth = make_week_planning(rng)
        clean = rng.random() < args.clean_ratio
        if clean:
            img = np.array(img_pil)[:, :, ::-1].copy()
        else:
            img, boxes = photo_degrade(img_pil, boxes, rng, nrng)

        h, w = img.shape[:2]
        name = plan.name(i)
        cv2.imwrite(str(plan.image_path(split, name)), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(60, 92)])
        lines, counts = yolo_lines(boxes, w, h)
        plan.label_path(split, name).write_text("\n".join(lines) + "\n")
        totals = [a + b for a, b in zip(totals, counts)]

        if k < args.debug:
            cv2.imwrite(str(args.out / "debug" / f"{name}_boxes.jpg"), draw_debug(img, boxes))

        manifest.append(manifest_entry(plan, i, split, name, clean=clean, size=[w, h],
                                       boxes=dict(zip(CLASSES, counts)), **truth))

    finalize_output(plan, manifest, {
        "roster_codes.yaml": args.roster_codes.read_text() if codes else None,
    })
    for name, n in zip(CLASSES, totals):
        print(f"  {name:15s} {n:6d}")


if __name__ == "__main__":
    main()
