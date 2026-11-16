#!/usr/bin/env python3
"""
Générateur CalEvent avec labélisation personnalisable.

Même rendu que generate_calendar_dataset.py (même seed => mêmes images), mais au
lieu des 5 classes codées en dur, chaque élément dessiné est enregistré avec un
*rôle* et des *attributs*. Un fichier YAML (un par étudiant) décide quelles
boîtes deviennent quelles classes :

    classes:
      evenement: [event]                       # forme courte : liste de rôles
      jour:                                    # forme longue
        roles: [date_cell, event]
        group_by: [date]                       # 1 boîte par jour (union)
      heure:
        roles: [text]
        where: {kind: event_time}              # filtre sur les attributs

Avec group_by, les éléments sélectionnés qui partagent les mêmes valeurs
d'attributs sont fusionnés en un seul rectangle englobant ; ceux qui n'ont pas
l'attribut sont ignorés. Liste des rôles et attributs : --list-roles.

Usage :
    uv run python src/data/generate_calendar_custom.py --list-roles
    uv run python src/data/generate_calendar_custom.py --labels configs/labels/default.yaml --n 300 --out data/datasets/v2
    uv run python src/data/generate_calendar_custom.py --labels configs/labels/par_jour.yaml --n 20 --debug 10 --out /tmp/preview
    uv run python src/data/generate_calendar_custom.py --labels configs/labels/codes.yaml --roster-ratio 1 --n 20 --debug 10 --out /tmp/preview

Ajout à un dataset existant (--mode merge) : la numérotation reprend après la
dernière image <prefix>_NNNN (tous splits), manifest.json est complété et les
classes de data.yaml doivent être exactement celles du fichier de labels, dans
le même ordre. Sans --mode, le script refuse d'écrire dans un dossier non vide.

    # 2e lot dans le même dataset (même seed => images différentes du 1er lot)
    uv run python src/data/generate_calendar_custom.py --labels configs/labels/default.yaml --n 100 --out data/datasets/v2 --mode merge
    # images synthétiques ajoutées à un export CVAT (Ultralytics YOLO)
    uv run python src/data/generate_calendar_custom.py --labels mes_labels.yaml --n 200 --prefix synth --mode merge --out dataset/export_cvat
"""

import argparse
import calendar
import datetime as dt
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

from generate_calendar_dataset import (
    AGENDA_TITLES, AVAILABLE_FAMILIES, CRENEAUX, DEFAULT_ROSTER_CODES, ENTREPRISES,
    EVENT_COLORS, JOURS, JOURS_ABBR, MOIS, MOIS_ABBR, NOMS, PALETTES, POSTES, SITES,
    Fonts, Recorder, date_attrs, draw_banner, event_truth, fit_text, fmt_range,
    fmt_time, load_roster_codes, make_code_roster, mix, parse_days_range,
    photo_degrade, pick_time, text_box, time_attrs,
)
from dataset_output import (
    add_output_args, finalize_output, make_dirs, manifest_entry, prepare_output,
)

# --------------------------------------------------------------------------
# Vocabulaire disponible pour les fichiers de labels
# --------------------------------------------------------------------------
ROLES = {
    "calendar":        "le calendrier entier (bandeau du mois + grille)",
    "month_header":    "le titre du mois / de la semaine",
    "day_header_row":  "la rangée (ou colonne si transposé) des noms de jours",
    "day_name_cell":   "une case nom de jour (Lun, MARDI…)",
    "date_cell":       "une case portant une date (case du jour en grille mensuelle)",
    "out_month_cell":  "case grisée hors du mois (grille mensuelle)",
    "event":           "un événement (pavé, ligne de texte ou cellule remplie)",
    "empty_cell":      "case vide de la grille du planning hebdo",
    "resource_header": "case nom de personne / poste (planning hebdo, planning à codes)",
    "corner":          "case(s) en haut à gauche du tableau",
    "highlight":       "surlignage coloré autour d'un code (planning à codes, style marqueur)",
    "legend":          "bloc légende des codes (planning à codes)",
    "text":            "un bloc de texte (voir l'attribut kind)",
}

ATTRS = {
    "doc_type": "month_grid | week_planning | code_roster",
    "layout":   "month | week | week_transposed | roster",
    "row":      "index de ligne dans la grille (semaine en grille mensuelle)",
    "col":      "index de colonne dans la grille",
    "date":     "date ISO, ex. '2026-11-03' (identifie un jour)",
    "day":      "numéro du jour dans le mois",
    "weekday":  "0 = lundi … 6 = dimanche",
    "weekend":  "true pour samedi / dimanche",
    "week":     "numéro de semaine ISO (en-têtes du planning à codes)",
    "kind":     "pour text : title, subtitle, footer, day_name, date, resource, "
                "corner, event_time, event_title, event_site, code, week, month, legend",
    "parent":   "pour text : rôle de la case qui le contient",
    "resource": "personne ou poste (planning hebdo)",
    "title":    "intitulé de l'événement",
    "site":     "site (planning hebdo)",
    "start":    "heure de début, ex. '08:30' (null si aucune)",
    "end":      "heure de fin (null si aucune)",
    "has_time": "true si l'événement affiche une heure",
    "night":    "true si le créneau passe minuit",
    "slot":     "rang de l'événement dans sa case (grille mensuelle, 0 = premier)",
    "code":     "code du planning à codes, ex. 'G03', 'RH'",
    "category": "catégorie du code : work | training | rest | leave",
}


# --------------------------------------------------------------------------
# Type 1 : grille mensuelle. Copie de make_month_grid() qui enregistre des
# éléments ; l'ordre des appels au rng est strictement identique.
# --------------------------------------------------------------------------
def make_month_grid(rng):
    W = rng.choice([1200, 1400, 1600])
    H = rng.choice([1000, 1100, 1200])
    bg, hdr_bg, hdr_fg, grid, txt, accent = rng.choice(PALETTES)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    fonts = Fonts(rng.choice(AVAILABLE_FAMILIES))
    rec = Recorder(d, doc_type="month_grid", layout="month")

    year, month = rng.randint(2024, 2027), rng.randint(1, 12)
    first_wd = rng.choice([0, 0, 0, 6])
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

    events = []
    rec.add("calendar", (x1, top, x2, gy2))

    title = f"{MOIS[month - 1].upper()} {year}" if rng.random() < 0.6 else f"{MOIS[month - 1].capitalize()} {year}"
    rec.add("month_header", draw_banner(d, rng, fonts, title, x1, x2, top, banner_h, hdr_bg, hdr_fg, txt))

    wd_y1 = top + banner_h
    abbr = rng.random() < 0.5
    dark_hdr = rng.random() < 0.5
    for c in range(7):
        cx1 = x1 + c * col_w
        wd = (first_wd + c) % 7
        d.rectangle([cx1, wd_y1, cx1 + col_w, gy1], fill=hdr_bg if dark_hdr else accent, outline=grid)
        label, f = fit_text(d, JOURS_ABBR[wd] if abbr else JOURS[wd], fonts.b, 20, col_w - 10)
        rec.text((cx1 + col_w / 2, wd_y1 + wd_h / 2), label, f, hdr_fg if dark_hdr else txt,
                 kind="day_name", anchor="mm", parent="day_name_cell", col=c, weekday=wd)
        rec.add("day_name_cell", (cx1, wd_y1, cx1 + col_w, gy1), col=c, weekday=wd, weekend=wd >= 5)
    rec.add("day_header_row", (x1, wd_y1, x2, gy1))

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
            pos = {"row": r, "col": c}
            in_month = date.month == month
            if not in_month:
                d.rectangle([cx1, cy1, cx2, cy2], fill=mix(bg, (200, 200, 200), 0.35), outline=grid)
                rec.add("out_month_cell", (cx1, cy1, cx2, cy2), **pos, **date_attrs(date))
                continue
            da = {**pos, **date_attrs(date)}
            weekend = date.weekday() >= 5
            d.rectangle([cx1, cy1, cx2, cy2], fill=mix(bg, accent, 0.5) if weekend else bg, outline=grid)
            nxy = (cx2 - 8, cy1 + num_h / 2) if num_right else (cx1 + 8, cy1 + num_h / 2)
            rec.text(nxy, str(date.day), fonts.b(num_sz), txt, kind="date",
                     anchor="rm" if num_right else "lm", parent="date_cell", **da)
            rec.add("date_cell", (cx1, cy1, cx2, cy2), **da)

            n_ev = 0
            if max_ev and rng.random() < ev_density:
                n_ev = min(max_ev, rng.choices([1, 2, 3], [6, 3, 1])[0])
            for k in range(n_ev):
                ey = cy1 + num_h + k * (ev_h + 3)
                t1, t2 = pick_time(rng) if rng.random() > 0.2 else (None, None)
                name = rng.choice(AGENDA_TITLES)
                t_txt = fmt_time(t1, tstyle) if t1 else None
                label = f"{t_txt} {name}" if t1 else name
                color = rng.choice(EVENT_COLORS)
                ea = {**da, **time_attrs(t1, t2), "title": name, "slot": k}
                if ev_style == "strip":
                    d.rounded_rectangle([cx1 + 4, ey, cx2 - 4, ey + ev_h], radius=4, fill=color)
                    label, f = fit_text(d, label, fonts.r, ev_sz, col_w - 18)
                    txy, tfill = (cx1 + 9, ey + ev_h / 2), (20, 20, 20)
                    ebox = (cx1 + 4, ey, cx2 - 4, ey + ev_h)
                else:
                    label, f = fit_text(d, label, fonts.r, ev_sz, col_w - 28)
                    d.ellipse([cx1 + 7, ey + ev_h / 2 - 4, cx1 + 15, ey + ev_h / 2 + 4], fill=mix(color, (0, 0, 0), 0.35))
                    txy, tfill = (cx1 + 19, ey + ev_h / 2), txt
                    tb = text_box(d, txy, label, f, "lm", pad=3)
                    ebox = (cx1 + 4, tb[1], tb[2], tb[3])
                d.text(txy, label, font=f, fill=tfill, anchor="lm")
                # Boîtes texte séparées pour l'heure et l'intitulé
                if t1 and label.startswith(t_txt + " "):
                    rec.add("text", text_box(d, txy, t_txt, f, "lm", pad=2),
                            kind="event_time", parent="event", **ea)
                    rest_xy = (txy[0] + d.textlength(t_txt + " ", font=f), txy[1])
                    rest = label[len(t_txt) + 1:]
                    if rest:
                        rec.add("text", text_box(d, rest_xy, rest, f, "lm", pad=2),
                                kind="event_title", parent="event", **ea)
                else:
                    rec.add("text", text_box(d, txy, label, f, "lm", pad=2),
                            kind="event_title", parent="event", **ea)
                rec.add("event", ebox, **ea)
                events.append(event_truth(date, name, t1, t2))

    fy = H - rng.randint(25, 40)
    foot = rng.choice([
        "Calendrier familial", "Planning de l'équipe — usage interne",
        f"Édité le {rng.randint(1, 28):02d}/{month:02d}/{year}"])
    rec.text((x1, fy), foot, fonts.r(13), (120, 120, 120), kind="footer", anchor="lm")

    truth = {"doc_type": "month_grid", "year": year, "month": month, "events": events}
    return img, rec.elements, truth


# --------------------------------------------------------------------------
# Type 2 : planning hebdomadaire. Copie de make_week_planning(), même ordre
# des appels au rng.
# --------------------------------------------------------------------------
def make_week_planning(rng):
    W = rng.choice([1400, 1600, 1700])
    H = rng.choice([900, 1000, 1100])
    bg, hdr_bg, hdr_fg, grid, txt, accent = rng.choice(PALETTES)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    fonts = Fonts(rng.choice(AVAILABLE_FAMILIES))

    d0 = dt.date(rng.randint(2024, 2027), 1, 1) + dt.timedelta(days=rng.randint(0, 364))
    monday = d0 - dt.timedelta(days=d0.weekday())
    n_days = rng.choice([5, 6, 6, 7])
    days = [monday + dt.timedelta(i) for i in range(n_days)]
    transposed = rng.random() < 0.25
    abbr = rng.random() < 0.5
    date_fmt = rng.randint(0, 2)
    tstyle, sep_style = rng.randint(0, 2), rng.randint(0, 3)
    color_cells = rng.random() < 0.6
    use_names = rng.random() < 0.5
    hdr2 = mix(hdr_bg, (255, 255, 255), 0.25)
    rec = Recorder(d, doc_type="week_planning",
                   layout="week_transposed" if transposed else "week")

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
    rec.text((x1, top), f"{ent}  —  Personnel intérimaire", fonts.r(15), (90, 90, 90),
             kind="subtitle", anchor="lm")

    iso = monday.isocalendar()
    title = rng.choice([
        f"SEMAINE {iso.week:02d} — {MOIS[monday.month - 1].upper()} {monday.year}",
        f"PLANNING SEMAINE {iso.week:02d}",
        f"Planning du {monday.day:02d} {MOIS[monday.month - 1]} {monday.year}",
    ])
    banner_y, banner_h = top + 22, rng.randint(52, 70)
    banner_y2 = banner_y + banner_h
    events = []
    rec.add("month_header", draw_banner(d, rng, fonts, title, x1, x2, banner_y, banner_h, hdr_bg, hdr_fg, txt))

    def draw_event(cx1, cy1, cx2, cy2, day, resource, slot, site, pos):
        fill = rng.choice(EVENT_COLORS) if color_cells else None
        d.rectangle([cx1, cy1, cx2, cy2], fill=fill or bg, outline=grid)
        t_txt = fmt_range(slot[0], slot[1], tstyle, sep_style)
        l1, f1 = fit_text(d, t_txt, fonts.r, 17, cx2 - cx1 - 10)
        l2, f2 = fit_text(d, site, fonts.r, 14, cx2 - cx1 - 10)
        mid = (cy1 + cy2) / 2
        ea = {**pos, **date_attrs(day), **time_attrs(*slot),
              "resource": resource, "site": site, "title": site}
        rec.text(((cx1 + cx2) / 2, mid - 9), l1, f1, txt, kind="event_time",
                 anchor="mm", parent="event", **ea)
        rec.text(((cx1 + cx2) / 2, mid + 11), l2, f2, (70, 70, 70), kind="event_site",
                 anchor="mm", parent="event", **ea)
        rec.add("event", (cx1, cy1, cx2, cy2), **ea)
        events.append(event_truth(day, site, slot[0], slot[1], resource=resource))

    def empty_cell(cx1, cy1, cx2, cy2, day, resource, pos):
        d.rectangle([cx1, cy1, cx2, cy2], fill=bg, outline=grid)
        rec.add("empty_cell", (cx1, cy1, cx2, cy2), **pos, **date_attrs(day), resource=resource)

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
        rec.add("corner", (x1, banner_y2, bx1, y_dt))
        rec.add("corner", (x1, y_dt, bx1, y_body))
        if rng.random() < 0.6:
            rec.text((x1 + 8, banner_y2 + wd_h / 2), "Équipe" if use_names else "Poste",
                     fonts.b(16), hdr_fg, kind="corner", anchor="lm", parent="corner")
        for j, day in enumerate(days):
            cx1 = bx1 + j * col_w
            da = {"col": j, **date_attrs(day)}
            d.rectangle([cx1, banner_y2, cx1 + col_w, y_dt], fill=hdr2, outline=grid)
            lab, f = fit_text(d, day_label(day), fonts.b, 19, col_w - 8)
            rec.text((cx1 + col_w / 2, banner_y2 + wd_h / 2), lab, f, hdr_fg, kind="day_name",
                     anchor="mm", parent="day_name_cell", **da)
            rec.add("day_name_cell", (cx1, banner_y2, cx1 + col_w, y_dt), **da)
            d.rectangle([cx1, y_dt, cx1 + col_w, y_body], fill=accent, outline=grid)
            lab, f = fit_text(d, date_label(day), fonts.r, 17, col_w - 8)
            rec.text((cx1 + col_w / 2, y_dt + dt_h / 2), lab, f, txt, kind="date",
                     anchor="mm", parent="date_cell", **da)
            rec.add("date_cell", (cx1, y_dt, cx1 + col_w, y_body), **da)
        rec.add("day_header_row", (bx1, banner_y2, x2, y_dt))

        p_fill = rng.uniform(0.5, 0.8)
        for i in range(n_rows):
            ry1 = y_body + i * row_h
            ry2 = ry1 + row_h
            d.rectangle([x1, ry1, bx1, ry2], fill=accent if i % 2 == 0 else bg, outline=grid)
            lab, f = fit_text(d, res[i], fonts.r, 16, label_w - 14)
            rec.text((x1 + 8, (ry1 + ry2) / 2), lab, f, txt, kind="resource", anchor="lm",
                     parent="resource_header", row=i, resource=res[i])
            rec.add("resource_header", (x1, ry1, bx1, ry2), row=i, resource=res[i])
            for j, day in enumerate(days):
                cx1 = bx1 + j * col_w
                pos = {"row": i, "col": j}
                if rng.random() < p_fill:
                    draw_event(cx1, ry1, cx1 + col_w, ry2, day, res[i],
                               rng.choice(CRENEAUX), rng.choice(SITES), pos)
                else:
                    empty_cell(cx1, ry1, cx1 + col_w, ry2, day, res[i], pos)
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
        rec.add("corner", (x1, banner_y2, bx1, y_body))
        rec.text((x1 + 8, banner_y2 + hdr_h / 2), "Jour", fonts.b(17), hdr_fg,
                 kind="corner", anchor="lm", parent="corner")
        for j in range(n_cols):
            cx1 = bx1 + j * col_w
            d.rectangle([cx1, banner_y2, cx1 + col_w, y_body], fill=hdr2, outline=grid)
            lab, f = fit_text(d, res[j], fonts.b, 17, col_w - 10)
            rec.text((cx1 + col_w / 2, banner_y2 + hdr_h / 2), lab, f, hdr_fg, kind="resource",
                     anchor="mm", parent="resource_header", col=j, resource=res[j])
            rec.add("resource_header", (cx1, banner_y2, cx1 + col_w, y_body), col=j, resource=res[j])

        p_fill = rng.uniform(0.5, 0.8)
        for i, day in enumerate(days):
            ry1 = y_body + i * row_h
            ry2 = ry1 + row_h
            da = {"row": i, **date_attrs(day)}
            d.rectangle([x1, ry1, bx_wd2, ry2], fill=accent, outline=grid)
            lab, f = fit_text(d, day_label(day), fonts.b, 18, wd_w - 12)
            rec.text((x1 + 8, (ry1 + ry2) / 2), lab, f, txt, kind="day_name", anchor="lm",
                     parent="day_name_cell", **da)
            rec.add("day_name_cell", (x1, ry1, bx_wd2, ry2), **da)
            d.rectangle([bx_wd2, ry1, bx1, ry2], fill=accent, outline=grid)
            lab, f = fit_text(d, date_label(day), fonts.r, 17, dt_w - 8)
            rec.text(((bx_wd2 + bx1) / 2, (ry1 + ry2) / 2), lab, f, txt, kind="date",
                     anchor="mm", parent="date_cell", **da)
            rec.add("date_cell", (bx_wd2, ry1, bx1, ry2), **da)
            for j in range(n_cols):
                cx1 = bx1 + j * col_w
                pos = {"row": i, "col": j}
                if rng.random() < p_fill:
                    draw_event(cx1, ry1, cx1 + col_w, ry2, day, res[j],
                               rng.choice(CRENEAUX), rng.choice(SITES), pos)
                else:
                    empty_cell(cx1, ry1, cx1 + col_w, ry2, day, res[j], pos)
        rec.add("day_header_row", (x1, y_body, bx_wd2, y_body + n_days * row_h))
        gy2 = y_body + n_days * row_h

    rec.add("calendar", (x1, banner_y, x2, gy2))
    fy = H - rng.randint(35, 55)
    foot = rng.choice([
        f"Édité le {rng.randint(1, 28):02d}/{monday.month:02d} — Document interne",
        f"Contact agence : 04.7{rng.randint(0, 9)}.{rng.randint(10, 99)}.{rng.randint(10, 99)}.{rng.randint(10, 99)}",
        "Toute modification doit être signalée avant 16h la veille."])
    rec.text((x1, fy), foot, fonts.r(14), (120, 120, 120), kind="footer", anchor="lm")

    truth = {"doc_type": "week_planning", "year": monday.year, "month": monday.month,
             "events": events}
    return img, rec.elements, truth


# --------------------------------------------------------------------------
# Fichier de labels -> boîtes YOLO
# --------------------------------------------------------------------------
def load_label_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    if not raw or not raw.get("classes"):
        raise SystemExit(f"{path} : clé 'classes' manquante ou vide")

    classes = []
    for name, spec in raw["classes"].items():
        if isinstance(spec, (list, str)):
            spec = {"roles": spec}
        roles = spec.get("roles")
        roles = [roles] if isinstance(roles, str) else roles
        if not roles:
            raise SystemExit(f"{path} : classe '{name}' sans 'roles'")
        for r in roles:
            if r not in ROLES:
                raise SystemExit(f"{path} : rôle inconnu '{r}' (classe '{name}'). "
                                 f"Rôles : {', '.join(ROLES)}")
        where = spec.get("where") or {}
        group_by = spec.get("group_by") or []
        group_by = [group_by] if isinstance(group_by, str) else group_by
        for a in list(where) + group_by:
            if a not in ATTRS:
                raise SystemExit(f"{path} : attribut inconnu '{a}' (classe '{name}'). "
                                 f"Attributs : {', '.join(ATTRS)}")
        classes.append({"name": str(name), "roles": set(roles),
                        "where": where, "group_by": group_by})
    return classes


def matches(el, where):
    for k, v in where.items():
        if el.get(k) not in (v if isinstance(v, list) else [v]):
            return False
    return True


def elements_to_boxes(elements, classes):
    """[(cls_id, x1, y1, x2, y2)] en coordonnées de l'image propre."""
    boxes = []
    for cls_id, c in enumerate(classes):
        sel = [e for e in elements if e["role"] in c["roles"] and matches(e, c["where"])]
        if not c["group_by"]:
            boxes += [(cls_id, *e["box"]) for e in sel]
            continue
        groups = {}
        for e in sel:
            key = tuple(e.get(a) for a in c["group_by"])
            if None not in key:
                groups.setdefault(key, []).append(e["box"])
        for bs in groups.values():
            a = np.array(bs)
            boxes.append((cls_id, a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()))
    return boxes


def yolo_lines(boxes, w, h, n_classes, min_px=4):
    lines, counts = [], [0] * n_classes
    for (cls, x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < min_px or y2 - y1 < min_px:
            continue
        lines.append(f"{int(cls)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        counts[int(cls)] += 1
    return lines, counts


def draw_debug(img, boxes, names):
    out = img.copy()
    for (cls, x1, y1, x2, y2) in boxes:
        hue = int(cls) * 47 % 180
        c = cv2.cvtColor(np.uint8([[[hue, 220, 230]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        cv2.putText(out, names[int(cls)], (int(x1) + 3, int(y1) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=Path("configs/labels/default.yaml"),
                    help="fichier YAML définissant les classes")
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
    ap.add_argument("--list-roles", action="store_true",
                    help="affiche les rôles et attributs disponibles puis quitte")
    add_output_args(ap, "data/datasets/custom")
    args = ap.parse_args()

    if args.list_roles:
        print("Rôles :")
        for k, v in ROLES.items():
            print(f"  {k:<16} {v}")
        print("\nAttributs (utilisables dans where / group_by) :")
        for k, v in ATTRS.items():
            print(f"  {k:<16} {v}")
        return
    if args.clean:
        args.clean_ratio = 1.0
    if args.month_ratio + args.roster_ratio > 1:
        ap.error("--month-ratio + --roster-ratio doit être <= 1")
    codes = load_roster_codes(args.roster_codes) if args.roster_ratio > 0 else None

    classes = load_label_config(args.labels)
    names = [c["name"] for c in classes]

    plan = prepare_output(args, names)  # vérifie --out / --mode, n'écrit rien
    make_dirs(plan, args.debug)

    manifest, totals = [], [0] * len(names)
    for k, split in enumerate(plan.splits):
        i = plan.start + k  # index global : graine propre à chaque image
        rng = random.Random(f"{args.seed}-{i}")
        nrng = np.random.default_rng(args.seed * 1_000_003 + i)

        u = rng.random()
        if u < args.month_ratio:
            img_pil, elements, truth = make_month_grid(rng)
        elif u < args.month_ratio + args.roster_ratio:
            img_pil, elements, truth = make_code_roster(rng, codes, args.roster_days)
        else:
            img_pil, elements, truth = make_week_planning(rng)
        boxes = elements_to_boxes(elements, classes)
        clean = rng.random() < args.clean_ratio
        if clean:
            img = np.array(img_pil)[:, :, ::-1].copy()
        else:
            img, boxes = photo_degrade(img_pil, boxes, rng, nrng)

        h, w = img.shape[:2]
        name = plan.name(i)
        cv2.imwrite(str(plan.image_path(split, name)), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(60, 92)])
        lines, counts = yolo_lines(boxes, w, h, len(names))
        plan.label_path(split, name).write_text("\n".join(lines) + "\n")
        totals = [a + b for a, b in zip(totals, counts)]

        if k < args.debug:
            cv2.imwrite(str(args.out / "debug" / f"{name}_boxes.jpg"), draw_debug(img, boxes, names))

        manifest.append(manifest_entry(plan, i, split, name, clean=clean, size=[w, h],
                                       boxes=dict(zip(names, counts)), **truth))

    finalize_output(plan, manifest, {
        "labels_config.yaml": args.labels.read_text(),
        "roster_codes.yaml": args.roster_codes.read_text() if codes else None,
    })
    print(f"  labels : {args.labels}")
    for n, k in zip(names, totals):
        print(f"  {n:20s} {k:6d}")


if __name__ == "__main__":
    main()
