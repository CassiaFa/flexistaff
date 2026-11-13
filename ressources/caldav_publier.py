"""Publication d'événements dans un agenda CalDAV (Radicale) : module FOURNI, jour 3.

Vous n'avez pas à le modifier : votre travail est le *parsing* (texte OCR -> événements).
Ce module prend vos événements et les publie dans l'agenda partagé ``calevent`` de votre squad.

Dépendances : ``pip install caldav icalendar`` (Python 3.10+, fonctionne dans Colab).

Format d'un événement (le schéma du cours)::

    {"title": "Réunion équipe", "date": "2026-03-12",
     "start_time": "14:00", "end_time": "18:00",   # ou None
     "location": "Salle 2",                          # ou None
     "confidence": 0.92,
     # champs optionnels (planning à codes) :
     "person": "Roux Inès",     # la personne : entre dans l'UID, ajoutée au titre si absente
     "all_day": False,          # True : journée entière (repos, congé), heures ignorées
     "overnight": False}        # True : poste de nuit, fin le lendemain (20:00 -> 08:00)

Règles appliquées :

- heures en ``Europe/Paris`` (TZID + VTIMEZONE) ;
- UID stable, calculé depuis la date, les heures et le titre : envoyer deux fois le même
  créneau ne crée qu'un seul événement (statut « déjà présent ») ;
- sans heure de début : événement « toute la journée » ; sans heure de fin : durée de 1 h ;
  fin <= début (erreur d'OCR) : durée de 1 h ;
- ``person`` (si présent) entre dans l'UID (deux personnes, même poste, même jour = deux événements)
  et est ajoutée au titre (« Matin — Roux Inès ») seulement si le titre ne la contient pas déjà ;
- ``all_day: True`` : journée entière (DTSTART/DTEND en dates, lendemain exclusif), heures ignorées, y compris dans l'UID ;
- ``overnight: True`` et fin <= début : la fin tombe le lendemain (sinon : durée de 1 h) ;
- ``location`` -> champ LOCATION, ``confidence`` -> DESCRIPTION.

Exemple d'utilisation::

    from caldav_publier import publier, charger_identifiants

    evenements = [{"title": "Équipe A", "date": "2026-03-12", "start_time": "14:00",
                   "end_time": "18:00", "location": None, "confidence": 0.9}]

    # 1. Essai à blanc : construit les événements, ne contacte aucun serveur
    print(publier(evenements, url="", utilisateur="", mot_de_passe="", dry_run=True))

    # 2. Publication réelle, avec le fichier identifiants-<login>.json du portail
    ident = charger_identifiants("identifiants-<login>.json")
    for r in publier(evenements, **ident):
        print(r["statut"], r["date"], r["titre"])
"""
from __future__ import annotations

import json
import re
import unicodedata
import uuid
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from icalendar import Calendar, Event as ICalEvent

AGENDA_PAR_DEFAUT = "calevent"
DUREE_PAR_DEFAUT = timedelta(hours=1)
TZ = ZoneInfo("Europe/Paris")
# Espace de noms des UID déterministes (même que l'application CalEvent)
UID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "calevent.isen-ouest.fr")

# VTIMEZONE statique de Europe/Paris (règles UE), pour que le TZID soit compris de tous les clients
VTIMEZONE_PARIS = """BEGIN:VTIMEZONE
TZID:Europe/Paris
BEGIN:DAYLIGHT
TZOFFSETFROM:+0100
TZOFFSETTO:+0200
TZNAME:CEST
DTSTART:19700329T020000
RRULE:FREQ=YEARLY;BYMONTH=3;BYDAY=-1SU
END:DAYLIGHT
BEGIN:STANDARD
TZOFFSETFROM:+0200
TZOFFSETTO:+0100
TZNAME:CET
DTSTART:19701025T030000
RRULE:FREQ=YEARLY;BYMONTH=10;BYDAY=-1SU
END:STANDARD
END:VTIMEZONE
"""


def charger_identifiants(chemin_json: str) -> dict:
    """Lit la partie CalDAV du fichier d'identifiants téléchargé depuis le portail.

    Le fichier contient une clé ``caldav`` avec ``url``, ``username``, ``password``
    (et ``web_url``, l'interface web de Radicale). Le résultat se passe directement à
    :func:`publier` : ``publier(evenements, **charger_identifiants(chemin))``.

    Les identifiants ne sont jamais écrits en dur dans le code ni affichés.
    """
    with open(chemin_json, encoding="utf-8") as f:
        donnees = json.load(f)
    if "caldav" not in donnees or not donnees["caldav"]:
        raise KeyError("Pas de clé « caldav » dans le fichier : votre équipe a-t-elle été choisie sur le portail ?")
    c = donnees["caldav"]
    return {"url": c["url"], "utilisateur": c["username"], "mot_de_passe": c["password"]}


def _normaliser_titre(titre: str) -> str:
    """Titre sans accents, en minuscules, espaces réduits (pour l'UID)."""
    texte = unicodedata.normalize("NFKD", titre or "")
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", texte).strip().lower()


def _journee_entiere(evenement: dict) -> bool:
    """Journée entière : all_day explicite, ou pas d'heure de début."""
    return bool(evenement.get("all_day")) or not evenement.get("start_time")


def _titre_complet(evenement: dict) -> str:
    """Titre, suivi de la personne si elle n'y figure pas déjà."""
    titre = evenement.get("title") or "Événement"
    personne = evenement.get("person")
    if personne and _normaliser_titre(personne) not in _normaliser_titre(titre):
        return f"{titre} — {personne}"
    return titre


def calculer_uid(evenement: dict) -> str:
    """UID déterministe : même date, mêmes heures, même titre -> même UID."""
    journee = _journee_entiere(evenement)
    cle = [
        evenement.get("date") or "",
        "" if journee else evenement.get("start_time") or "",
        "" if journee else evenement.get("end_time") or "",
        _normaliser_titre(evenement.get("title") or ""),
    ]
    if evenement.get("person"):  # sans personne : UID inchangé
        cle.append(_normaliser_titre(evenement["person"]))
    cle = "|".join(cle)
    return f"{uuid.uuid5(UID_NAMESPACE, cle)}@calevent"


def construire_ical(evenement: dict, uid: str | None = None) -> bytes:
    """Construit le contenu iCalendar (VCALENDAR + VEVENT) d'un événement.

    Lève ``ValueError`` si la date est absente ou mal formée.
    """
    if not evenement.get("date"):
        raise ValueError("date manquante : impossible de dater l'événement")
    jour = date.fromisoformat(evenement["date"])
    uid = uid or calculer_uid(evenement)

    evt = ICalEvent()
    evt.add("uid", uid)
    evt.add("summary", _titre_complet(evenement))
    evt.add("dtstamp", datetime.now(tz=ZoneInfo("UTC")))
    if evenement.get("location"):
        evt.add("location", evenement["location"])
    if evenement.get("confidence") is not None:
        evt.add("description", f"Confiance du parsing : {float(evenement['confidence']):.2f}")

    journee = _journee_entiere(evenement)
    debut_txt = None if journee else evenement.get("start_time")
    if debut_txt:
        h, m = map(int, debut_txt.split(":"))
        debut = datetime(jour.year, jour.month, jour.day, h, m, tzinfo=TZ)
        fin = debut + DUREE_PAR_DEFAUT
        if evenement.get("end_time"):
            h2, m2 = map(int, evenement["end_time"].split(":"))
            fin_lue = datetime(jour.year, jour.month, jour.day, h2, m2, tzinfo=TZ)
            if fin_lue > debut:
                fin = fin_lue
            elif evenement.get("overnight"):  # poste de nuit : fin le lendemain
                fin = fin_lue + timedelta(days=1)
            # sinon : fin incohérente (OCR) -> durée par défaut
        evt.add("dtstart", debut)
        evt.add("dtend", fin)
    else:
        # Toute la journée : DTEND est exclusif (lendemain)
        evt.add("dtstart", jour)
        evt.add("dtend", jour + timedelta(days=1))

    cal = Calendar()
    cal.add("prodid", "-//CalEvent//ISEN Ouest//FR")
    cal.add("version", "2.0")
    if debut_txt:
        cal.add_component(Calendar.from_ical(VTIMEZONE_PARIS))
    cal.add_component(evt)
    return cal.to_ical()


def _nom_agenda(cal) -> str:
    try:
        return str(cal.name or "")
    except Exception:
        return ""


def _segment_url(cal) -> str:
    return str(getattr(cal, "url", "")).rstrip("/").rsplit("/", 1)[-1]


def trouver_agenda(principal, agenda: str = AGENDA_PAR_DEFAUT):
    """Cherche l'agenda PAR SON NOM (nom affiché ou dernier segment de l'adresse).

    Le compte de la squad contient aussi un agenda par membre : on ne prend jamais le premier.
    Lève ``LookupError`` en listant les agendas disponibles si rien ne correspond
    (l'agenda n'est pas créé : il vient du portail).
    """
    agendas = list(principal.calendars())
    for cal in agendas:
        if agenda in (_nom_agenda(cal), _segment_url(cal)):
            return cal
    dispo = ", ".join(f"« {_nom_agenda(c) or _segment_url(c)} »" for c in agendas) or "(aucun)"
    raise LookupError(f"Agenda « {agenda} » introuvable. Agendas disponibles : {dispo}")


def _existe(agenda, uid: str) -> bool:
    from caldav.lib.error import NotFoundError
    try:
        agenda.event_by_uid(uid)
        return True
    except NotFoundError:
        return False


def publier(evenements: list[dict], url: str, utilisateur: str, mot_de_passe: str,
            agenda: str = AGENDA_PAR_DEFAUT, dry_run: bool = False,
            _agenda_obj=None) -> list[dict]:
    """Publie des événements dans l'agenda CalDAV ``agenda``.

    Args:
        evenements: liste de dicts au schéma du cours (title, date, start_time, end_time,
            location, confidence).
        url, utilisateur, mot_de_passe: voir :func:`charger_identifiants`.
        agenda: nom de l'agenda partagé de la squad (``calevent``).
        dry_run: si vrai, construit les événements sans se connecter (statut « simulation »).

    Returns:
        Une liste de ``{"titre", "date", "uid", "statut"}`` (même ordre que l'entrée), statut
        parmi ``créé``, ``déjà présent``, ``simulation`` ou ``erreur: ...``. Un événement
        invalide n'arrête jamais les autres.

    Lève ``LookupError`` si l'agenda est introuvable (avant tout envoi).
    """
    cal_obj = _agenda_obj
    if not dry_run and cal_obj is None:
        import caldav
        client = caldav.DAVClient(url=url, username=utilisateur, password=mot_de_passe)
        cal_obj = trouver_agenda(client.principal(), agenda)

    resultats = []
    for ev in evenements:
        res = {"titre": ev.get("title"), "date": ev.get("date"), "uid": None, "statut": ""}
        try:
            res["uid"] = calculer_uid(ev)
            ical = construire_ical(ev, res["uid"])  # valide aussi la date et les heures
            if dry_run:
                res["statut"] = "simulation"
            elif _existe(cal_obj, res["uid"]):
                res["statut"] = "déjà présent"
            else:
                cal_obj.save_event(ical)
                res["statut"] = "créé"
        except Exception as e:  # on continue avec l'événement suivant
            res["statut"] = f"erreur: {e}"
        resultats.append(res)
    return resultats
