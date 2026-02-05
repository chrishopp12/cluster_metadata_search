#!/usr/bin/env python3
"""
general_data_search.py

General (object-centric) galaxy cluster data search.

Core idea:
- For "general cluster info" (coords, z, alt names, publications), we must avoid
  returning every galaxy in a region. So this script is OBJECT-CENTRIC:
    1) Resolve input (name or coordinate) to ONE best "cluster-like" target object
    2) Query SIMBAD / NED for metadata about THAT target object

Inputs:
- --name "..."  OR  --radec RA_deg DEC_deg
- Optional CSV mapping file next to this script: cluster_id_map.csv
    * maps custom keys (e.g., obsids) and short tokens (e.g., RMJ0003) to a canonical name

Outputs (in --outdir):
- <tag>_general_summary.json
- <tag>_alt_names.txt
- <tag>_publications.txt (bibcodes or reference IDs best-effort)

Dependencies:
- astropy, astroquery, pandas, numpy

Notes:
- Each backend can fail independently; failures go to "notes".
- This is v1: robust, minimal, and extensible without heavy edge-case logic.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import os
import time
import requests

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import astropy.units as u
from astropy.coordinates import SkyCoord, get_icrs_coordinates

from astroquery.simbad import Simbad
from astroquery.ipac.ned import Ned



# ------------------------------------
# Defaults/ Constants
# ------------------------------------

DEFAULT_MAP_FILENAME = "cluster_id_map.csv"
DEFAULT_OUTDIR = "./../cluster_data_search_output"

DEFAULT_SEED_RADII_ARCSEC = [30, 60, 120] 
DEFAULT_SIMBAD_CLUSTER_OTYPES = [
    "ClG",  
]
DEFAULT_NED_CLUSTER_TYPES = [
    "GClstr", 
]

# Catalog-style rewrites for cross-ID normalization
CATALOG_PREFIX_MAP: dict[str, str] = {
    "WHL2009": "WHL",
    "RRB2014": "RMJ", 
}

ADS_API_URL = "https://api.adsabs.harvard.edu/v1/search/query"



# -----------------------------
# Data models
# -----------------------------

@dataclass
class RedshiftCandidate:
    z: float
    z_err: float | None
    source: str                 # e.g. "simbad", "ned"
    detail: str | None = None   # column/table notes


@dataclass
class TargetObject:
    name: str | None            
    coord: SkyCoord | None
    source: str | None          # "input", "sesame", "simbad_seed", "ned_seed"
    otype: str | None = None
    separation_arcsec: float | None = None


@dataclass
class GeneralSummary:
    # Input
    # input_name: str | None
    input_radec_deg: tuple[float, float] | None

    # Name normalization / mapping
    mapped_name: str | None  # after applying CSV map / short-token rules

    # Adopted target object
    target_name: str | None
    target_otype: str | None
    ra_deg: float | None
    dec_deg: float | None
    coord_source: str | None
    separation_arcsec: float | None

    # Data products
    alt_names: list[str]
    cross_ids: dict[str, list[str]]
    redshifts_by_source: dict[str, float]  # e.g. {"simbad": 0.3214, "ned": 0.322}
    redshift_candidates: list[RedshiftCandidate]

    publications: list[str]               # bibcodes or NED ref IDs 
    notes: list[str]


# -----------------------------
# Utilities
# -----------------------------

def safe_stem(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_+") else "_" for ch in s).strip("_")

def _digits(s: str) -> str:
    """Return only digits from a string."""
    return "".join(re.findall(r"\d+", str(s)))

def normalize_minus_hyphen(s: str) -> str:
    """
    Normalize common Unicode minus/dash characters to ASCII hyphen-minus '-' (which happens if you copy-paste a cluster name, especially from LaTeX).

    This catches cases like:
      RM J015949.3−084958.9  (U+2212 minus sign)
    and converts to:
      RM J015949.3-084958.9

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        String with normalized hyphens.
    """
    if s is None:
        return s

    # Common “looks like a hyphen/minus” code points encountered in copy/paste
    dash_chars = (
        "\u2212"  # MINUS SIGN
        "\u2010"  # HYPHEN
        "\u2011"  # NON-BREAKING HYPHEN
        "\u2012"  # FIGURE DASH
        "\u2013"  # EN DASH
        "\u2014"  # EM DASH
        "\u2015"  # HORIZONTAL BAR
        "\uFE63"  # SMALL HYPHEN-MINUS
        "\uFF0D"  # FULLWIDTH HYPHEN-MINUS
    )

    trans = {ord(ch): "-" for ch in dash_chars}
    return str(s).translate(trans)

def _short4_from_full_name(full_name: str) -> str | None:
    """
    Extract the first 4 digits after 'RMJ' from a canonical RMJ name.

    Example:
      RMJ132724.2+534656.5 -> '1327'
      RMJ000343.8+100123.8 -> '0003'

    Returns None if it cannot be derived.
    """
    s = str(full_name).strip()
    if not s.upper().startswith("RMJ"):
        return None
    # Take everything after RMJ, strip to digits, take first 4
    d = _digits(s[3:])
    if len(d) < 4:
        return None
    return d[:4]

def ensure_outdir(path: str | Path) -> Path:
    outdir = Path(path).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        val = str(it).strip()
        if not val:
            continue
        key = val.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out



def normalize_cross_id(raw: str) -> str:
    """
    Convert a cross-ID to a canonical-ish form for deduping.
    Keep this conservative and extend over time.

    Returns a *key* used for deduping (not necessarily pretty for display).
    """
    if raw is None:
        return ""

    s = str(raw).strip()
    if not s:
        return ""

    # Collapse internal whitespace
    s = re.sub(r"\s+", " ", s)

    # Extract bracket tag if present: [RRB2014] ...
    m = re.match(r"^\[([A-Za-z0-9]+)\]\s*(.*)$", s)
    if m:
        tag = m.group(1)
        rest = m.group(2).strip()


        if tag == "RRB2014":
            # Examples:
            # [RRB2014] RM J121917.6+505432.8 = RMJ121917.6+505432.8
            rest = re.sub(r"^RM\s+J", "RMJ", rest)
            rest = re.sub(r"^RMJ\s+", "RMJ", rest)
            s = rest

        elif tag == "WHL2009":
            # [WHL2009] J121917.6+505432 = WHL J121917.6+505432
            if re.match(r"^J\d", rest):
                s = "WHL " + rest
            else:
                s = "WHL " + rest if not rest.upper().startswith("WHL") else rest

        else:
            # Default: drop the [TAG] and keep remainder
            s = rest

    # Standardize spacing around catalog tokens
    s = re.sub(r"\s+", " ", s).strip()

    # Normalize RM J... forms if they show up without [RRB2014]
    s = re.sub(r"^RM\s+J", "RMJ", s, flags=re.IGNORECASE)

    # Normalize "WHL J..." (ensure single space)
    s = re.sub(r"^WHL\s+J", "WHL J", s, flags=re.IGNORECASE)

    # Normalize PSZ variants: PSZ1/PSZ2/PSZRX keep as-is but unify spaces
    s = re.sub(r"\s+", " ", s).strip()

    # Uppercase catalog token (first word), keep the rest as-is
    parts = s.split(" ", 1)
    if len(parts) == 2:
        s = parts[0].upper() + " " + parts[1]
    else:
        s = parts[0].upper()

    return s


def choose_preferred_label(candidates: list[str]) -> str:
    """
    Pick the nicest display string among duplicates.
    Heuristics: prefer strings without bracket tags, and with fewer weird chars.
    """
    def score(x: str) -> tuple[int, int, int]:
        # lower score is better
        has_brackets = 1 if re.search(r"^\[.*\]", x.strip()) else 0
        length = len(x)
        spaces = x.count(" ")
        return (has_brackets, spaces, length)

    return sorted(candidates, key=score)[0]


def dedupe_cross_ids(raw_ids: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """
    Returns:
      - deduped_display: list of preferred labels
      - groups: dict[normalized_key] -> list of original strings
    """
    groups: dict[str, list[str]] = {}
    for r in raw_ids:
        key = normalize_cross_id(r)
        if not key:
            continue
        groups.setdefault(key, []).append(str(r).strip())

    deduped_display = [choose_preferred_label(v) for v in groups.values()]
    deduped_display.sort()

    return deduped_display, groups


def write_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def resolve_bibcodes_ads(
    bibcodes: list[str],
    *,
    token_env: str = "ADS_API_TOKEN",
    fields: list[str] | None = None,
    sleep_s: float = 0.2,
) -> list[dict]:
    """
    Resolve ADS bibcodes to bibliographic metadata via ADS Search API.

    Requires env var ADS_API_TOKEN to be set.
    """
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(
            f"Missing ADS token. Set {token_env} in your environment. "
            "You can generate one in ADS Settings -> API Token."  # see docs
        )

    if fields is None:
        fields = ["bibcode", "title", "author", "pub", "year", "doi", "citation_count"]

    headers = {"Authorization": f"Bearer {token}"}

    out: list[dict] = []
    for bc in bibcodes:
        bc = str(bc).strip()
        if not bc:
            continue

        params = {
            "q": f"bibcode:{bc}",
            "fl": ",".join(fields),
            "rows": 1,
        }

        r = requests.get(ADS_API_URL, headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            out.append({"bibcode": bc, "error": f"HTTP {r.status_code}: {r.text[:200]}"})
            continue

        data = r.json()
        docs = data.get("response", {}).get("docs", [])
        if not docs:
            out.append({"bibcode": bc, "error": "not found"})
            continue

        doc = docs[0]
        # Normalize a few common fields to your preferred format
        title = doc.get("title")
        if isinstance(title, list):
            title = title[0] if title else None

        authors = doc.get("author") or []
        first_author = authors[0] if authors else None

        doi = doc.get("doi")
        if isinstance(doi, list):
            doi = doi[0] if doi else None

        out.append(
            {
                "bibcode": doc.get("bibcode", bc),
                "year": doc.get("year"),
                "first_author": first_author,
                "title": title,
                "pub": doc.get("pub"),
                "doi": doi,
                "citation_count": doc.get("citation_count"),
                "ads_url": f"https://ui.adsabs.harvard.edu/abs/{doc.get('bibcode', bc)}/abstract",
            }
        )

        time.sleep(sleep_s)

    return out


def load_id_map_csv(map_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """
    Load a 2-column mapping CSV placed next to the script.

    Required columns (case-insensitive):
      - alias
      - full_name

    Returns
    -------
    alias_map : dict[str, str]
        Maps alias strings (and lightly normalized variants) to full_name.
    short_map : dict[str, str]
        Maps 'short4' (e.g. '1327') to full_name, derived from full_name.

    Notes
    -----
    - We intentionally *do not* enforce strict alias formatting.
    - Resolution later will use digit-stripping on the user input to match short_map.
    """
    if not map_path.exists():
        return {}, {}

    df = pd.read_csv(map_path)

    col_map = {c.lower().strip(): c for c in df.columns}
    if "alias" not in col_map or "full_name" not in col_map:
        raise ValueError(
            f"Mapping CSV must have columns: alias, full_name. Found: {list(df.columns)}"
        )

    alias_col = col_map["alias"]
    full_col = col_map["full_name"]

    alias_map: dict[str, str] = {}
    short_map: dict[str, str] = {}

    for _, row in df.iterrows():
        alias = str(row[alias_col]).strip()
        full = str(row[full_col]).strip()

        if not alias or not full or alias.lower() == "nan" or full.lower() == "nan":
            continue

        # Direct alias match
        alias_map[alias] = full
        # Alias without spaces
        alias_map[alias.replace(" ", "")] = full

        # Derived short-name mapping from full name
        short4 = _short4_from_full_name(full)
        if short4 is not None:
            # If collisions ever occur, keep the first and note later if needed.
            short_map.setdefault(short4, full)

    return alias_map, short_map


def resolve_name(identifier: str, alias_map: dict[str, str], short_map: dict[str, str]) -> str:
    """
    Resolve user identifier to a canonical full RMJ name when possible.

    Priority
    --------
    1) direct alias match (raw + raw-without-spaces)
    2) digit-stripping short match:
         user_digits[:4] compared to short_map keys
    3) fallback: return identifier unchanged
    """
    if identifier is None:
        raise ValueError("No identifier provided.")

    raw = str(identifier).strip()
    if not raw:
        raise ValueError("No identifier provided.")

    if raw in alias_map:
        return alias_map[raw]

    raw_ns = raw.replace(" ", "")
    if raw_ns in alias_map:
        return alias_map[raw_ns]

    d = _digits(raw)
    if len(d) >= 4:
        short4 = d[:4]
        if short4 in short_map:
            return short_map[short4]

    return raw


def normalize_cluster_key(s: str) -> str:
    # Remove whitespace, underscores, hyphens; uppercase.
    return re.sub(r"[\s\-_]+", "", s.strip().upper())


def maybe_expand_short_rmj(user_input: str) -> str | None:
    """
    Accepts: '0003', 'RMJ0003', 'RMJ_0003', 'RMJ-0003'
    Returns: 'RMJ0003' or None
    """
    key = normalize_cluster_key(user_input)
    if re.fullmatch(r"\d{4}", key):
        return f"RMJ{key}"
    if re.fullmatch(r"RMJ\d{4}", key):
        return key
    return None


def apply_map_or_short(identifier: str, id_map: dict[str, str]) -> str:
    """
    1) direct map match
    2) short-token normalization, then map match
    3) return identifier unchanged
    """
    raw = str(identifier).strip()

    if raw in id_map:
        return id_map[raw]

    short = maybe_expand_short_rmj(raw)
    if short and short in id_map:
        return id_map[short]

    return raw


def redshift_by_source(cands: list[RedshiftCandidate]) -> dict[str, float]:
    """
    v1: report one z per source (SIMBAD/NED), without trying to combine.
    If a source provides multiple candidates, take the first one added
    (which we control in code).
    """
    out: dict[str, float] = {}
    for c in cands:
        if c.source not in out and c.z is not None and np.isfinite(c.z) and c.z >= 0:
            out[c.source] = float(c.z)
    return out



# -----------------------------
# Target resolution
# -----------------------------

def resolve_target_from_name(mapped_name: str, notes: list[str]) -> TargetObject:
    """
    Resolve name -> coordinate using Astropy Sesame resolver (object-centric).
    Returns a target with coord and source "sesame".
    """
    try:
        coord = get_icrs_coordinates(mapped_name)
        return TargetObject(
            name=mapped_name,
            coord=coord,
            source="sesame",
        )
    except Exception as e:
        notes.append(f'Name resolution failed via Sesame for "{mapped_name}": {e}')
        return TargetObject(name=mapped_name, coord=None, source=None)


def search_by_coord(
    *,
    tab,
    coord: SkyCoord,
    cols: dict[str, str],
    service: str,
    notes: list[str],
    allowed_types: list[str]
) -> TargetObject | None:
    if tab is None or len(tab) == 0:
        return None

    for key in ("name", "ra", "dec", "type"):
        if key not in cols or cols[key] not in tab.colnames:
            notes.append(
                f"{service} seed: missing required column '{cols.get(key)}'. Columns: {tab.colnames}"
            )
            return None
        
    allowed = {str(x).strip().casefold() for x in allowed_types if str(x).strip()}

    def clean_keywords(t: str) -> set[str]:
        # Keep underscores; split on common separators
        parts = re.split(r"[,\s;|]+", t.strip())
        return {p.casefold() for p in parts if p}


    ras = np.asarray(tab[cols["ra"]], dtype=float)
    decs = np.asarray(tab[cols["dec"]], dtype=float)
    types = [str(v).strip() for v in tab[cols["type"]]]

    valid = np.isfinite(ras) & np.isfinite(decs)
    if not np.any(valid):
        return None

    ras = ras[valid]
    decs = decs[valid]
    types = [t for t, v in zip(types, valid) if v]
    names = [str(tab[cols["name"]][i]).strip() for i in np.flatnonzero(valid)]

    coords = SkyCoord(ras * u.deg, decs * u.deg, frame="icrs")
    seps = coord.separation(coords).arcsec

    allowed  = {cluster_type.lower() for cluster_type in allowed_types}

    cluster_idxs = [i for i, t in enumerate(types) if clean_keywords(t) & allowed]
    if not cluster_idxs:
        return None

    i_best = cluster_idxs[int(np.nanargmin(seps[cluster_idxs]))]

    return TargetObject(
        name=names[i_best],
        coord=coords[i_best],
        source=f"{service}_seed",
        otype=types[i_best],
        separation_arcsec=float(seps[i_best]),
    )


def simbad_search_by_coord(
    coord: SkyCoord,
    notes: list[str],
    seed_radii_arcsec: list[int] = DEFAULT_SEED_RADII_ARCSEC,
    allowed_types: list[str] = DEFAULT_SIMBAD_CLUSTER_OTYPES,
) -> TargetObject | None:
    if not isinstance(coord, SkyCoord):
        raise TypeError("SIMBAD seed requires SkyCoord")

    sim = Simbad()
    sim.TIMEOUT = 20
    sim.add_votable_fields("otype")

    cols = {"name": "main_id", "ra": "ra", "dec": "dec", "type": "otype"}

    for r in seed_radii_arcsec:
        try:
            tab = sim.query_region(coord, radius=r * u.arcsec)
        except Exception as e:
            notes.append(f'SIMBAD seed query failed at r={r}": {e}')
            continue

        result = search_by_coord(
            tab=tab,
            coord=coord,
            cols=cols,
            service="simbad",
            notes=notes,
            allowed_types=allowed_types,
        )
        if result:
            return result

    return None


def ned_search_by_coord(
    coord: SkyCoord,
    notes: list[str],
    seed_radii_arcsec: list[int] = DEFAULT_SEED_RADII_ARCSEC,
    allowed_types: list[str] = DEFAULT_NED_CLUSTER_TYPES,
) -> TargetObject | None:
    if not isinstance(coord, SkyCoord):
        raise TypeError("NED seed requires SkyCoord")

    cols = {"name": "Object Name", "ra": "RA", "dec": "DEC", "type": "Type"}

    for r in seed_radii_arcsec:
        try:
            tab = Ned.query_region(coord, radius=r * u.arcsec)
        except Exception as e:
            notes.append(f'NED seed query failed at r={r}": {e}')
            continue

        result = search_by_coord(
            tab=tab,
            coord=coord,
            cols=cols,
            service="ned",
            notes=notes,
            allowed_types=allowed_types,
        )
        if result:
            return result

    return None


def search_by_name(
    *,
    tab: Any,
    cols: dict[str, str],
    service: str,
    notes: list[str],
) -> TargetObject | None:
    if tab is None or len(tab) == 0:
        return None

    # Lowercase → actual column name mapping
    colmap = {c.casefold(): c for c in tab.colnames}

    def get_col(key: str) -> str | None:
        want = cols.get(key)
        return colmap.get(want.casefold()) if want else None

    name_col = get_col("name")
    ra_col = get_col("ra")
    dec_col = get_col("dec")
    type_col = get_col("type")  # optional

    if name_col is None or ra_col is None or dec_col is None:
        notes.append(
            f"{service} name: missing required columns. "
            f"Expected {list(cols.values())}, got {tab.colnames}"
        )
        return None

    obj = str(tab[name_col][0]).strip()
    ra = float(tab[ra_col][0])
    dec = float(tab[dec_col][0])
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")

    otype = str(tab[type_col][0]).strip() if type_col is not None else None

    return TargetObject(
        name=obj,
        coord=coord,
        source=f"{service}_name",
        otype=otype,
        separation_arcsec=None,
    )



def simbad_target_from_name(name: str, notes: list[str]) -> TargetObject | None:
    """
    Try SIMBAD object lookup by name. Returns a TargetObject with SIMBAD main_id and coord if successful.
    """
    try:
        sim = Simbad()
        sim.TIMEOUT = 20
        sim.add_votable_fields("otype")
        tab = sim.query_object(name)
        if tab is None or len(tab) == 0:
            return None

        cols = {"name": "main_id", "ra": "ra", "dec": "dec", "type": "otype"}

        return search_by_name(
            tab=tab,
            cols=cols,
            service="simbad",
            notes=notes,
        )
    
    except Exception as e:
        notes.append(f'SIMBAD name lookup failed for "{name}": {e}')
        return None

def ned_target_from_name(name: str, notes: list[str]) -> TargetObject | None:
    """
    Try NED object lookup by name. Returns a TargetObject with NED Object Name and coord if successful.
    """
    try:
        Ned.clear_cache()
        tab = Ned.query_object(name)
        if tab is None or len(tab) == 0:
            return None

        cols = {"name": "Object Name", "ra": "RA", "dec": "DEC", "type": "Type"}

        return search_by_name(
            tab=tab,
            cols=cols,
            service="ned",
            notes=notes,
        )
    
    except Exception as e:
        notes.append(f'NED name lookup failed for "{name}": {e}')
        return None


def define_target_by_name(
        *,
        mapped_name: str,
        notes: list[str],
        seed_radii_arcsec: list[int] = DEFAULT_SEED_RADII_ARCSEC,
        simbad_allowed_types: list[str] = DEFAULT_SIMBAD_CLUSTER_OTYPES,
        ned_allowed_types: list[str] = DEFAULT_NED_CLUSTER_TYPES,
) -> tuple[TargetObject | None, TargetObject | None]:
    
    simbad_target = simbad_target_from_name(mapped_name, notes=notes)
    if simbad_target is None: # Fall back to coordinate search
        notes.append(f'SIMBAD name lookup failed for "{mapped_name}", trying coordinate search fallback.')
        resolved_target = resolve_target_from_name(mapped_name, notes=notes)
        if resolved_target.coord is not None:
            simbad_target = simbad_search_by_coord(
                coord=resolved_target.coord,
                notes=notes,
                seed_radii_arcsec=seed_radii_arcsec,
                allowed_types=simbad_allowed_types,
            )
        else:
            notes.append(f'Cannot perform SIMBAD coordinate search fallback for "{mapped_name}" due to missing coordinates.')
        if simbad_target is None:
            notes.append(f'SIMBAD coordinate search fallback failed for "{mapped_name}".')

    ned_target = ned_target_from_name(mapped_name, notes=notes)
    if ned_target is None: # Fall back to coordinate search
        notes.append(f'NED name lookup failed for "{mapped_name}", trying coordinate search fallback.')
        resolved_target = resolve_target_from_name(mapped_name, notes=notes)
        if resolved_target.coord is not None:
            ned_target = ned_search_by_coord(
                coord=resolved_target.coord,
                notes=notes,
                seed_radii_arcsec=seed_radii_arcsec,
                allowed_types=ned_allowed_types,
            )
        else:
            notes.append(f'Cannot perform NED coordinate search fallback for "{mapped_name}" due to missing coordinates.')
        if ned_target is None:
            notes.append(f'NED coordinate search fallback failed for "{mapped_name}".')

    return simbad_target, ned_target


# -----------------------------
# Metadata queries
# -----------------------------

def simbad_object_metadata(target_name: str, notes: list[str]) -> dict[str, Any]:
    """
    Query SIMBAD for object-centric metadata:
      - coord (best-effort)
      - ids (alt names)
      - otype
      - rvz_redshift (best-effort)
      - bibliography (bibcodes best-effort)

    Returns a dict with keys: coord, ids, otype, redshift, publications
    """
    sim = Simbad()
    sim.TIMEOUT = 20
    sim.add_votable_fields("ra", "dec", "ids", "otype", "rvz_redshift", "biblio")

    out: dict[str, Any] = {"coord": None, "ids": [], "otype": None, "redshift": None, "publications": []}

    try:
        tab = sim.query_object(target_name)
    except Exception as e:
        notes.append(f'SIMBAD query_object failed for "{target_name}": {e}')
        return out

    if tab is None or len(tab) == 0:
        notes.append(f'SIMBAD: no object match for "{target_name}"')
        return out

    cols = {c.lower(): c for c in tab.colnames}

    # coordinates
    ra_col = cols.get("ra")
    dec_col = cols.get("dec")
    if ra_col and dec_col:
        try:
            ra_deg = float(tab[ra_col][0])
            dec_deg = float(tab[dec_col][0])
            out["coord"] = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
        except Exception:
            pass

    # ids
    ids_col = cols.get("ids")
    if ids_col:
        raw_ids = tab[ids_col][0]
        if raw_ids:
            out["ids"] = [s.strip() for s in str(raw_ids).split("|") if s.strip()]

    # otype
    otype_col = cols.get("otype")
    if otype_col:
        out["otype"] = str(tab[otype_col][0]).strip()

    # redshift
    z_col = cols.get("rvz_redshift")
    if z_col:
        zv = tab[z_col][0]
        try:
            z = float(zv) if zv is not None else None
            if z is not None and np.isfinite(z) and z >= 0:
                out["redshift"] = z
        except Exception:
            pass

    # bibcodes
    bib_col = cols.get("biblio")
    if bib_col:
        raw_bibs = tab[bib_col][0]
        if raw_bibs:
            out["publications"] = [s.strip() for s in str(raw_bibs).split("|") if s.strip()]

    return out


def ned_object_metadata(target_name: str, notes: list[str]) -> dict[str, Any]:
    """
    Query NED for object-centric metadata:
      - coord (from query_object)
      - redshift (from query_object if present; else from redshifts table)
      - alt names (from names table best-effort)
      - references (best-effort; may fail depending on service/table format)

    Returns dict with keys: coord, names, redshift, redshift_err, publications
    """
    out: dict[str, Any] = {"coord": None, "names": [], "redshift": None, "redshift_err": None, "publications": []}
    Ned.clear_cache()
    # query_object for coord + quick z
    try:
        tab = Ned.query_object(target_name)
    except Exception as e:
        notes.append(f'NED query_object failed for "{target_name}": {e}')
        tab = None

    if tab is not None and len(tab) > 0:
        cols = {c.lower(): c for c in tab.colnames}
        ra_col = cols.get("ra")
        dec_col = cols.get("dec")
        if ra_col and dec_col:
            try:
                ra_deg = float(tab[ra_col][0])
                dec_deg = float(tab[dec_col][0])
                out["coord"] = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
            except Exception:
                pass

        z_col = cols.get("redshift")
        if z_col:
            z = tab[z_col][0]
            try:
                zf = float(z) if z is not None else None
                if zf is not None and np.isfinite(zf) and zf >= 0:
                    out["redshift"] = zf
            except Exception:
                pass

        zerr_col = cols.get("redshift uncertainty")
        if zerr_col:
            ze = tab[zerr_col][0]
            try:
                zef = float(ze) if ze is not None else None
                if zef is not None and np.isfinite(zef) and zef > 0:
                    out["redshift_err"] = zef
            except Exception:
                pass

        name_col = cols.get("object name")
        if name_col:
            out["names"].append(str(tab[name_col][0]).strip())


    # redshifts table fallback (if no z yet)
    if out["redshift"] is None:
        try:
            ztab = Ned.get_table(target_name, table="redshifts")
            if ztab is not None and len(ztab) > 0:
                zcols = {c.lower(): c for c in ztab.colnames}
                zc = zcols.get("redshift")
                if zc:
                    z = ztab[zc][0]
                    try:
                        zf = float(z) if z is not None else None
                        if zf is not None and np.isfinite(zf) and zf >= 0:
                            out["redshift"] = zf
                    except Exception:
                        pass
        except Exception as e:
            notes.append(f'NED redshifts table failed for "{target_name}": {e}')

    # references table best-effort
    try:
        rtab = Ned.get_table(target_name, table="references")
        if rtab is not None and len(rtab) > 0:
            rcols = {c.lower(): c for c in rtab.colnames}
            # column might be "Refcode" or similar
            ref_col = next((rcols[k] for k in rcols if "ref" in k or "bib" in k or "code" in k), None)
            if ref_col:
                out["publications"] = [str(v).strip() for v in rtab[ref_col] if str(v).strip()]
    except Exception as e:
        notes.append(f'NED references table failed for "{target_name}": {e}')

    return out


# -----------------------------
# Orchestrator
# -----------------------------

def run_general_search(
    *,
    mapped_name: str | None,
    input_coord: SkyCoord | None,
) -> GeneralSummary:
    notes: list[str] = []
    alt_names: list[str] = []
    cross_ids: dict[str, list[str]] = {}
    z_cands: list[RedshiftCandidate] = []
    publications: list[str] = []


    if mapped_name is None and input_coord is None:
        raise ValueError("Internal: mapped_name expected when input_coord is None.")
    
    simbad_target: TargetObject | None = None
    ned_target: TargetObject | None = None

    if mapped_name is not None:
        simbad_target, ned_target = define_target_by_name(
            mapped_name=mapped_name,
            notes=notes,
        )
    elif input_coord is not None:
        simbad_target = simbad_search_by_coord(
            input_coord,
            notes=notes,
        )
        ned_target = ned_search_by_coord(
            input_coord,
            notes=notes,
        )

    if not simbad_target and not ned_target:
        if input_coord:
            # As a last resort, we cannot safely identify a unique cluster object.
            # We still return the coordinate, but we won't do publications/alt-names by region.
            notes.append("Failed to identify a unique cluster-like target object from coordinates.")
            target = TargetObject(name=None, coord=input_coord, source="input", separation_arcsec=0.0)

        else:
            # Name-only: use Sesame resolver for coord; and use name itself as target_name.
            if mapped_name is None:
                raise ValueError("Internal: mapped_name expected when input_coord is None.")
            
            target = resolve_target_from_name(mapped_name, notes=notes)
            if target.coord is None:
                # Can't continue meaningfully without a coord
                notes.append("Cannot proceed without coordinates.")
                return GeneralSummary(
                    input_radec_deg=None,
                    mapped_name=mapped_name,
                    target_name=None,
                    target_otype=None,
                    ra_deg=None,
                    dec_deg=None,
                    coord_source=None,
                    separation_arcsec=None,
                    alt_names=[],
                    cross_ids={},
                    redshifts_by_source={},
                    redshift_candidates=[],
                    publications=[],
                    notes=notes,
                )
    # Last ditch effort to cross-search if only one source found
    if ned_target and not simbad_target:
        notes.append("Attempting Simbad coordinate search with NED target coordinates.")
        simbad_target = simbad_search_by_coord(
            coord=ned_target.coord,
            notes=notes,
        )
    if simbad_target and not ned_target:
        notes.append("Attempting NED coordinate search with SIMBAD target coordinates.")
        ned_target = ned_search_by_coord(
            coord=simbad_target.coord,
            notes=notes,
        )


    # Step 3: Object-centric metadata queries
    if simbad_target and simbad_target.name:
        # SIMBAD object metadata
        s = simbad_object_metadata(simbad_target.name, notes=notes)

        if s.get("ids"):
            cross_ids["simbad"] = list(s["ids"])
            alt_names.extend(list(s["ids"]))

        if s.get("otype") and simbad_target.otype is None:
            simbad_target.otype = s["otype"]

        if s.get("redshift") is not None:
            z_cands.append(RedshiftCandidate(z=float(s["redshift"]), z_err=None, source="simbad", detail="rvz_redshift"))

        if s.get("publications"):
            publications.extend(list(s["publications"]))
    else:
        notes.append("No unique target object name; skipping object-centric SIMBAD metadata lookup.")

        # NED object metadata
    if ned_target and ned_target.name:
        n = ned_object_metadata(ned_target.name, notes=notes)

        if n.get("names"):
            cross_ids["ned"] = list(n["names"])
            alt_names.extend(list(n["names"]))

        if n.get("redshift") is not None:
            z_cands.append(
                RedshiftCandidate(
                    z=float(n["redshift"]),
                    z_err=float(n["redshift_err"]) if n.get("redshift_err") is not None else None,
                    source="ned",
                    detail="query_object/redshifts",
                )
            )
        if n.get("publications"):
            publications.extend(list(n["publications"]))
    else:
        notes.append("No unique target object name; skipping object-centric NED metadata lookup.")

    # Step 4: Deduplicate names + publications
    alt_names, _ = dedupe_cross_ids(alt_names)
    alt_names = dedupe_preserve_order(alt_names)
    publications = dedupe_preserve_order(publications)




    # Step 5: Report redshift per source
    redshifts_by_source = redshift_by_source(z_cands)

    # Step 6: Final target selection
    if simbad_target and ned_target is None:
        target = simbad_target
    elif ned_target and simbad_target is None:
        target = ned_target
    elif simbad_target and ned_target:
        # Both available: first, check if the Ned name matches SIMBAD name or alt-names
        ned_name_norm = normalize_cluster_key(ned_target.name) if ned_target.name else None
        simbad_name_norm = normalize_cluster_key(simbad_target.name) if simbad_target.name else None
        alt_name_norms = {normalize_cluster_key(n) for n in alt_names} if alt_names else set()
        if ned_name_norm and (ned_name_norm == simbad_name_norm or ned_name_norm in alt_name_norms):
            target = ned_target
        else:
            # Otherwise, pick the one with smaller separation
            if simbad_target.separation_arcsec is not None and ned_target.separation_arcsec is not None:
                if simbad_target.separation_arcsec <= ned_target.separation_arcsec:
                    target = simbad_target
                else:
                    target = ned_target
            elif simbad_target.separation_arcsec is not None:
                target = simbad_target
            elif ned_target.separation_arcsec is not None:
                target = ned_target
            else:
                target = simbad_target  # arbitrary fallback
    else:
        # Neither available: use input coord only
        if input_coord is None:
            raise ValueError("Internal: input_coord expected when no target found.")
        target = TargetObject(name=None, coord=input_coord, source="input", separation_arcsec=0.0)

    ra_deg = float(target.coord.ra.deg) if target.coord is not None else None
    dec_deg = float(target.coord.dec.deg) if target.coord is not None else None

    return GeneralSummary(
        input_radec_deg=(float(input_coord.ra.deg), float(input_coord.dec.deg)) if input_coord is not None else None,
        mapped_name=mapped_name,
        target_name=target.name,
        target_otype=target.otype,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        coord_source=target.source,
        separation_arcsec=target.separation_arcsec,
        alt_names=alt_names,
        cross_ids=cross_ids,
        redshifts_by_source=redshifts_by_source,
        redshift_candidates=z_cands,
        publications=publications,
        notes=notes,
    )


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Object-centric general cluster data search (SIMBAD/NED/VizieR)."
    )

    tgt = p.add_mutually_exclusive_group(required=True)
    tgt.add_argument("--name", type=str, help='Cluster name or identifier (e.g., "0881900801", "RMJ_0003", "RMJ121917.6+505432.8").')
    tgt.add_argument("--radec", nargs=2, type=float, metavar=("RA_DEG", "DEC_DEG"), help="ICRS degrees.")
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help=f"Output directory. [default: {DEFAULT_OUTDIR}]")
    p.add_argument("--map-csv", type=str, default=None, help=f"Mapping CSV path. Default is {DEFAULT_MAP_FILENAME} next to this script.")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    outdir = ensure_outdir(args.outdir)

    # mapping CSV path next to script by default
    script_dir = Path(__file__).resolve().parent
    default_map = script_dir.parent / DEFAULT_MAP_FILENAME
    map_path = Path(args.map_csv).expanduser().resolve() if args.map_csv else default_map


    try:
        alias_map, short_map = load_id_map_csv(map_path) if map_path.exists() else ({}, {})
        if alias_map or short_map:
            logging.info(
                f"Loaded mapping CSV: {map_path} "
                f"({len(alias_map)} aliases, {len(short_map)} short tokens)"
            )
        else:
            logging.info("No mapping CSV loaded (file missing or empty mapping).")
    except Exception as e:
        logging.error(f"Failed to load mapping CSV: {e}")
        return 2

    if args.name:
        input_name = args.name
        input_coord = None
        mapped_name = resolve_name(input_name, alias_map, short_map)
        tag = safe_stem(mapped_name)
    else:
        mapped_name = None
        ra, dec = args.radec
        input_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        tag = safe_stem(f"RA{ra:.5f}_DEC{dec:.5f}")

    summary = run_general_search(
        mapped_name=mapped_name,
        input_coord=input_coord,
    )

    # Write outputs

    bibcodes = dedupe_preserve_order(summary.publications)

    write_json(
        outdir / f"{tag}_publications_bibcodes.json",
        [{"bibcode": b} for b in bibcodes],
    )

    # Resolve via ADS if token exists
    if os.environ.get("ADS_API_TOKEN"):
        try:
            pubs_resolved = resolve_bibcodes_ads(bibcodes)
            write_json(
                outdir / f"{tag}_publications_resolved.json",
                pubs_resolved,
            )
        except Exception as e:
            summary.notes.append(f"ADS publication resolution failed: {e}")
    else:
        summary.notes.append("ADS_API_TOKEN not set; skipping publication metadata resolution.")

    payload = asdict(summary)
    summary_path = outdir / f"{tag}_general_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    logging.info(f"Wrote: {summary_path}")

    alt_path = outdir / f"{tag}_alt_names.txt"
    alt_path.write_text("\n".join(summary.alt_names) + ("\n" if summary.alt_names else ""))
    logging.info(f"Wrote: {alt_path}")

    pubs_path = outdir / f"{tag}_publications.txt"
    pubs_path.write_text("\n".join(summary.publications) + ("\n" if summary.publications else ""))
    logging.info(f"Wrote: {pubs_path}")


    # Console summary
    logging.info("----- General data summary -----")
    logging.info(f"Mapped name: {summary.mapped_name}")
    logging.info(f"Target name: {summary.target_name}")
    logging.info(f"Coord: {summary.ra_deg}, {summary.dec_deg}  (source={summary.coord_source})")
    logging.info(f"Target otype: {summary.target_otype}")
    logging.info(f"Separation: {summary.separation_arcsec} arcsec")
    logging.info(f"Alt names: {len(summary.alt_names)}")
    logging.info(f"Publications: {len(summary.publications)}")
    if summary.redshifts_by_source:
        z_parts = [f"{src}={z:.6f}" for src, z in summary.redshifts_by_source.items()]
        logging.info("Redshifts: " + ", ".join(z_parts))
    else:
        logging.info("Redshifts: (none)")


    if summary.notes:
        logging.info("Notes:")
        for n in summary.notes:
            logging.info(f"  - {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
