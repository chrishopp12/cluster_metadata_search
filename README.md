# Cluster General Data Search

A lightweight Python tool to retrieve object-centric metadata for galaxy clusters
(or cluster candidates) from major public databases, including SIMBAD and NED.

This script is intentionally standalone, minimal, and conservative in scope.
It retrieves cluster-level information (coordinates, redshifts, alternate names,
publications) without returning every galaxy in a field, making it suitable as an
early-stage lookup or orchestration component in larger cluster analysis pipelines.

---

## Core Design Philosophy

This script is object-centric, not region-centric.

Instead of performing wide cone searches that return thousands of member galaxies,
it:

1) Resolves the input (name or coordinates) to one best cluster-like object
2) Queries SIMBAD and NED about that object only
3) Collects:
   - Coordinates
   - Object type
   - Redshift estimates
   - Alternate names / cross-identifications
   - Bibliographic references (best-effort)

This avoids the “member-galaxy soup” problem and keeps outputs lightweight and
interpretable.

---

## Features

- Resolve targets by:
  - Name / identifier (e.g. RMJ IDs, observation IDs, catalog names)
  - RA / Dec (ICRS degrees)
- Optional mapping CSV for:
  - Short numerical tokens
  - Observation IDs
  - Internal naming schemes
- Object-centric metadata retrieval:
  - SIMBAD:
    - Coordinates
    - Object type
    - Alternate IDs
    - Redshift (rvz)
    - Bibliography (bibcodes)
  - NED:
    - Coordinates
    - Redshift (+ uncertainty, if available)
    - Alternate names
    - References
- Intelligent fallback logic:
  - Name lookup → coordinate seed search → nearest cluster-type object
- Clean, machine-readable outputs (JSON + text)
- Each backend can fail independently; failures are recorded in notes

---

## Installation

Create and activate a Python environment (recommended):

    conda create -n cluster_lookup python=3.11
    conda activate cluster_lookup

Install required packages:

    pip install numpy pandas astropy astroquery requests

Optional: resolving publication metadata via ADS

- Create an ADS API token
- Set it as an environment variable:

    export ADS_API_TOKEN="your_token_here"

---

## Usage

Resolve by name:

    python cluster_data_search.py --name "RMJ121917.6+505432.8"

Resolve by coordinates:

    python cluster_data_search.py --radec 184.8235 50.9091

Specify output directory:

python cluster_data_search.py --name "RMJ0003" --outdir ./cluster_general_output

Use a custom mapping CSV:

    python cluster_data_search.py --name "0881900801" --map-csv ./cluster_id_map.csv

---

## Mapping CSV (Optional)

If a file named cluster_id_map.csv exists next to the script (or is passed
explicitly), it will be used to normalize identifiers.

Required columns (case-insensitive):

    alias,full_name

Example:

    alias,full_name
    0003,RMJ000343.8+100123.8
    0881900801,RMJ121917.6+505432.8

This allows:
- Short numerical tokens
- Observation IDs
- Internal naming schemes

to resolve cleanly to a canonical cluster name.

---

## Output Structure

For a target resolved as RMJ121917_6+505432_8, the output directory may contain:

    cluster_general_output/
    ├── RMJ121917_6+505432_8_general_summary.json
    ├── RMJ121917_6+505432_8_alt_names.txt
    ├── RMJ121917_6+505432_8_publications.txt
    ├── RMJ121917_6+505432_8_publications_bibcodes.json
    └── RMJ121917_6+505432_8_publications_resolved.json  (if ADS token set)

Files are only written when meaningful content exists.

---

## General Summary JSON

The primary output is a structured JSON summary containing:

- Input resolution information
- Adopted target object name and type
- Coordinates and source of coordinate resolution
- Redshift candidates by source
- Deduplicated alternate names
- Cross-ID groupings
- Bibliographic references
- Notes describing failures, fallbacks, or ambiguities

This file is designed for downstream automation or ingestion by larger pipelines.

---

## Known Limitations

- Redshifts are reported by source, not combined or reconciled
- Object classification depends on SIMBAD/NED typing
- Bibliography coverage varies by service
- No bulk / batch mode yet
- No attempt to infer cluster membership or extent

---

## Roadmap (Planned, Not Implemented)


- Configurable allowed object types
- Better handling of ambiguous multi-object regions
- Integration with photometric, redshift, and radio lookup tools
- Unified multi-wavelength cluster metadata driver


