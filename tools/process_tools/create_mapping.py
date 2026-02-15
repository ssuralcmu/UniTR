"""
Utility to harmonize map file identifiers between different NuScenes-based BEV map
generation pipelines.  In the BEVMapMatch/BEVFusion codebase the generated
map images (and corresponding metadata) are saved with file names that embed
both the sample timestamp and the NuScenes sample token.  For example, the
visualization script constructs the filename using the pattern
``f"{timestamp}-{token}_generated_map_image.png"``【299543150027250†L158-L163】.  If another
pipeline (e.g. UniTR) stores its map outputs using a different naming
convention—often only containing the sample token—it becomes difficult to
line‑up corresponding BEV maps.  This script builds a mapping between
UniTR‑style filenames and the BEVMapMatch naming convention.

Given a directory containing UniTR map outputs and (optionally) a directory
containing the BEVMapMatch/BEVFusion outputs, the script extracts the
NuScenes sample token from each filename, queries the NuScenes API to
retrieve the timestamp for that token, and then constructs the BEVFusion
equivalent filename ``f"{timestamp}-{token}_generated_map_image.png"``.  If a
matching file already exists in the BEVMapMatch directory, the script uses
that exact filename; otherwise it synthesizes a new filename using the
timestamp lookup.  The resulting mapping from the UniTR filename to the
BEVMapMatch filename is written to a JSON file for downstream use.

Example usage:

    python map_id_mapping.py \
        --unitr-dir /path/to/unitr/maps \
        --bev-dir /path/to/bev/maps \
        --output mapping.json \
        --nuscenes-root /data/sets/nuscenes \
        --version v1.0-trainval

The script requires the NuScenes library to be installed (``pip install
nuscenes-devkit``).  If you only want to rely on the filenames already
present in the BEV directory and not perform a timestamp lookup, you can
omit ``--nuscenes-root`` and ``--version``; in that case tokens missing in
the BEV directory will be skipped.
"""

import argparse
import json
import os
import re
from typing import Dict, Optional

try:
    # Import lazily to allow running without NuScenes when not needed.
    from nuscenes.nuscenes import NuScenes  # type: ignore
except ImportError:
    NuScenes = None  # type: ignore

TOKEN_PATTERN = re.compile(r"[0-9a-fA-F]{32}")


def extract_token(filename: str) -> Optional[str]:
    """Search for a 32‑character hex string in a filename.

    NuScenes sample tokens are 32 hexadecimal characters.  This helper
    extracts the first occurrence of such a token from a filename.

    Args:
        filename: Name of the file (not including directories).

    Returns:
        The 32‑character token if found, otherwise ``None``.
    """
    match = TOKEN_PATTERN.search(filename)
    return match.group(0) if match else None


def load_bev_token_map(bev_dir: str) -> Dict[str, str]:
    """Build a mapping from sample token to BEVMapMatch filename.

    Each file in the BEVMapMatch output directory contains both a timestamp
    and token separated by a hyphen (``{timestamp}-{token}_generated_map_image.png``)
   【299543150027250†L158-L163】.  By extracting the token from each filename, we can
    associate the token with its full BEV filename.

    Args:
        bev_dir: Path to the directory with BEVMapMatch map images.

    Returns:
        A dictionary mapping ``token -> filename`` for all files in ``bev_dir``
        that contain a token.
    """
    mapping: Dict[str, str] = {}
    if bev_dir and os.path.isdir(bev_dir):
        for fname in os.listdir(bev_dir):
            token = extract_token(fname)
            if token:
                # Only record the first occurrence; duplicates should not occur.
                mapping.setdefault(token, fname)
    return mapping


def lookup_timestamp(nusc: 'NuScenes', token: str) -> str:
    """Retrieve the timestamp for a given sample token.

    Args:
        nusc: An instance of ``NuScenes``.
        token: The 32‑character sample token.

    Returns:
        The integer timestamp as a string.

    Raises:
        ValueError: If the token cannot be found in the dataset.
    """
    sample = nusc.get('sample', token)
    if sample is None:
        raise ValueError(f"Token {token} not found in NuScenes dataset")
    return str(sample['timestamp'])


def create_mapping(unitr_dir: str,
                   bev_dir: Optional[str],
                   output_json: str,
                   nuscenes_root: Optional[str] = None,
                   version: str = 'v1.0-trainval') -> None:
    """Generate a mapping from UniTR filenames to BEVMapMatch filenames.

    Args:
        unitr_dir: Directory containing UniTR map outputs.
        bev_dir: Directory containing BEVMapMatch map outputs.  If provided,
            tokens present in this directory will be used directly.  Tokens
            missing here will fall back to timestamp lookup if NuScenes is
            available.
        output_json: Path where the mapping JSON will be written.
        nuscenes_root: Root directory of the NuScenes dataset.  Required
            when you need to reconstruct filenames using timestamps.
        version: Which NuScenes version to load (e.g. 'v1.0-trainval').

    The resulting JSON will map each UniTR filename to the corresponding
    BEVMapMatch filename.  If a token is not found in either the BEV
    directory or the NuScenes dataset, that entry will be omitted and a
    warning printed.
    """
    if not os.path.isdir(unitr_dir):
        raise FileNotFoundError(f"UniTR directory '{unitr_dir}' does not exist")

    # Pre‑build a mapping from token -> BEV filename if a BEV directory is provided.
    bev_token_map: Dict[str, str] = load_bev_token_map(bev_dir) if bev_dir else {}

    # Initialise NuScenes if timestamp lookup is required.
    nusc = None
    if nuscenes_root:
        if NuScenes is None:
            raise ImportError(
                "The nuscenes-devkit is not installed.  Install it with 'pip install nuscenes-devkit'."
            )
        nusc = NuScenes(dataroot=nuscenes_root, version=version, verbose=False)

    mapping: Dict[str, str] = {}
    missing_tokens = []

    for fname in os.listdir(unitr_dir):
        token = extract_token(fname)
        if not token:
            # Cannot find a token; skip this file.
            missing_tokens.append((fname, None))
            continue

        bev_fname = None

        # Prefer the BEV directory if it has this token.
        if token in bev_token_map:
            bev_fname = bev_token_map[token]
        elif nusc:
            # Fall back to constructing the filename using NuScenes timestamp lookup.
            try:
                timestamp = lookup_timestamp(nusc, token)
                bev_fname = f"{timestamp}-{token}_generated_map_image.png"
            except Exception:
                # Token not in dataset or other error.
                missing_tokens.append((fname, token))
                bev_fname = None

        if bev_fname:
            mapping[fname] = bev_fname

    # Report any files that were skipped.
    if missing_tokens:
        for fname, token in missing_tokens:
            if token:
                print(f"Warning: token {token} (from {fname}) not found in BEV directory or NuScenes dataset.  Skipping.")
            else:
                print(f"Warning: could not extract NuScenes token from filename {fname}.  Skipping.")

    # Write mapping to JSON.
    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(mapping, f, indent=4)
    print(f"Wrote mapping for {len(mapping)} files to {output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create mapping between UniTR and BEVMapMatch map filenames")
    parser.add_argument('--unitr-dir', required=True, help='Directory containing UniTR-generated map files')
    parser.add_argument('--bev-dir', default=None, help='Directory containing BEVMapMatch-generated map files')
    parser.add_argument('--output', default='unitr_to_bev_mapping.json', help='Output JSON file name')
    parser.add_argument('--nuscenes-root', default=None, help='Path to the root of the NuScenes dataset')
    parser.add_argument('--version', default='v1.0-trainval', help='NuScenes version (e.g. v1.0-trainval)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    create_mapping(
        unitr_dir=args.unitr_dir,
        bev_dir=args.bev_dir,
        output_json=args.output,
        nuscenes_root=args.nuscenes_root,
        version=args.version
    )