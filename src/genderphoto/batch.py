"""
Batch processing: classify a DataFrame of inventors.

Only runs the photo pipeline on names flagged as ambiguous by
the name classifier. Adds result columns to the DataFrame.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from genderphoto.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MAX_IMAGES,
    DEFAULT_SLEEP,
    DEFAULT_VLM,
    OLLAMA_URL,
    DEFAULT_SEARCH_ENGINE,
)
from genderphoto.pipeline import classify_inventor
from genderphoto.utils import extract_first_name

log = logging.getLogger(__name__)


def classify_batch(
    df: pd.DataFrame,
    name_col: str = 'inventor_name',
    affiliation_col: str = 'affiliation',
    country_col: str = 'country_code',
    max_images: int = DEFAULT_MAX_IMAGES,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    sleep: float = DEFAULT_SLEEP,
    save_photos: bool = True,
    photo_dir: str = './inventor_photos',
    vlm_model: str = DEFAULT_VLM,
    ollama_url: str = OLLAMA_URL,
    checkpoint_path: str = 'checkpoint.csv',
    checkpoint_every: int = 100,
    verbose: bool = False,
    name_threshold: float = None,
    search_engine: str = DEFAULT_SEARCH_ENGINE,
    detector_backend: str = 'retinaface',
) -> pd.DataFrame:
    """
    Process a DataFrame of inventors, adding gender classification columns.

    Only runs the photo/ensemble pipeline on ambiguous names. Unambiguous
    names are resolved by name only.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least a name column.
    name_col : str
        Column name for inventor full names.
    affiliation_col : str
        Column name for affiliations (optional, may be absent).
    country_col : str
        Column name for country codes (optional, may be absent).
    max_images : int
        Maximum images per inventor.
    confidence_threshold : float
        Minimum confidence to accept photo classification.
    sleep : float
        Seconds to sleep between inventors (rate limiting).
    save_photos : bool
        Whether to save best photos to disk.
    photo_dir : str
        Directory for saved photos.
    vlm_model : str
        Ollama VLM model name (e.g. 'qwen2.5vl:7b', 'llava:7b').
    ollama_url : str
        Ollama API endpoint.
    checkpoint_path : str, optional
        Path to periodically save intermediate results as CSV.
    checkpoint_every : int
        Save checkpoint every N inventors.
    verbose : bool
        If True, show detailed step-by-step logging for each inventor.
        If False (default), show only a progress bar.
    name_threshold : float, optional
        Probability threshold for name-based classification.
    search_engine : str
        'bing' (default) or 'duckduckgo'.
    detector_backend : str
        Face detector backend ('retinaface', 'opencv', etc.).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns: gender_name, gender_photo,
        gender_final, gender_method, name_probability, photo_confidence, 
        is_ambiguous, etc.
    """
    from tqdm import tqdm

    df = df.copy()
    n_total = len(df)
    
    # Save original index to handle duplicate indices safely, then reset
    orig_index = df.index
    df = df.reset_index(drop=True)

    # Control logging verbosity
    gp_logger = logging.getLogger('genderphoto')
    _prev_level = gp_logger.level
    if not verbose:
        gp_logger.setLevel(logging.WARNING)

    try:
        # Stage 1: Name-based classification for ALL rows
        if verbose:
            log.info("Stage 1: Name-based classification for %d inventors", n_total)

        from genderphoto.name_classifier import classify_name

        name_results = []
        kwargs = {}
        if name_threshold is not None:
            kwargs['threshold'] = name_threshold
            
        for _, row in df.iterrows():
            full_name = row[name_col]
            first_name = extract_first_name(full_name)
            country = row.get(country_col) if country_col in df.columns else None
            nr = classify_name(first_name, country, **kwargs)
            name_results.append(nr)

        name_df = pd.DataFrame(name_results)
        df['gender_name'] = name_df['gender'].values
        df['gender_name_raw'] = name_df['gender_raw'].values
        df['name_probability'] = name_df['name_probability'].values
        df['is_ambiguous'] = name_df['is_ambiguous'].values
        df['ambiguity_reason'] = name_df['ambiguity_reason'].values

        n_ambiguous = int(df['is_ambiguous'].sum())
        n_resolved = n_total - n_ambiguous

        if verbose:
            log.info(
                "Name-based: %d resolved, %d ambiguous (need photos)",
                n_resolved, n_ambiguous,
            )
        elif n_total > 0:
            tqdm.write(
                f"✓ Name-based: {n_resolved}/{n_total} resolved, "
                f"{n_ambiguous} need photo analysis"
            )

        # Initialize result columns
        result_cols = [
            'gender_photo', 'photo_confidence', 'photo_url', 'photo_saved_path',
            'photo_images_tried', 'photo_classifier', 'photo_error',
        ]
        for c in result_cols:
            if c not in df.columns:
                df[c] = None

        # Stage 2: Photo pipeline for ambiguous names only
        ambiguous_mask = df['is_ambiguous'] == True  # noqa: E712
        ambiguous_indices = df[ambiguous_mask].index.tolist()

        # Setup progress bar (shown only when not verbose)
        pbar = None
        if not verbose and n_ambiguous > 0:
            pbar = tqdm(
                total=n_ambiguous,
                desc="Photo analysis",
                unit="inv",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
            )

        for i, idx in enumerate(ambiguous_indices):
            row = df.loc[idx]
            name = row[name_col]
            affil = row.get(affiliation_col) if affiliation_col in df.columns else None

            if verbose:
                log.info("[%d/%d] %s (%s)", i + 1, n_ambiguous, name, affil or 'no affiliation')
            elif pbar:
                pbar.set_description(f"{name[:30]}")

            result = classify_inventor(
                name=name,
                affiliation=affil,
                country_code=row.get(country_col) if country_col in df.columns else None,
                max_images=max_images,
                confidence_threshold=confidence_threshold,
                save_photo_flag=save_photos,
                photo_dir=photo_dir,
                vlm_model=vlm_model,
                ollama_url=ollama_url,
                verbose=verbose,
                name_threshold=name_threshold,
                search_engine=search_engine,
                detector_backend=detector_backend,
            )

            df.at[idx, 'gender_photo'] = result['gender'] if result['gender'] != 'UNKNOWN' else None
            df.at[idx, 'photo_confidence'] = result.get('confidence')
            df.at[idx, 'photo_url'] = result.get('photo_url')
            df.at[idx, 'photo_saved_path'] = result.get('photo_saved_path')
            df.at[idx, 'photo_images_tried'] = result.get('images_tried', 0)
            df.at[idx, 'photo_classifier'] = result.get('method')
            df.at[idx, 'photo_error'] = result.get('error')

            # Update progress bar
            if pbar:
                gender = result.get('gender', '?')
                method = result.get('method', '')
                pbar.set_postfix_str(f"→ {gender} ({method[:25]})")
                pbar.update(1)

            # Periodic checkpoint
            if checkpoint_path and (i + 1) % checkpoint_every == 0:
                df.to_csv(checkpoint_path, index=False)
                if verbose:
                    log.info("Checkpoint saved (%d/%d)", i + 1, n_ambiguous)

            if i < len(ambiguous_indices) - 1:
                time.sleep(sleep)

        if pbar:
            pbar.close()

        # Stage 3: Consolidate final gender
        def _assign_final(row):
            if not row.get('is_ambiguous', True):
                g = str(row.get('gender_name_raw', '')).lower()
                if g == 'male':
                    return 'M', 'name_based'
                elif g == 'female':
                    return 'F', 'name_based'
            if pd.notna(row.get('gender_photo')):
                return row['gender_photo'], row.get('photo_classifier', 'photo')
            return 'UNKNOWN', 'unresolved'

        finals = df.apply(_assign_final, axis=1)
        df['gender_final'] = [f[0] for f in finals]
        df['gender_method'] = [f[1] for f in finals]

        # Final checkpoint
        if checkpoint_path:
            df.to_csv(checkpoint_path, index=False)
            if verbose:
                log.info("Final results saved to %s", checkpoint_path)

        n_classified = (df['gender_final'] != 'UNKNOWN').sum()
        if verbose:
            log.info("Batch complete: %d total, %d resolved", n_total, n_classified)
        else:
            tqdm.write(f"✓ Complete: {n_classified}/{n_total} classified")

        df.index = orig_index
        return df

    finally:
        gp_logger.setLevel(_prev_level)
