"""
Utility functions: image loading, saving, logging setup, name parsing.
"""

from __future__ import annotations

import logging
import os
import re
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
import pandas as pd

log = logging.getLogger(__name__)



def setup_logging(
    log_file: str = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure the root genderphoto logger.

    Parameters
    ----------
    log_file : str, optional
        Path to a log file. If provided, logs are also written to this file
        with UTF-8 encoding (Windows-safe).
    level : int
        Logging level (default INFO).

    Returns
    -------
    logging.Logger
        The configured 'genderphoto' logger.
    """
    logger = logging.getLogger('genderphoto')
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

        # Console handler
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        # File handler (utf-8 for Windows compatibility)
        if log_file:
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


def load_image(path_or_url: str, timeout: int = 10) -> Image.Image | None:
    """
    Load an image from a local path or URL.

    Returns None if the image cannot be loaded or is smaller than 50x50.
    """
    try:
        if os.path.isfile(path_or_url):
            img = Image.open(path_or_url).convert('RGB')
        else:
            r = requests.get(
                path_or_url,
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=timeout,
            )
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert('RGB')
        if img.width >= 50 and img.height >= 50:
            return img
        return None
    except Exception:
        return None


def save_photo(
    img: Image.Image,
    name: str,
    photo_dir: str = './inventor_photos',
) -> str:
    """
    Save an image to disk with a filesystem-safe name derived from the
    inventor's name.

    Returns the saved file path as a string.
    """
    photo_dir = Path(photo_dir)
    photo_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    path = photo_dir / f"{safe}.jpg"
    img.save(path, quality=90)
    log.debug("Saved photo -> %s", path)
    return str(path)


def extract_first_name(full_name: str) -> str:
    """
    Extract the first name from a full name string.

    Handles both 'First Last' and 'Last, First' formats.
    """
    full_name = full_name.strip()
    if ',' in full_name:
        return full_name.split(',')[1].strip().split()[0]
    return full_name.split()[0]


ASIAN_SURNAMES = {
    'wang', 'li', 'zhang', 'liu', 'chen', 'yang', 'huang', 'zhao', 'wu', 'zhou',
    'xu', 'sun', 'ma', 'zhu', 'hu', 'guo', 'lin', 'he', 'gao', 'liang', 'zheng',
    'luo', 'song', 'xie', 'tang', 'han', 'cao', 'deng', 'xiao', 'feng', 'cheng',
    'cai', 'yuan', 'peng', 'pan', 'shen', 'ding', 'wei', 'jiang', 'ye', 'hong',
    'kim', 'lee', 'park', 'choi', 'jeong', 'kang', 'cho', 'yoon', 'jang', 'lim',
    'nguyen', 'tran', 'le', 'pham', 'huynh', 'hoang', 'phan', 'vu', 'vo', 'dang',
}

def is_asian_name(full_name: str) -> bool:
    """
    Check if the name likely belongs to an East Asian inventor based on a 
    set of the most common Chinese, Korean, and Vietnamese surnames.
    """
    parts = full_name.lower().replace('-', ' ').split()
    return any(p in ASIAN_SURNAMES for p in parts)


def compute_partial_identification_bounds(
    df, 
    gender_col: str = 'gender_final',
    method_col: str = 'gender_method',
    country_col: str = 'country_code'
) -> dict:
    """
    Compute partial-identification bounds (Manski bounds, 1989) for the female share
    in a classified population where some records remain unclassified ('UNKNOWN' or None).
    Also separates name-resolved vs photo-resolved shares and computes the country-matched
    plausible scenario (imputing unknown records from country-specific female shares).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing classification results.
    gender_col : str
        Column containing final gender values ('M', 'F', 'UNKNOWN', None).
    method_col : str, optional
        Column indicating resolution stage ('name_based', 'deepface_consensus', 'ensemble_vlm_override', etc.).
    country_col : str, optional
        Column containing ISO 2-letter country of residence codes.

    Returns
    -------
    dict
        {
            'total_population': int,
            'classified_count': int,
            'unknown_count': int,
            'female_count': int,
            'male_count': int,
            'observed_female_share': float,               # Overall % F among classified
            'observed_female_share_name_resolved': float, # p_N (% F among name-resolved)
            'observed_female_share_photo_resolved': float,# p_P (% F among photo-resolved)
            'lower_bound': float,                         # % F assuming all UNKNOWN = M
            'upper_bound': float,                         # % F assuming all UNKNOWN = F
            'country_matched_share': float,               # Plausible scenario (% F imputed via country shares)
        }
    """
    total = len(df)
    if total == 0:
        return {
            'total_population': 0,
            'classified_count': 0,
            'unknown_count': 0,
            'female_count': 0,
            'male_count': 0,
            'observed_female_share': 0.0,
            'observed_female_share_name_resolved': 0.0,
            'observed_female_share_photo_resolved': 0.0,
            'lower_bound': 0.0,
            'upper_bound': 0.0,
            'country_matched_share': 0.0,
        }

    s = df[gender_col].astype(str).str.upper()
    f_mask = (s == 'F')
    m_mask = (s == 'M')
    classified_mask = f_mask | m_mask
    unknown_mask = ~classified_mask

    f_count = int(f_mask.sum())
    m_count = int(m_mask.sum())
    classified = f_count + m_count
    unknown = total - classified

    obs_share = round((f_count / classified * 100.0), 2) if classified > 0 else 0.0
    lower_bound = round((f_count / total * 100.0), 2)
    upper_bound = round(((f_count + unknown) / total * 100.0), 2)

    # Separate p_N (name resolved) from p_P (photo/vlm resolved)
    p_N = 0.0
    p_P = 0.0
    if method_col in df.columns:
        methods = df[method_col].astype(str).str.lower()
        name_mask = classified_mask & (methods == 'name_based')
        photo_mask = classified_mask & (methods != 'name_based') & (methods != 'unknown') & (methods != 'none')
        
        name_classified = int(name_mask.sum())
        name_f = int((name_mask & f_mask).sum())
        p_N = round((name_f / name_classified * 100.0), 2) if name_classified > 0 else 0.0

        photo_classified = int(photo_mask.sum())
        photo_f = int((photo_mask & f_mask).sum())
        p_P = round((photo_f / photo_classified * 100.0), 2) if photo_classified > 0 else 0.0

    # Country-matched plausible scenario (impute unknowns using country-specific F shares among classified)
    imputed_f_count = float(f_count)
    if unknown > 0:
        if country_col in df.columns:
            # Calculate observed F share by country among classified
            country_f_shares = {}
            for country, group in df[classified_mask].groupby(country_col):
                if pd.notna(country) and str(country).strip() != '':
                    c_f = (group[gender_col].astype(str).str.upper() == 'F').sum()
                    country_f_shares[str(country).upper()] = c_f / len(group)
            
            # Impute each unknown record
            for _, row in df[unknown_mask].iterrows():
                c = str(row.get(country_col, '')).upper()
                if c in country_f_shares:
                    imputed_f_count += country_f_shares[c]
                else:
                    imputed_f_count += (obs_share / 100.0)
        else:
            imputed_f_count += unknown * (obs_share / 100.0)

    country_matched_share = round((imputed_f_count / total * 100.0), 2) if total > 0 else 0.0

    return {
        'total_population': total,
        'classified_count': classified,
        'unknown_count': unknown,
        'female_count': f_count,
        'male_count': m_count,
        'observed_female_share': obs_share,
        'observed_female_share_name_resolved': p_N,
        'observed_female_share_photo_resolved': p_P,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'country_matched_share': country_matched_share,
    }


