import re
import logging
from duckduckgo_search import DDGS
import time
import random

log = logging.getLogger(__name__)

def count_pronouns(text: str) -> dict:
    """Counts masculine and feminine pronouns in a given text."""
    if not text:
        return {'M': 0, 'F': 0}
    
    # Lowercase and pad with spaces to match whole words easily
    text = " " + text.lower() + " "
    
    # Using regex word boundaries for accurate matching
    # M pronouns: he, his, him, himself
    m_count = len(re.findall(r'\b(he|his|him|himself)\b', text))
    # F pronouns: she, her, hers, herself
    f_count = len(re.findall(r'\b(she|her|hers|herself)\b', text))
    
    return {'M': m_count, 'F': f_count}

def classify_by_pronouns(name: str, affiliation: str = None, max_results: int = 5) -> dict:
    """
    Searches DuckDuckGo for the inventor and counts pronouns in the snippets.
    Returns M, F, or UNKNOWN.
    """
    # Build query
    query = f'"{name}"'
    if affiliation and str(affiliation).lower() not in ['nan', 'none', '']:
        query += f' "{affiliation}"'
    
    # Add keywords that often yield biographies
    query += ' (LinkedIn OR university OR inventor OR patent OR biography OR "he is" OR "she is")'
    
    result = {
        'gender': None,
        'method': 'web_search_pronouns',
        'm_count': 0,
        'f_count': 0
    }
    
    try:
        # Rate limiting protection: sleep randomly between 1 and 3 seconds
        time.sleep(random.uniform(1.0, 3.0))
        
        with DDGS() as ddgs:
            # Get text search results
            search_results = list(ddgs.text(query, max_results=max_results))
            
            combined_text = ""
            for item in search_results:
                combined_text += item.get('body', '') + " " + item.get('title', '') + " "
            
            counts = count_pronouns(combined_text)
            result['m_count'] = counts['M']
            result['f_count'] = counts['F']
            
            # Decision logic: needs at least 2 pronouns to be confident, and a clear majority
            if counts['M'] >= 2 and counts['M'] > counts['F'] * 2:
                result['gender'] = 'M'
            elif counts['F'] >= 2 and counts['F'] > counts['M'] * 2:
                result['gender'] = 'F'
            else:
                result['gender'] = 'UNKNOWN'
                
            return result
            
    except Exception as e:
        log.warning(f"Web search failed for {name}: {e}")
        result['gender'] = 'UNKNOWN'
        return result
