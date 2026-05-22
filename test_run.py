import pandas as pd
from genderphoto import classify_batch
import logging

logging.basicConfig(level=logging.INFO)

df = pd.DataFrame([
    {'inventor_name': 'James Heckman', 'affiliation': 'University of Chicago', 'country_code': 'US'},
    {'inventor_name': 'Andrea Cavalleri', 'affiliation': 'Max Planck Hamburg', 'country_code': 'DE'},
    {'inventor_name': 'Jennifer Doudna', 'affiliation': 'UC Berkeley', 'country_code': 'US'},
    {'inventor_name': 'Fei-Fei Li', 'affiliation': 'Stanford University', 'country_code': 'US'},
])

# Use duckduckgo just to test the new feature as well!
result_df = classify_batch(
    df, 
    save_photos=False, 
    verbose=True, 
    search_engine='bing'
)

print("\n\n=== RESULTS ===")
print(result_df[['inventor_name', 'gender_final', 'gender_method', 'is_ambiguous', 'name_probability']].to_markdown())
