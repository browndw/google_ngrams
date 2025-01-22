
import re
import pandas as pd
import requests
from io import StringIO


def google_ngram(word_forms, variety=["eng", "gb", "us", "fiction"], by=["year", "decade"]):
    word_forms = [re.sub(r'([a-zA-Z0-9])-([a-zA-Z0-9])', r'\1 - \2', wf) for wf in word_forms]
    word_forms = [wf.strip() for wf in word_forms]
    n = [len(re.findall(r'\S+', wf)) for wf in word_forms]
    n = list(set(n))
    
    if len(n) > 1:
        raise ValueError("Check spelling. Word forms should be lemmas of the same word (e.g. 'teenager' and 'teenagers' or 'walk' , 'walks' and 'walked'")
    if n[0] > 5:
        raise ValueError("Ngrams can be a maximum of 5 tokens. Hyphenated words are split and include the hyphen, so 'x-ray' would count as 3 tokens.")
    
    gram = [wf[:2] if n[0] > 1 else wf[:1] for wf in word_forms]
    gram = list(set([g.lower() for g in gram]))
    
    if len(gram) > 1:
        raise ValueError("Check spelling. Word forms should be lemmas of the same word (e.g. 'teenager' and 'teenagers' or 'walk' , 'walks' and 'walked'")
    
    if re.match(r'^[a-z][^a-z]', gram[0]):
        gram[0] = re.sub(r'[^a-z]', '_', gram[0])
    if re.match(r'^[0-9]', gram[0]):
        gram[0] = gram[0][:1]
    if re.match(r'^[\W]', gram[0]):
        gram[0] = "punctuation"
    
    if any(re.match(r'^[ßæðøłœıƒþȥəħŋªºɣđĳɔȝⅰʊʌʔɛȡɋⅱʃɇɑⅲ]', g) for g in gram):
        gram[0] = "other"
    
    gram[0] = gram[0].encode('latin-1', 'replace').decode('latin-1')
    
    if variety[0] == "eng":
        repo = f"http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-{n[0]}gram-20120701-{gram[0]}.gz"
    else:
        repo = f"http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-{variety[0]}-all-{n[0]}gram-20120701-{gram[0]}.gz"
    
    print("Accessing repository. For larger ones (e.g., ngrams containing 2 or more words) this may take a few minutes. A progress bar should appear shortly...")
    
    word_forms = [re.sub(r'(\.|\\?|\\$|\\^|\\)|\\(|\\}|\\{|\\]|\\[|\\*)', r'\\\1', wf) for wf in word_forms]
    grep_words = "|".join([f"^{wf}$" for wf in word_forms])
    
    # Read the data from the repository
    response = requests.get(repo)
    data = StringIO(response.content.decode('utf-8'))
    all_grams = pd.read_csv(data, sep='\t', header=None, names=["token", "Year", "AF", "pages"])
    all_grams = all_grams[all_grams['token'].str.contains(grep_words, case=False)]
    
    if variety[0] == "eng":
        total_counts = ngramr_plus.googlebooks_eng_all_totalcounts_20120701
    elif variety[0] == "gb":
        total_counts = ngramr_plus.googlebooks_eng_gb_all_totalcounts_20120701
    elif variety[0] == "us":
        total_counts = ngramr_plus.googlebooks_eng_us_all_totalcounts_20120701
    
    if by[0] == "year":
        total_counts = total_counts.groupby('Year').sum().reset_index()
    if by[0] == "decade":
        total_counts['Decade'] = total_counts['Year'].str.replace(r'\d$', '0', regex=True)
        total_counts = total_counts.groupby('Decade').sum().reset_index()
    
    all_grams['token'] = all_grams['token'].str.lower()
    sum_tokens = all_grams.groupby('Year')['AF'].sum().reset_index()
    
    if by[0] == "decade":
        sum_tokens['Decade'] = sum_tokens['Year'].str.replace(r'\d$', '0', regex=True)
        sum_tokens = sum_tokens.groupby('Decade')['AF'].sum().reset_index()
        sum_tokens = sum_tokens.merge(total_counts[['Decade', 'Total']], on='Decade')
        sum_tokens['Decade'] = sum_tokens['Decade'].astype(int)
    if by[0] == "year":
        sum_tokens = sum_tokens.merge(total_counts[['Year', 'Total']], on='Year')
    
    counts_norm = (sum_tokens['AF'] / sum_tokens['Total']) * 1000000
    sum_tokens['Per_10.6'] = counts_norm
    
    return sum_tokens
