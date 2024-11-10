# from google.colab import files
# # Download the file
# files.download('interactions6.csv')

import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_amino_acid_sequence(protein_id, sequence_cache):
# Fetch the amino acid sequence using the protein ID

    if protein_id in sequence_cache:
        return sequence_cache[protein_id]

    url = f'https://rest.uniprot.org/uniprotkb/{protein_id}.fasta'
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to retrieve sequence for protein ID {protein_id}, Status Code: {response.status_code}")
        return None

    # The response will be in FASTA format --- the sequence starts after the first line
    sequence = ''.join(response.text.splitlines()[1:])
    sequence_cache[protein_id] = sequence
    return sequence

def get_pubmed_abstract(pubmed, abstract_cache):
# Fetch the abstract text from a PubMed article using the provided PubMed ID

    if pubmed in abstract_cache:
        return abstract_cache[pubmed]


    pubmed_url = f'https://pubmed.ncbi.nlm.nih.gov/{pubmed}/'
    response = requests.get(pubmed_url)

    if response.status_code != 200:
        print(f"Failed to retrieve abstract from PubMed, Status Code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    abstract_div = soup.find('div', class_='abstract-content selected')

    if abstract_div:
        abstract_text = abstract_div.get_text(strip=True, separator=' ')
        abstract_cache[pubmed] = abstract_text
        return abstract_text
    else:
        print("No abstract found.")
        return None

def scrape_mint_data(protein_name, sequence_cache, abstract_cache):
    base_url = f'https://mint.bio.uniroma2.it/index.php/results-interactions/?id={protein_name}'
    page = 0
    data = []  

    while True:
        url = f'{base_url}&page={page}' if page > 0 else base_url
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to retrieve data for {protein_name} on page {page}")
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'tablebygenes'})
        if table is None:
            print(f"No data found for {protein_name}")
            break

        rows = table.find('tbody').find_all('tr')
        if not rows:
            print(f"No interaction data found for {protein_name}")
            break

        for row in rows:
            cols = row.find_all('td')

            if len(cols) >= 6:
                gene_a_name = cols[0].find('a').text.strip()
                gene_a_href = cols[0].find('a')['href']
                gene_a_id = gene_a_href.split('/')[-1]

                gene_b_name = cols[1].find('a').text.strip()
                gene_b_href = cols[1].find('a')['href']
                gene_b_id = gene_b_href.split('/')[-1]

                if not gene_a_name or not gene_b_name:
                  continue

                interaction_type = cols[2].text.strip()
                detection_method = cols[3].text.strip()
                pubmed_id = cols[4].find('a').text.strip()  
                # details_link = cols[5].find('a')['href']

                data.append({
                    'Gene A': gene_a_name,
                    'Gene A ID': gene_a_id,
                    'Gene A Sequence': get_amino_acid_sequence(gene_a_id, sequence_cache),
                    'Gene B': gene_b_name,
                    'Gene B ID': gene_b_id,
                    'Gene B Sequence': get_amino_acid_sequence(gene_b_id, sequence_cache),
                    'Interaction Type': interaction_type,
                    'Detection Method': detection_method,
                    'PubMed ID': pubmed_id,
                    'Pubmed Abstract': get_pubmed_abstract(pubmed_id, abstract_cache) 
                })

                print(  {
                    'Gene A': gene_a_name,
                    'Gene A ID': gene_a_id,
                    'Gene A Sequence': get_amino_acid_sequence(gene_a_id, sequence_cache),
                    'Gene B': gene_b_name,
                    'Gene B ID': gene_b_id,
                    'Gene B Sequence': get_amino_acid_sequence(gene_b_id, sequence_cache),
                    'Interaction Type': interaction_type,
                    'Detection Method': detection_method,
                    'PubMed ID': pubmed_id,
                    'Pubmed Abstract': get_pubmed_abstract(pubmed_id, abstract_cache) 
                })

        next_page_link = soup.find('a', {'class': 'next'})
        if next_page_link is None:
            break  # no more pages available

        page += 1  # move to the next page

    return data  

def scrape_multiple_proteins(file_path):
# Read protein ids from a file and scrape interaction data for each 
    with open(file_path, 'r') as file:
        protein_ids = [line.strip() for line in file if line.strip()]

    all_data = []  
    sequence_cache = {}  # cache for sequences
    abstract_cache = {} # cache for abstracts

    total_proteins = len(protein_ids)

    for i, protein_id in enumerate(protein_ids, start=1):
        print(f"{i}: {protein_id}")  # current protein number
        protein_data = scrape_mint_data(protein_id, sequence_cache, abstract_cache)
        # print(protein_data)
        all_data.extend(protein_data)

    combined_df = pd.DataFrame(all_data)
    return combined_df




protein_file = 'valid_protein_batch_7.txt'
data_frame = scrape_multiple_proteins(protein_file)

data_frame.to_csv('protein_interactions_7.csv', index=False)
