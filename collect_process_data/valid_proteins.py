# we check if there are any interaction data present for the proteins 
# in the mint database

# check if protein is valid and edit txt files
import requests
from bs4 import BeautifulSoup
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def scrape_mint_data(protein_name):
    base_url = f'https://mint.bio.uniroma2.it/index.php/results-interactions/?id={protein_name}'
    page = 0

    while True:
        url = f'{base_url}&page={page}' if page > 0 else base_url
        try:
            response = requests.get(url)
            response.raise_for_status() 

            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'id': 'tablebygenes'})
            if table is None:
                return False  # no data found

            rows = table.find('tbody').find_all('tr')
            if len(rows) == 0:
                # print(f"No interaction data found for {protein_name}")
                return False

            # If data is found, write the valid protein name to the file
            with open("validp_extra.txt", 'a') as valid_file:
                valid_file.write(f"{protein_name}\n")  # write the valid protein name

            return True  # data was found

        except requests.HTTPError as http_err:
            print(f"HTTP error occurred for {protein_name}: {http_err}", flush=True)
            return False
        except Exception as err:
            print(f"An error occurred for {protein_name}: {err}", flush=True)
            return False

protein_file = '/content/protein_batch_extra.txt'  
if not os.path.exists(protein_file):
    print(f"file {protein_file} does not exist")
    exit()

with open(protein_file, 'r') as f:
    protein_names = f.read().splitlines()

# ThreadPoolExecutor for concurrent requests
with ThreadPoolExecutor(max_workers=10) as executor:  
    future_to_protein = {executor.submit(scrape_mint_data, protein_name): protein_name for protein_name in protein_names}

    for count, future in enumerate(as_completed(future_to_protein), start=1):
        protein_name = future_to_protein[future]
        try:
            if future.result():
                print(f"[{count}] Protein added: {protein_name}", flush=True)  
            else:
                print(f"[{count}] Protein removed: {protein_name}", flush=True)  
        except Exception as exc:
            print(f"[{count}] {protein_name} generated an exception: {exc}", flush=True)  
