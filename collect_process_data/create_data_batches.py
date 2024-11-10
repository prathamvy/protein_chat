import os

protein_file = 'all_valid_proteins.txt'  
batch_size = 4000

if not os.path.exists(protein_file):
    print(f"file {protein_file} does not exist")
    exit()

with open(protein_file, 'r') as f:
    protein_names = f.read().splitlines()

num_batches = (len(protein_names) + batch_size - 1) // batch_size 

for i in range(num_batches):
    batch = protein_names[i * batch_size:(i + 1) * batch_size]
    output_file = f'valid_protein_batch_{i + 1}.txt'  

    with open(output_file, 'w') as f_out:
        f_out.write('\n'.join(batch))

    print(f'Created {output_file} with {len(batch)} records')