import pandas as pd
import numpy as np

def extract_inter_idx(pep_seq, binding_vec):
    """Extract binding indices, handling padding and trimming without strict validation."""
    if pd.notna(binding_vec):
        # Pad or trim the binding vector to match the peptide sequence length
        if len(binding_vec) < len(pep_seq):
            print(f"Padding binding vector for {pep_seq}")
            binding_vec = binding_vec.ljust(len(pep_seq), '0')
        elif len(binding_vec) > len(pep_seq):
            print(f"Trimming binding vector for {pep_seq}")
            binding_vec = binding_vec[:len(pep_seq)]

        # Extract binding residue indices
        binding_lst = [idx for idx, val in enumerate(binding_vec) if val == '1']
        return ','.join(str(e) for e in binding_lst)
    return '-99999'

# Load input data
df_train = pd.read_csv(r'C:\Users\kwcoo\Documents\GitHub\CAMP\train_PDB_data_pos', header=0, sep='#')
df_zy_pep = pd.read_csv('./peptide-mapping (1).txt', header=None, sep='\t')
df_zy_pep.columns = ['bdb_id', 'bdb_pep_seq', 'pep_binding_vec']

# Extract relevant fields from bdb_id
df_zy_pep['pdb_id'] = df_zy_pep.bdb_id.apply(lambda x: x.split('_')[0])
df_zy_pep['pep_chain'] = df_zy_pep.bdb_id.apply(lambda x: x.split('_')[1].lower())
df_zy_pep['prot_chain'] = df_zy_pep.bdb_id.apply(lambda x: x.split('_')[2].upper())

# Drop duplicates based on bdb_id
df_zy_pep.drop_duplicates(['bdb_id'], inplace=True)

# Merge with the training data
df_join = pd.merge(df_train, df_zy_pep, how='left',
                   left_on=['pdb_id', 'pep_chain', 'prot_chain'],
                   right_on=['pdb_id', 'pep_chain', 'prot_chain'])

# Extract binding indices
df_join['binding_idx'] = df_join.apply(
    lambda x: extract_inter_idx(x.pep_seq, x.pep_binding_vec), axis=1
)

# Save the valid entries to a CSV for further analysis
df_join.to_csv('pdb_pairs_bindingsites.csv', index=False, sep=',')

print("Process completed successfully.")
