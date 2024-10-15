import pandas as pd
import numpy as np


def check_abnormal_aa(peptide_seq):
    len_seq = len(peptide_seq)
    cnt = sum(1 for i in peptide_seq if i in ['G', 'A', 'P', 'V', 'L', 'I', 'M', 'F',
                                              'Y', 'W', 'S', 'T', 'C', 'N', 'Q', 'K',
                                              'H', 'R', 'D', 'E'])
    return float(cnt) / len_seq


def lower_chain(input_str):
    return ''.join([item.lower() if item.isalpha() else item for item in input_str])


# Step 1: Load and process PDB peptide data and PLIP results
def load(pdb_pep_dataset, plip_result_filename):
    # Load the PDB peptide dataset
    df_fasta_pep = pd.read_csv(pdb_pep_dataset, sep='\t', header=0)
    df_fasta_pep = df_fasta_pep.reset_index(drop=True)
    df_predict = pd.DataFrame(columns=['pdb_id', 'pep_chain', 'predicted_chain'])

    # Process each PDB ID and chain to find corresponding PLIP result files
    for i in range(df_fasta_pep.shape[0]):
        pdb_id = df_fasta_pep['PDB_id'][i]
        chain = df_fasta_pep['chain'][i]
        result_file_name = f'./peptide_result/{pdb_id}_{chain}_result.txt'

        # Debugging: Check if file exists
        print(f"Checking PLIP result file: {result_file_name}")

        try:
            # Process the PLIP result file if it exists
            for line in open(result_file_name):
                if line.startswith('Interacting chain(s):'):
                    df_predict.loc[i] = [pdb_id, chain,
                                         line.replace('\n', '')
                                         .replace('\r', '')
                                         .replace('Interacting chain(s):', '')
                                         .lower()]
            if i % 5000 == 0:
                print(f"Processed {i} files so far...")
        except FileNotFoundError:
            print(f"File not found: {result_file_name}. Skipping...")

    print("Finished loading PLIP results.")
    print("-----------------------------------------------------")
    print(f"Loaded df_predict shape: {df_predict.shape}")
    print(df_predict.head())  # Debug: Print a sample of the data

    # Handle cases with no valid predictions
    if df_predict.empty:
        print("Warning: No predictions found. Ensure PLIP result files are present.")
        return df_predict  # Return an empty DataFrame

    # Filter out empty or invalid predicted chains
    df_predict['predicted_chain'] = df_predict['predicted_chain'].apply(lambda x: x.replace(' ', ''))
    df_predict['pep_chain'] = df_predict['pep_chain'].apply(lambda x: x.replace(' ', ''))
    df_predict = df_predict[df_predict['predicted_chain'].str.len() > 0]

    # Handle cases where no valid predictions remain
    if df_predict.empty:
        print("Error: No valid predicted chains found. Skipping concatenation step.")
        return df_predict  # Return the DataFrame as-is

    # Explode the comma-separated predicted chains into individual rows
    df_predict['predicted_chain'] = df_predict['predicted_chain'].apply(lambda x: x.split(','))
    lst_col = 'predicted_chain'
    df1 = pd.DataFrame({
        col: np.repeat(df_predict[col].values, df_predict[lst_col].str.len())
        for col in df_predict.columns.difference([lst_col])
    }).assign(**{lst_col: np.concatenate(df_predict[lst_col].values)})[df_predict.columns.tolist()]

    # Save the processed PLIP predictions to a file
    df_predict = df1
    df_predict.to_csv(plip_result_filename, encoding='utf-8', index=False, sep='\t')
    print(f"Finished exploding predicted chains. Saved {df_predict.shape[0]} records.")

    print("Step 1 is finished by generating the PLIP prediction file.")
    return df_predict


# Step 2: Load the predicted chains and their sequences from PDB fasta data
def load_all_fasta(all_fasta_file, input_dataset):
    df_fasta = pd.read_csv(all_fasta_file, sep='\t', header=0)
    df_fasta_protein = df_fasta[df_fasta['PDB_type'] == 'protein']

    # Prepare vocabulary for merging
    df_fasta_protein['PDB_id'] = df_fasta_protein['PDB_id_chain'].apply(lambda x: x.split('_')[0])
    df_fasta_protein['chain'] = df_fasta_protein['PDB_id_chain'].apply(lambda x: x.split('_')[1].lower())
    df_fasta_vocabulary = df_fasta_protein[['PDB_id', 'chain', 'PDB_seq']]

    # Merge with the input dataset
    df_predict_det = pd.merge(input_dataset, df_fasta_vocabulary,
                              how='left', left_on=['pdb_id', 'pep_chain'],
                              right_on=['PDB_id', 'chain'])

    df_predict_det1 = pd.merge(df_predict_det, df_fasta_vocabulary,
                               how='left', left_on=['pdb_id', 'predicted_chain'],
                               right_on=['PDB_id', 'chain'])
    df_predict_det1 = df_predict_det1.drop(['PDB_id_x', 'chain_x', 'PDB_id_y', 'chain_y'], axis=1)
    df_predict_det1.columns = ['pdb_id', 'pep_chain', 'predicted_chain', 'pep_seq', 'prot_seq']

    # Filter sequences based on length
    df_predict_det1['pep_seq_len'] = df_predict_det1['pep_seq'].apply(len)
    df_predict_det1['prot_seq_len'] = df_predict_det1['prot_seq'].apply(len)
    df_predict_det1 = df_predict_det1[(df_predict_det1['pep_seq_len'] <= 50) &
                                      (df_predict_det1['prot_seq_len'] > 50)]

    # Remove sequences with too many non-standard residues
    df_predict_det1['peptide_seq_score'] = df_predict_det1['pep_seq'].apply(check_abnormal_aa)
    df_predict_det1 = df_predict_det1[df_predict_det1['peptide_seq_score'] >= 0.8]

    print("Finished removing sequences with too many non-standard residues.")
    return df_predict_det1


# Main execution
if __name__ == "__main__":
    # Step 1: Load PLIP predictions
    df_predict = load('pdb_pep_chain', 'plip_predict_result')

    # Step 2: Load and filter fasta sequences
    if not df_predict.empty:
        df_predict_det1 = load_all_fasta('pdbid_all_fasta', df_predict)
    else:
        print("Skipping Step 2: No valid predictions available.")
