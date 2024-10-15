import pandas as pd


def extract_binding_indices(pep_seq, binding_vec):
    """Extract binding indices, with logs for mismatched lengths."""
    if pd.notna(binding_vec):
        # Ensure lengths match; otherwise, log and return '-99999'
        if len(binding_vec) != len(pep_seq):
            print(f"Length Mismatch: Peptide: {pep_seq} (len={len(pep_seq)}) vs "
                  f"Binding Vec: {binding_vec} (len={len(binding_vec)})")
            # Handle padding or trimming as needed
            if len(binding_vec) < len(pep_seq):
                print(f"Padding binding vector for {pep_seq}")
                binding_vec = binding_vec.ljust(len(pep_seq), '0')
            else:
                print(f"Trimming binding vector for {pep_seq}")
                binding_vec = binding_vec[:len(pep_seq)]

        # Extract indices where binding occurs (value '1')
        return ','.join(str(i) for i, c in enumerate(binding_vec) if c == '1')
    return '-99999'


def check_binding_indices(df_train, df_zy_pep):
    """Merge and test binding index extraction."""
    # Merge datasets on pdb_id, pep_chain, and prot_chain
    df_join = pd.merge(
        df_train, df_zy_pep, how='left',
        left_on=['pdb_id', 'pep_chain', 'prot_chain'],
        right_on=['pdb_id', 'pep_chain', 'prot_chain']
    )

    # Log NaN values in 'pep_binding_vec' after merging
    nan_count = df_join['pep_binding_vec'].isna().sum()
    print(f"Number of NaN in pep_binding_vec: {nan_count}")

    # Apply the binding index extraction function
    df_join['binding_idx'] = df_join.apply(
        lambda x: extract_binding_indices(x.pep_seq, x.pep_binding_vec), axis=1
    )

    # Save the valid entries to a CSV for further analysis
    df_valid = df_join[df_join['binding_idx'] != '-99999']
    df_valid.to_csv('valid_bindingsites.csv', index=False, sep=',')

    print("\nBinding Index Extraction Sample:")
    print(df_join[['pep_seq', 'pep_binding_vec', 'binding_idx']].head())


def main():
    # Load input data with appropriate error handling
    try:
        df_train = pd.read_csv(
            r'C:\Users\kwcoo\Documents\GitHub\CAMP\train_PDB_data_pos',
            header=0, sep='#'
        )
        df_zy_pep = pd.read_csv(
            './peptide-mapping (1).txt', header=None, sep='\t'
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Set column names for the peptide mapping file
    df_zy_pep.columns = ['bdb_id', 'bdb_pep_seq', 'pep_binding_vec']

    # Extract pdb_id, pep_chain, and prot_chain from bdb_id
    df_zy_pep['pdb_id'] = df_zy_pep.bdb_id.apply(lambda x: x.split('_')[0])
    df_zy_pep['pep_chain'] = df_zy_pep.bdb_id.apply(lambda x: x.split('_')[1].lower())
    df_zy_pep['prot_chain'] = df_zy_pep.bdb_id.apply(lambda x: x.split('_')[2].upper())

    print("=== Binding Index Extraction ===")
    check_binding_indices(df_train, df_zy_pep)


if __name__ == "__main__":
    main()
