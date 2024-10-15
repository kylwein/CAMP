import pandas as pd


def validate_lengths(row):
    pep_seq_len = len(row['pep_seq'])
    binding_vec_len = len(row['pep_binding_vec']) if pd.notna(row['pep_binding_vec']) else 0

    if pep_seq_len != binding_vec_len:
        print(f"[Filtered] {row['pdb_id']}: Peptide length={pep_seq_len}, Binding vec length={binding_vec_len}")
        return False  # Mark as invalid
    return True

def analyze_binding_labels(df):
    """Analyze the binding labels in the dataset."""
    num_samples = len(df)
    num_binding_sites = df['pep_binding_vec'].apply(lambda x: x.count('1')).sum()
    avg_binding_sites = num_binding_sites / num_samples if num_samples > 0 else 0

    print(f"Total samples: {num_samples}")
    print(f"Total binding sites: {num_binding_sites}")
    print(f"Average binding sites per peptide: {avg_binding_sites:.2f}")

def prepare_for_training(df_cleaned):
    """Prepare the data for PyTorch model integration."""
    try:
        import torch
    except ImportError:
        print("Error: PyTorch is not installed. Run 'pip install torch' to install it.")
        return

    # Convert binding vectors to PyTorch tensors
    binding_vectors = df_cleaned['pep_binding_vec'].apply(lambda x: torch.tensor([int(i) for i in x]))
    print(f"Sample Binding Vector Tensor:\n{binding_vectors.iloc[0]}")

def main():
    try:
        # Load the valid binding sites dataset
        df = pd.read_csv('valid_bindingsites.csv')
    except FileNotFoundError:
        print("Error: 'valid_bindingsites.csv' not found.")
        return

    # Validate lengths and filter out mismatches
    df_valid = df[df.apply(validate_lengths, axis=1)].copy()

    # Handle missing values, if necessary
    df_valid.fillna('-99999', inplace=True)

    # Analyze binding labels
    print("\n=== Binding Labels Analysis ===")
    analyze_binding_labels(df_valid)

    # Prepare a cleaned DataFrame with relevant columns
    df_cleaned = df_valid[['pdb_id', 'pep_seq', 'pep_binding_vec', 'binding_idx']]
    df_cleaned.to_csv('cleaned_binding_sites.csv', index=False)
    print("\nData cleaning completed. Cleaned file saved as 'cleaned_binding_sites.csv'.")

    # Prepare data for PyTorch training
    print("\n=== Preparing Data for PyTorch Training ===")
    prepare_for_training(df_cleaned)

if __name__ == "__main__":
    main()
