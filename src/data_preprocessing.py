import pandas as pd
import os


def preprocess_data(input_csv="data/HIV_pIC50.csv", output_smi="data/molecule.smi"):
    """Extracts SMILES and molecule IDs from CSV and saves as a .smi file for PaDEL."""

    df = pd.read_csv(input_csv)

    if "canonical_smiles" not in df.columns or "molecule_chembl_id" not in df.columns:
        raise ValueError("CSV file must contain 'canonical_smiles' and 'molecule_chembl_id' columns.")

    df[['canonical_smiles', 'molecule_chembl_id']].to_csv(output_smi, sep='\t', index=False, header=False)

    print(f"SMILES file saved at: {output_smi}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    preprocess_data()
