import pandas as pd
import os


def merge_data(fp_csv="data/PubChem.csv", pic50_csv="data/HIV_pIC50.csv", output_csv="data/HIVfinaldataset.csv"):
    """Merges molecular fingerprints with experimental activity values (pIC50)."""

    # Load fingerprints
    descriptors = pd.read_csv(fp_csv)

    # Load pIC50 values
    df_pic50 = pd.read_csv(pic50_csv)

    # Merge data
    dataset = pd.concat([descriptors.drop('Name', axis=1), df_pic50['pIC50']], axis=1)

    # Save
    dataset.to_csv(output_csv, index=False)
    print(f"Final dataset saved at: {output_csv}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    merge_data()
