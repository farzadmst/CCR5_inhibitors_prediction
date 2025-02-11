import glob
import os
import pandas as pd
from padelpy import padeldescriptor


def generate_fingerprints(mol_file="data/molecule.smi", output_csv="data/PubChem.csv", fp_type="PubChem"):
    """Generates molecular fingerprints using PaDEL."""

    xml_files = glob.glob("descriptors/*.xml")
    xml_files.sort()

    fingerprint_dict = dict(AtomPairs2DCount='descriptors/AtomPairs2DFingerprintCount.xml',
                            AtomPairs2D='descriptorsAtomPairs2DFingerprinter.xml', EState='descriptors/EStateFingerprinter.xml',
                            CDKextended='descriptors/ExtendedFingerprinter.xml', CDK='descriptors/Fingerprinter.xml',
                            CDKgraphonly='descriptors/GraphOnlyFingerprinter.xml',
                            KlekotaRothCount='descriptors/KlekotaRothFingerprintCount.xml',
                            KlekotaRoth='descriptors/KlekotaRothFingerprinter.xml', MACCS='descriptors/MACCSFingerprinter.xml',
                            PubChem='descriptors/PubchemFingerprinter.xml',
                            SubstructureCount='descriptors/SubstructureFingerprintCount.xml',
                            Substructure='descriptors/SubstructureFingerprinter.xml')

    if fp_type not in fingerprint_dict:
        raise ValueError("Invalid fingerprint type. Choose from: " + ", ".join(fingerprint_dict.keys()))

    padeldescriptor(mol_dir=mol_file,
                    d_file=output_csv,
                    descriptortypes=fingerprint_dict[fp_type],
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    removesalt=True,
                    log=True,
                    fingerprints=True)

    print(f"Fingerprints saved at: {output_csv}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    generate_fingerprints()
