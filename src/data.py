import numpy as np
import pandas as pd

from rdkit.Chem import MolFromSmiles, MolToSmiles


class SLAPData:
    """
    Container for a SLAP data set.

    Implements methods to load data and split reactionSMILES.
    """

    def __init__(self, file):
        """
        Args:
            file: Path to file to load data from
        """
        self.all_index = None
        self.all_X = None
        self.all_y = None
        self.groups = None
        self.file = file
        self.from_cache = None
        self.path = None

    def load_data_from_file(self):
        """Extracts data from CSV file.

        We expect to receive a CSV with a column 'reactionSMILES' xor 'SMILES' and a column 'target' xor 'label'.
        """
        data_df = pd.read_csv(self.file)
        # we perform minuscule preprocessing here
        try:
            self.all_X = data_df["reactionSMILES"].values
        except KeyError:
            self.all_X = data_df["SMILES"].values
        try:
            self.all_y = data_df["target"].values
        except KeyError:
            self.all_y = data_df["label"].values
        self.all_index = np.arange(len(self.all_X))
        assert self.all_X.shape[0] == self.all_y.shape[0]
        assert self.all_X.shape[0] == self.all_index.shape[0]

    def split_reaction_smiles(self):
        """Splits reactionSMILES into reactants and products (and assigns some attributes based on SMILES).

        Uses the reactionSMILES in self.all_X, splits them into the individual molecules, removes any atom-mapping
        information, and saves a np.ndarray of shape (len(all_X), 4) in self.all_X.
        The array contains [reaction_SMILES, reactant1_SMILES, reactant2_SMILES, product_SMILES] along axis 1.

        Further, sets the attribute .groups on SLAPData, based on reactant1_SMILES and reactant2_SMILES.

        Returns:
            None
        """
        new_x = np.zeros(
            (len(self.all_X), 4), object
        )  # 4 for 1 reaction_smiles + 2 reactants + 1 product
        new_x[:, 0] = self.all_X
        single_smiles = []
        for reaction_smiles in self.all_X:
            reactants_and_product = reaction_smiles.split(">>")
            product = reactants_and_product[1]
            reactants = reactants_and_product[0].split(".")
            # remove the mapping
            unmapped_smiles = []
            for smiles in reactants + [
                product,
            ]:
                m = MolFromSmiles(smiles)
                for atom in m.GetAtoms():
                    if atom.HasProp("molAtomMapNumber"):
                        atom.ClearProp("molAtomMapNumber")
                unmapped_smiles.append(MolToSmiles(m))
            single_smiles.append(unmapped_smiles)
        new_x[:, 1:] = single_smiles
        self.all_X = new_x
        groups1 = np.unique(self.all_X[:, 1], return_inverse=True)[1]
        groups2 = np.unique(self.all_X[:, 2], return_inverse=True)[1]
        self.groups = np.array([groups1, groups2]).transpose()
