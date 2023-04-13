from rdkit import Chem
from rdkit.Chem import rdChemReactions


def canonicalize_smiles(smiles: str, remove_explicit_H: bool = False) -> str:
    """
    Canonicalize a SMILES string.

    Removes any atom-mapping numbers. Optionally, removes explicit Hs.
    """
    mol = Chem.MolFromSmiles(smiles)
    for a in mol.GetAtoms():
        if a.HasProp("molAtomMapNumber"):
            a.ClearProp("molAtomMapNumber")
    if remove_explicit_H:
        mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)


def remove_mapno_from_reaction(rxn: rdChemReactions.ChemicalReaction) -> None:
    """
    Remove all atom-mapping information from a reaction. Operates inplace.
    """
    for ri in range(rxn.GetNumReactantTemplates()):
        rt = rxn.GetReactantTemplate(ri)
        for atom in rt.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
    for pi in range(rxn.GetNumProductTemplates()):
        pt = rxn.GetProductTemplate(pi)
        for atom in pt.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
    return
