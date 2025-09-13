from rdkit import Chem
from rdkit.Chem import rdmolops


def is_non_aromatic_carbon_double_bond(bond):

    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()
    return (bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and
            not bond.GetIsAromatic() and
            atom1.GetAtomicNum() == 6 and
            atom2.GetAtomicNum() == 6)


def get_non_linear_adjacent_carbons(mol, atom_idx):

    adjacent_carbons = []
    for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
        if neighbor.GetAtomicNum() == 6 and mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx()).GetBondType() == Chem.rdchem.BondType.SINGLE:
            adjacent_carbons.append(neighbor.GetIdx())
    return adjacent_carbons


def is_stable_structure(mol):

    for atom in mol.GetAtoms():
        bonds = [bond.GetBondType() for bond in atom.GetBonds()]
        if bonds.count(Chem.rdchem.BondType.DOUBLE) > 1:
            return False
    return True


def generate_modified_products(mol):

    products = []
    for bond in mol.GetBonds():
        if is_non_aromatic_carbon_double_bond(bond):
            c0 = bond.GetBeginAtomIdx()
            c1 = bond.GetEndAtomIdx()

            c0_adjacent_carbons = get_non_linear_adjacent_carbons(mol, c0)
            for c_adj in c0_adjacent_carbons:
                new_mol = Chem.RWMol(mol)
                new_bond_0_1 = new_mol.GetBondBetweenAtoms(c0, c1)
                new_bond_0_1.SetBondType(Chem.rdchem.BondType.SINGLE)
                bond_0_adj = new_mol.GetBondBetweenAtoms(c0, c_adj)
                if bond_0_adj.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    bond_0_adj.SetBondType(Chem.rdchem.BondType.DOUBLE)

                    oxygen_idx = new_mol.AddAtom(Chem.Atom(8))
                    new_mol.AddBond(c1, oxygen_idx, Chem.rdchem.BondType.SINGLE)
                    hydrogen_idx = new_mol.AddAtom(Chem.Atom(1))
                    new_mol.AddBond(oxygen_idx, hydrogen_idx, Chem.rdchem.BondType.SINGLE)

                    try:
                        rdmolops.SanitizeMol(new_mol)
                        if is_stable_structure(new_mol):
                            smiles = Chem.MolToSmiles(new_mol)
                            products.append(smiles)
                    except ValueError:
                        pass

            c1_adjacent_carbons = get_non_linear_adjacent_carbons(mol, c1)
            for c_adj in c1_adjacent_carbons:
                new_mol = Chem.RWMol(mol)
                new_bond_0_1 = new_mol.GetBondBetweenAtoms(c0, c1)
                new_bond_0_1.SetBondType(Chem.rdchem.BondType.SINGLE)
                bond_1_adj = new_mol.GetBondBetweenAtoms(c1, c_adj)
                if bond_1_adj.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    bond_1_adj.SetBondType(Chem.rdchem.BondType.DOUBLE)

                    oxygen_idx = new_mol.AddAtom(Chem.Atom(8))
                    new_mol.AddBond(c0, oxygen_idx, Chem.rdchem.BondType.SINGLE)
                    hydrogen_idx = new_mol.AddAtom(Chem.Atom(1))
                    new_mol.AddBond(oxygen_idx, hydrogen_idx, Chem.rdchem.BondType.SINGLE)

                    try:
                        rdmolops.SanitizeMol(new_mol)
                        if is_stable_structure(new_mol):
                            smiles = Chem.MolToSmiles(new_mol)
                            products.append(smiles)
                    except ValueError:
                        pass

    return products


def process_smiles_from_file(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r') as infile:
            smiles_list = infile.read().splitlines()

        all_products = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                products = generate_modified_products(mol)
                all_products.extend(products)

        with open(output_file_path, 'w') as outfile:
            for product in all_products:
                outfile.write(product + '\n')

        print(f"处理完成，结果已保存到 {output_file_path}")
    except FileNotFoundError:
        print(f"错误：未找到文件 {input_file_path}")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == "__main__":
    input_file = "XX.txt"
    output_file = "YY.txt"
    process_smiles_from_file(input_file, output_file)
    