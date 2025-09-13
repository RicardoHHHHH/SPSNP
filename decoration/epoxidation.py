from rdkit import Chem
from rdkit.Chem import rdmolops


def is_non_aromatic_double_bond(bond):

    return bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and not bond.GetIsAromatic()


def add_oxygen_and_form_ring(mol):

    products = []
    aromatic_atom_indices = set()
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) == 6:
            all_aromatic = True
            for atom_idx in ring:
                atom = mol.GetAtomWithIdx(atom_idx)
                if not atom.GetIsAromatic():
                    all_aromatic = False
                    break
            if all_aromatic:
                aromatic_atom_indices.update(ring)

    for bond in mol.GetBonds():
        if is_non_aromatic_double_bond(bond):
            c0 = bond.GetBeginAtomIdx()
            c1 = bond.GetEndAtomIdx()
            if (mol.GetAtomWithIdx(c0).GetAtomicNum() == 6 and 
                mol.GetAtomWithIdx(c1).GetAtomicNum() == 6 and 
                c0 not in aromatic_atom_indices and 
                c1 not in aromatic_atom_indices):
                new_mol = Chem.RWMol(mol)
                o_idx = new_mol.AddAtom(Chem.Atom(8))

                new_mol.GetBondBetweenAtoms(c0, c1).SetBondType(Chem.rdchem.BondType.SINGLE)

                new_mol.AddBond(c0, o_idx, Chem.rdchem.BondType.SINGLE)
                new_mol.AddBond(c1, o_idx, Chem.rdchem.BondType.SINGLE)

                try:
                    rdmolops.SanitizeMol(new_mol)
                    smiles = Chem.MolToSmiles(new_mol)
                    products.append((smiles, "氧原子与 C0C1 成环"))
                except ValueError:
                    pass

    return products


def process_smiles_from_file(input_file, output_file):
    try:
        with open(input_file, 'r') as infile:
            smiles_list = infile.read().splitlines()

        with open(output_file, 'w') as outfile:
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    products = add_oxygen_and_form_ring(mol)
                    line = f"{smiles} "
                    for product_smiles, _ in products:
                        line += f"{product_smiles} "
                    outfile.write(line.rstrip() + '\n')
    except FileNotFoundError:
        print(f"错误: 文件 {input_file} 未找到。")
    except Exception as e:
        print(f"发生未知错误: {e}")


    

if __name__ == "__main__":
    input_file = "XX.txt"
    output_file = "YY.txt"
    process_smiles_from_file(input_file, output_file)
    