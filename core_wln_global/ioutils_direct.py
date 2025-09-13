import rdkit.Chem as Chem
from core_wln_global.mol_graph import bond_fdim, bond_features
import numpy as np

BOND_TYPE = ["NOBOND", Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC] 
N_BOND_CLASS = len(BOND_TYPE)
binary_fdim = 4 + bond_fdim
INVALID_BOND = -1

def get_bin_feature(r, max_natoms):
    '''
    This function is used to generate descriptions of atom-atom relationships, including
    the bond type between the atoms (if any) and whether they belong to the same molecule.
    It is used in the global attention mechanism.
    '''
    try:
        comp = {}
        for i, s in enumerate(r.split('.')):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                return None
            for atom in mol.GetAtoms():
                comp[atom.GetIntProp('molAtomMapNumber') - 1] = i
                
        n_comp = len(r.split('.'))
        rmol = Chem.MolFromSmiles(r)
        if rmol is None:
            return None
            
        n_atoms = rmol.GetNumAtoms()
        bond_map = {}
        for bond in rmol.GetBonds():
            a1 = bond.GetBeginAtom().GetIntProp('molAtomMapNumber') - 1
            a2 = bond.GetEndAtom().GetIntProp('molAtomMapNumber') - 1
            bond_map[(a1,a2)] = bond_map[(a2,a1)] = bond
            
        features = []
        for i in range(max_natoms):
            for j in range(max_natoms):
                f = np.zeros((binary_fdim,))
                if i >= n_atoms or j >= n_atoms or i == j:
                    features.append(f)
                    continue
                if (i,j) in bond_map:
                    bond = bond_map[(i,j)]
                    f[1:1+bond_fdim] = bond_features(bond)
                else:
                    f[0] = 1.0
                f[-4] = 1.0 if comp[i] != comp[j] else 0.0
                f[-3] = 1.0 if comp[i] == comp[j] else 0.0
                f[-2] = 1.0 if n_comp == 1 else 0.0
                f[-1] = 1.0 if n_comp > 1 else 0.0
                features.append(f)
        return np.vstack(features).reshape((max_natoms,max_natoms,binary_fdim))
    except:
        return None

bo_to_index  = {0.0: 0, 1:1, 2:2, 3:3, 1.5:4}
nbos = len(bo_to_index)
def get_bond_label(r, e, max_natoms):
    mol = Chem.MolFromSmiles(r)
    if mol is None:
        raise ValueError(f"无法解析SMILES: {r}")
        
    n_atoms = mol.GetNumAtoms()
    if n_atoms > max_natoms:
        raise ValueError(f"原子数 ({n_atoms}) 超过最大值 ({max_natoms})")

    # 初始化标签矩阵
    rmap = np.zeros((max_natoms, max_natoms, nbos))
    labels = []
    sp_labels = []

    # 处理编辑标签
    for s in e.split(';'):
        if not s: continue
        
        try:
            # 解析编辑格式
            parts = s.split('-')
            if len(parts) == 3:
                a1, a2, bo = parts
                bo = float(bo)
            elif len(parts) == 2:
                a1, a2 = parts
                bo = 1.0
            else:
                continue
                
            # 转换为索引
            x = min(int(a1)-1, int(a2)-1)
            y = max(int(a1)-1, int(a2)-1)
            z = bo_to_index[bo]
            
            # 设置标签
            rmap[x,y,z] = rmap[y,x,z] = 1

        except Exception as ex:
            print(f"警告: 处理编辑标签时出错: {s}")
            continue

    # 转换为一维标签列表
    for i in range(max_natoms):
        for j in range(max_natoms):
            for k in range(nbos):
                if i == j or i >= n_atoms or j >= n_atoms:
                    labels.append(INVALID_BOND)  # mask
                else:
                    labels.append(rmap[i,j,k])
                    if rmap[i,j,k] == 1:
                        sp_labels.append(i * max_natoms * nbos + j * nbos + k)

    return np.array(labels), sp_labels

def get_all_batch(re_list):
    mol_list = []
    max_natoms = 0
    
    # 第一遍：收集有效分子并计算max_natoms
    valid_mols = []
    for r, e in re_list:
        try:
            rmol = Chem.MolFromSmiles(r)
            if rmol is not None:
                valid_mols.append((r, e, rmol))
                if rmol.GetNumAtoms() > max_natoms:
                    max_natoms = rmol.GetNumAtoms()
        except Exception as ex:
            print(f"无法将 SMILES 字符串 {r} 转换为分子对象，错误信息: {str(ex)}")
            continue

            
    if not valid_mols:  # 如果没有有效分子，返回空batch
        return None, None, None
        
    # 第二遍：处理有效分子
    labels = []
    features = []
    sp_labels = []
    
    for r, e, _ in valid_mols:
        try:
            bin_feat = get_bin_feature(r, max_natoms)
            if bin_feat is None:
                continue
                
            l, sl = get_bond_label(r, e, max_natoms)
            features.append(bin_feat)
            labels.append(l)
            sp_labels.append(sl)
        except:
            continue
            
    if not features:  # 如果没有成功处理的分子，返回空batch
        return None, None, None
        
    # 确保所有列表长度一致
    n = len(features)
    while len(sp_labels) < n:
        sp_labels.append([])  # 用空列表填充缺失的sp_labels
        
    return np.array(features), np.array(labels), sp_labels

def get_feature_batch(r_list):
    max_natoms = 0
    for r in r_list:
        rmol = Chem.MolFromSmiles(r)
        if rmol.GetNumAtoms() > max_natoms:
            max_natoms = rmol.GetNumAtoms()

    features = []
    for r in r_list:
        features.append(get_bin_feature(r,max_natoms))
    return np.array(features)
