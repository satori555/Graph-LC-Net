import ase.geometry
import torch
from torch_geometric.data import Data

import numpy as np
from ase.atoms import Atoms
from ase.io import read
from ase.neighborlist import NeighborList

import time, copy

num_orb = 13  # s2p2d1
# rcut = 7.0  # neighbour cutoff (Ang)
# rcut_bohr = rcut / 0.529177249


def read_dat(file='si.dat'):
    """
    Get structure information from .dat file.
    """
    natom = 0
    cell, positions, types = [], [], []
    pos_unit = None
    cell_unit = None
    with open(file) as f:
        line = f.readline()
        while line:
            if line.startswith('Atoms.Number'):
                natom = int(line.split()[-1])
            if line.startswith('<Atoms.SpeciesAndCoordinates'):
                for _ in range(natom):
                    line = f.readline()
                    types.append(line.split()[1])
                    pos = [float(d) for d in line.split()[2:5]]
                    positions.append(pos)
            if line.startswith('<Atoms.UnitVectors'):
                for _ in range(3):
                    line = f.readline()
                    c = [float(d) for d in line.split()]
                    cell.append(c)
            if line.startswith('Atoms.SpeciesAndCoordinates.Unit'):
                pos_unit = line.split()[1]
            if line.startswith('Atoms.UnitVectors.Unit'):
                cell_unit = line.split()[1]
            line = f.readline()
    cell = np.array(cell)
    positions = np.array(positions)
    if pos_unit.upper() == 'FRAC':
        positions = np.dot(positions, cell)
    elif pos_unit.upper() == 'AU':
        positions = positions * 0.529177249
    elif pos_unit.upper() != 'ANG':
        raise Exception('check unit')
    if cell_unit.upper() == 'AU':
        cell = cell * 0.529177249
    elif cell_unit.upper() != 'ANG':
        raise Exception('check unit')
    # return natom, cell, positions, types
    atoms = Atoms(types, cell=cell, positions=positions, pbc=True)
    return atoms


def init_data(file=None, rcut=7.0):
    """
    根据原子坐标构建Graph
    """
    atoms = read_dat(file)
    natom = len(atoms)
    pos_0 = copy.deepcopy(atoms.positions)

    # 直接 wrap 会导致拓扑结构和HS.out文件不一样，所以把原子整体平移到盒子中心
    # 运行 OpenMX 之前保证所有原子都在盒子里
    atoms.center()
    translation = atoms.positions[0] - pos_0[0]
    pos_arr = atoms.positions

    nl = NeighborList(cutoffs=[rcut/2.0] * natom, skin=0.0, self_interaction=False, bothways=True)
    nl.update(atoms)

    # 初始化pos，规定第一个neighbor是自己
    pos = pos_arr.tolist()
    pbc_index = list(range(natom))  # index within box
    pbc_image = [[0, 0, 0] for _ in range(natom)]  # periodic image
    atoms_numbers = atoms.numbers.tolist()
    nbr_dist = [[[i, 0.0]] for i in range(natom)]

    for i in range(natom):
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            pos_j = pos_arr[j] + offset @ atoms.get_cell()  # source j 的global坐标
            dist = np.linalg.norm(pos_j - pos_arr[i])
            pos_j = pos_j.tolist()
            if pos_j in pos:
                idx_j = pos.index(pos_j)
            else:
                pos.append(pos_j)
                pbc_index.append(j)
                pbc_image.append(offset.tolist())
                atoms_numbers.append(atoms.numbers[j])
                idx_j = pos.index(pos_j)
            nbr_dist[i].append([idx_j, dist])

    nbr_dist = [sorted(l, key=lambda x:x[-1]) for l in nbr_dist]  # sorted by egde length
    
    # 邻居的数量可以不一致
    nbr = [[x[0] for x in y] for y in nbr_dist]  # remove distances

    edge_dist = [x[1] for y in nbr_dist for x in y]
    edge_index_list = [[j, i] for i in range(natom) for j in nbr[i]]

    nbr_dist = [[x[1] for x in y] for y in nbr_dist]

    data = Data(
        x=torch.as_tensor(atoms_numbers, dtype=torch.int),
        edge_index=torch.as_tensor(edge_index_list, dtype=torch.long).view(-1, 2).t(),
        edge_dist=torch.as_tensor(edge_dist, dtype=torch.float32).view(-1, 1),
        nbr_dist=[torch.tensor(d, dtype=torch.float32) for d in nbr_dist],
        pos=torch.as_tensor(pos, dtype=torch.float64),
        cell=torch.from_numpy(atoms.cell.array),
        translation=torch.from_numpy(translation),
        pbc_index=torch.as_tensor(pbc_index, dtype=torch.long),
        pbc_image=torch.as_tensor(pbc_image, dtype=torch.long),
        ham=torch.zeros((len(edge_index_list), num_orb, num_orb), dtype=torch.float),
        olp=torch.zeros((len(edge_index_list), num_orb, num_orb), dtype=torch.float),
        nbr=[torch.as_tensor(n, dtype=torch.long) for n in nbr]
    )
    return data


def get_HS_data(data, file, rcut=7.0):
    """
    把哈密顿量读取到现有的data里，保留最少的邻居数
    """
    rcut = rcut / 0.529177249
    with open(file) as f:
        content = f.readlines()
    idx = 0
    while not content[idx].startswith('Kohn-Sham Hamiltonian'):
        idx += 1

    edge_index_hs = []  # 保存HS.out文件对应的edge序号
    edge_index_list = data.edge_index.t().tolist()
    while content[idx+1].strip():
        tmp = content[idx+2].split()  # 读取坐标信息
        d_ij = float(tmp[1])
        if d_ij > rcut:
            idx += num_orb + 2
            continue

        vec_i = [float(d) * 0.529177249 for d in tmp[3:6]]
        vec_j = [float(d) * 0.529177249 for d in tmp[7:]]

        vec_i += data.translation.numpy()
        vec_j += data.translation.numpy()

        # 找到这两个向量在pos中的序号
        i = int(content[idx+1].split()[1].split('=')[-1]) - 1
        flag = False
        for j in data.nbr[i]:
            if np.isclose(data.pos[j], vec_j, atol=1e-3).all():
                flag = True
                break
        assert flag, f'{d_ij}'

        edge_idx = edge_index_list.index([j, i])
        edge_index_hs.append(edge_idx)

        # 读取hamiltonian
        ham_ij = []
        for i in range(num_orb):
            h = [float(d) for d in content[idx+3+i].split()]
            ham_ij.append(h)
        ham_ij = torch.tensor(ham_ij, dtype=torch.float) * 27.2113845
        data.ham[edge_idx] = ham_ij
        idx += num_orb + 2

    # 读取overlap
    edge_index_hs = iter(edge_index_hs)
    while not content[idx].startswith('Overlap'):
        idx += 1
    while content[idx+1].strip():
        d_ij = float(content[idx+2].split()[1])
        if d_ij > rcut:
            idx += num_orb + 2
            continue

        olp_ij = []
        for i in range(num_orb):
            o = [float(d) for d in content[idx+3+i].split()]
            olp_ij.append(o)
        edge_idx = next(edge_index_hs)
        data.olp[edge_idx] = torch.tensor(olp_ij)
        idx += num_orb + 2

    index_rev = []
    edge_list = data.edge_index.t().tolist()
    mask = torch.ones(data.num_edges, dtype=torch.bool)
    for idx, (j, i) in enumerate(edge_list):
        if j < len(data.nbr):
            index_rev.append(edge_list.index([i, j]))
            # diff = data.ham[idx] - data.ham[index_rev[-1]].transpose(1, 0)
            # assert (abs(diff) < 1e-5).all()
        else:
            i_rev = data.pbc_index[j].item()
            image_j = data.pbc_image[j]
            flag = False
            for n in data.nbr[i_rev]:
                if (data.pbc_image[n] == -image_j).all() and data.pbc_index[n] == i:
                    j_rev = n.item()
                    index_rev.append(edge_list.index([j_rev, i_rev]))
                    # diff = data.ham[idx] - data.ham[index_rev[-1]].transpose(1, 0)
                    # assert (abs(diff) < 1e-5).all()
                    flag = True
                    break
            if not flag:  # [i, j] 不在 edge_list 里面，把 [j, i] 也去掉
                idx_res = np.argwhere(data.nbr[i].numpy() != j)
                idx_res = torch.from_numpy(idx_res)
                data.nbr[i] = data.nbr[i][idx_res]
                data.nbr_dist[i] = data.nbr_dist[i][idx_res]
                mask[idx] = 0
                assert abs(data.ham[idx]).max() < 1e-4, f'{abs(data.ham[idx]).max()}'

    data.index_rev = torch.tensor(index_rev, dtype=torch.long)

    data.edge_index = data.edge_index[:, mask]
    data.edge_dist = data.edge_dist[mask]
    data.ham = data.ham[mask]
    data.olp = data.olp[mask]
    assert data.index_rev.shape[0] == data.edge_index.shape[1]

    herm_loss = abs(data.ham - data.ham[data.index_rev].transpose(1, 2)).max().item()
    assert herm_loss < 1e-5, f'herm loss too large: {herm_loss}'

    return data


if __name__ == '__main__':
    hs_dir = '/data/home/sumao/openmx_bn/calc/3830/'
    # hs_dir = '/data/home/sumao/openmx_graphene/perf/calc_1/2/'
    data = init_data(hs_dir+'bn.dat#', rcut=7.0)
    data = get_HS_data(data=data, file=hs_dir+'HS.out', rcut=7.0)
    print(data)
