"""
数据预处理：
    1. 读取openmx输出的 *.dat 文件和 HS.out 文件
    2. 确定local坐标系
    3. 每个样本是一个graph，其中每条edge对应一个hopping矩阵（13×13）
Last Modified: 2022/8/30 By Mao Su
"""
import copy
import time, os
import torch
from torch import arccos, arctan
from torch.linalg import norm
from scipy.special import sph_harm
from utils.trans_ham import local_ham, get_rotation_matrix
from openmx_io import init_data, get_HS_data

sqrt2 = 1.4142135623730951

rcut = 6.0

# spherical harmonics
l_max = 5
num_ylm = sum([2 * m + 1 for m in range(l_max)])
ll, mm = [], []
for i in range(l_max):
    l = [i] * (2 * i + 1)
    m = [x for x in range(-i, i+1)]
    ll.extend(l)
    mm.extend(m)
ll = torch.tensor(ll)
mm = torch.tensor(mm)
nn = torch.ones((num_ylm,), dtype=torch.complex64)
for i in range(num_ylm):
    if mm[i] < 0:
        nn[i] = sqrt2 * (-1.0) ** mm[i] * -1j
    if mm[i] > 0:
        nn[i] = sqrt2 * (-1.0) ** mm[i]


def gaussian_expansion(inputs=None, rcut=7.0, nfeat_bond=128, width2=0.0044):
    centers = torch.linspace(0, rcut+0.1, nfeat_bond)
    results = torch.exp(-((inputs - centers) ** 2) / width2)
    return results


def find_v2(pos, v1, offsite=False):
    # 已经确定 v1，从 pos 里面找 v2
    # 首先计算把 v1 转到 x 轴的转动矩阵
    if v1[0] == 0 and v1[1] == 0:
        # v1在z轴方向，绕y轴转90度
        rot_1 = torch.eye(3, dtype=torch.float64)
        if v1[2] > 0:
            rot_2 = torch.tensor([
                [0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0]
            ], dtype=torch.float64)
        else:
            rot_2 = torch.tensor([
                [0, 0, -1],
                [0, 1, 0],
                [1, 0, 0]
            ], dtype=torch.float64)
        v1_rot = rot_2 @ rot_1 @ v1
    else:
        # 先绕z轴转
        r1 = copy.deepcopy(v1)
        r1[2] = 0  # v1在xy面的投影
        sin_1 = -r1[1] / norm(r1)  # sin(2*pi - \theta) = -sin(\theta)
        cos_1 = r1[0] / norm(r1)
        rot_1 = torch.tensor([
            [cos_1, -sin_1, 0],
            [sin_1, cos_1, 0],
            [0, 0, 1]
        ])
        r2 = rot_1 @ v1
        # 再绕y轴转
        sin_2 = r2[2] / norm(r2)
        cos_2 = r2[0] / norm(r2)
        rot_2 = torch.tensor([
            [cos_2, 0, sin_2],
            [0, 1, 0],
            [-sin_2, 0, cos_2]
        ])
        v1_rot = rot_2 @ r2

    assert (abs(v1_rot[1]) + abs(v1_rot[2]) < 1e-10).all(), f'{v1_rot}'
    assert v1_rot[0] > 0, f'{v1_rot}'

    a_max = -999
    vv = torch.tensor([3.4, 4.5, 5.6], dtype=torch.float64)
    for i, p in enumerate(pos):
        p2 = rot_2 @ rot_1 @ p
        # 找和向量 vv 夹角最小的 p2
        res = (p2 * vv / norm(p2) / norm(vv)).sum(dim=-1)
        if res > a_max:
            v2 = p
            a_max = res

    return (v2, a_max) if offsite else v2


def get_lc_onsite(data, j):
    nbr_1 = [data.nbr[j][1].item()]  # 最近邻的序号
    k1 = 2
    while torch.isclose(data.nbr_dist[j][k1], data.nbr_dist[j][1]):
        nbr_1.append(data.nbr[j][k1].item())
        k1 += 1
    if len(nbr_1) == 1:
        # 只有一个最近邻，那么 v1 能直接确定
        v1 = data.pos[nbr_1[0]] - data.pos[j]
        nbr_2 = [data.nbr[j][2].item(), data.nbr[j][3].item()]  # 次近邻，在这里面找v2
        assert torch.isclose(data.nbr_dist[j][2], data.nbr_dist[j][3])
        k2 = 4
        while torch.isclose(data.nbr_dist[j][2], data.nbr_dist[j][k2]):
            nbr_2.append(data.nbr[j][k2].item())
            k2 += 1
    else:
        pos = data.pos[nbr_1] - data.pos[j]
        idx_x = torch.argsort(pos[:, 0], descending=True)
        pos = pos[idx_x]  # 按照 x 分量大小降序排序
        # pos_x = [pos[0]]  # x 分量最大且相等，再比较 y 分量
        for i in range(pos.size(0)-1):
            if not torch.isclose(pos[i][0], pos[i+1][0]):
                pos[i+1] = torch.tensor([-999, -999, -999])
        if pos[1][0] == -999:
            v1 = pos[0]
            idx_max = idx_x[0]
        else:
            idx_y = torch.argsort(pos[:, 1], descending=True)
            pos = pos[idx_y]
            for i in range(pos.size(0)-1):
                if not torch.isclose(pos[i][1], pos[i+1][1]):
                    pos[i+1] = torch.tensor([-999, -999, -999])
            if pos[1][1] == -999:
                v1 = pos[0]
                idx_max = idx_x[idx_y[0]]
            else:
                idx_z = torch.argsort(pos[:, 2], descending=True)
                pos = pos[idx_z]
                assert not torch.isclose(pos[0][2], pos[1][2])
                v1 = pos[0]
                idx_max = idx_x[idx_y][idx_z[0]]

        # 确定 v1 以后，把这个邻居弹出列表，v2 在剩下的邻居里找
        nbr_1.pop(idx_max)
        nbr_2 = nbr_1

    # 找 v2
    if len(nbr_2) == 1:
        v2 = data.pos[nbr_2[0]] - data.pos[j]
    else:
        v2 = find_v2(data.pos[nbr_2] - data.pos[j], v1)
    return v1, v2


def build_graph(data):
    # debug
    # torch.save(data, 'datatest.pt')

    edge_attr = gaussian_expansion(data.edge_dist)
    data.edge_attr = torch.as_tensor(edge_attr, dtype=torch.float32)

    edge_index_list = data.edge_index.t().tolist()
    data.rot_mat = torch.zeros((data.num_edges, 3, 3), dtype=torch.float64)
    data.wigner_D = torch.zeros((data.num_edges, 13, 13), dtype=torch.complex128)
    data.rev = torch.tensor([False] * data.num_edges)  # 如果按 i (source) 建立局域坐标，就标记为False
    for idx, (i, j) in enumerate(edge_index_list):
        if i == j:
            if torch.isclose(data.nbr_dist[j][1], data.nbr_dist[j][2]) or \
                    torch.isclose(data.nbr_dist[j][2], data.nbr_dist[j][3]):
                v1, v2 = get_lc_onsite(data, j)
            else:
                assert data.nbr_dist[j][1] < data.nbr_dist[j][2] < data.nbr_dist[j][3]
                k1, k2 = data.nbr[j][1], data.nbr[j][2]
                v1 = data.pos[k1] - data.pos[j]
                v2 = data.pos[k2] - data.pos[j]
        else:
            # 先判断能否正常处理
            i_img = data.pbc_index[i]
            k2_i = data.nbr[i_img][1]
            k2_j = data.nbr[j][1]
            nearest_i = norm(data.pos[k2_i] - data.pos[i_img])
            nearest_j = norm(data.pos[k2_j] - data.pos[j])
            flag = True
            if torch.isclose(nearest_i, nearest_j):
                nbr_i = []
                nbr_j = []
                v1_ij = data.pos[i] - data.pos[j]
                v1_ji = data.pos[j] - data.pos[i]
                k = 1
                while torch.isclose(data.nbr_dist[i_img][k], data.nbr_dist[i_img][1]):
                    if (abs(torch.cross(data.pos[data.nbr[i_img][k]] - data.pos[i_img], v1_ji)) > 1e-1).any():
                        nbr_i.append(data.nbr[i_img][k].item())
                    k += 1
                if not nbr_i:
                    kk = k
                    while torch.isclose(data.nbr_dist[i_img][kk], data.nbr_dist[i_img][k]):
                        nbr_i.append(data.nbr[i_img][kk].item())
                        kk += 1
                k = 1
                while torch.isclose(data.nbr_dist[j][k], data.nbr_dist[j][1]):
                    if (abs(torch.cross(data.pos[data.nbr[j][k]] - data.pos[j], v1_ij)) > 1e-1).any():
                        nbr_j.append(data.nbr[j][k].item())
                    k += 1
                if not nbr_j:
                    kk = k
                    while torch.isclose(data.nbr_dist[j][kk], data.nbr_dist[j][k]):
                        nbr_j.append(data.nbr[j][kk].item())
                        kk += 1

                if len(nbr_i) > 1 and len(nbr_j) == 1:  # 按 j 建立局域坐标
                    nearest_j = torch.tensor(0, dtype=torch.float64)
                    k2_j = nbr_j[0]
                elif len(nbr_i) == 1 and len(nbr_j) > 1:  # 按 i 建立局域坐标
                    nearest_i = torch.tensor(0, dtype=torch.float64)
                    k2_i = nbr_i[0]
                else:
                    # 分别计算 v1 取 ij 或 ji，计算两种情况下的 y_max
                    v2_ij, y_max_ij = find_v2(data.pos[nbr_j] - data.pos[j], v1_ij, offsite=True)
                    v2_ji, y_max_ji = find_v2(data.pos[nbr_i] - data.pos[i_img], v1_ji, offsite=True)
                    if torch.isclose(y_max_ij, y_max_ji):
                        if torch.isclose(v1_ij[0], v1_ji[0]):
                            if torch.isclose(v1_ij[1], v1_ji[1]):
                                assert not torch.isclose(v1_ij[2], v1_ji[2])
                                rev = False if v1_ij[2] > v1_ji[2] else True
                            else:
                                rev = False if v1_ij[1] > v1_ji[1] else True
                        else:
                            rev = False if v1_ij[0] > v1_ji[0] else True
                    else:
                        rev = False if y_max_ij > y_max_ji else True
                    if not rev:
                        v1 = v1_ij
                        v2 = v2_ij
                    else:
                        v1 = v1_ji
                        v2 = v2_ji
                        data.rev[idx] = True
                    flag = False

            if flag:
                assert not torch.isclose(nearest_i, nearest_j), f'{nearest_i}'
                if nearest_i > nearest_j:
                    # 按 j 建立局域坐标
                    v1 = data.pos[i] - data.pos[j]
                    v2 = data.pos[k2_j] - data.pos[j]
                elif nearest_i < nearest_j:
                    # 按 i 建立局域坐标
                    v1 = data.pos[j] - data.pos[i]
                    v2 = data.pos[k2_i] - data.pos[i_img]
                    data.rev[idx] = True

        data.rot_mat[idx], data.wigner_D[idx] = get_rotation_matrix(v1, v2)

    data.y = local_ham(data)  # 转动到local坐标

    herm_loss = abs(data.y - data.y[data.index_rev].transpose(1, 2)).max().item()
    # herm_loss = abs(data.ham - data.ham[data.index_rev].transpose(1, 2)).max().item()
    assert herm_loss < 1e-5, f'herm loss too large: {herm_loss}'

    data.ylm = ylm_new(data)
    return data


def ylm_new(data):
    num_nbr_max = max([n.shape[0] for n in data.nbr])
    ylm_j = torch.zeros((data.num_edges, num_nbr_max, 25), dtype=torch.float32)
    ylm_i = torch.zeros((data.num_edges, num_nbr_max, 25), dtype=torch.float32)
    for idx, (j, i) in enumerate(data.edge_index.t()):
        j = data.pbc_index[j]
        weight = 1.0 / (data.edge_dist[idx] + 1.0)  # 距离的倒数作为权重
        for k, ki in enumerate(data.nbr[i]):
            if k == 0:
                continue
            else:
                ylm_i[idx][k] = local_Ylm(data, idx, ki, i) * weight
        for k2, kj in enumerate(data.nbr[j]):
            if k2 == 0:
                continue
            else:
                ylm_j[idx][k2] = local_Ylm(data, idx, kj, j) * weight
    return ylm_j, ylm_i


def local_Ylm(data, idx_ij, source, target):
    """
    根据(i,j)定义局域坐标，返回(source,target)对应的球谐函数Y_ik^ij(0 <= l <= 4 共 25 个)
    """
    R = data.rot_mat[idx_ij]  # rotate a vector: v_rot = torch.matmul(v_test, R)
    vec = data.pos[target] - data.pos[source]
    
    vec_rot = torch.matmul(R, vec)

    r = norm(vec_rot)
    theta = arccos(vec_rot[2] / r)
    phi = arctan(vec_rot[1] / vec_rot[0]) if vec_rot[0] != 0.0 else 1

    ylm_ijk = sph_harm(abs(mm), ll, phi, theta)
    ylm_ijk = (ylm_ijk * nn).real

    ylm_ijk = torch.as_tensor(ylm_ijk, dtype=torch.float32)
    return ylm_ijk


def dump_Ylm(hs_file=None, symbol=None, dataset_dir=None, load_data=None):
    print(f'Processing file {hs_file}...')
    dat_file = hs_file[:-6] + f'{symbol}.dat#'
    i = hs_file.split('/')[-2]
    if os.path.isfile(dat_file) and os.path.isfile(hs_file):
        if load_data:
            data = torch.load(load_data)
        else:
            data = init_data(dat_file, rcut=rcut)
            data = get_HS_data(data, hs_file, rcut=rcut)
        data = build_graph(data)
        torch.save(data, dataset_dir + f'{i}.pt')
        print(f'Saved data {i}')
    else:
        print(f'Dropped data: {i}')


def dump_Ylm_mp(files, nproc=12, symbol=None, dataset_dir=None):
    """
    并行处理数据。测试发现12个进程一起算效率最高，再多就变慢了。
    """
    import multiprocessing as mp
    print('Processors number =', nproc)
    p = mp.Pool(nproc)
    for file in files:
        p.apply_async(dump_Ylm, (file, symbol, dataset_dir))
    p.close()
    p.join()


def main(sc=None):
    from glob import glob
    t1 = time.time()
    hs_dir = '/data/home/sumao/openmx_graphene/perf/calc_1/'  # openmx计算文件的位置
    dataset = './dataset/perf/'  # 处理后的数据保存的位置
    if not os.path.isdir(dataset):
        os.mkdir(dataset)

    hs_files = glob(hs_dir + '[0-9]*' + '/HS.out')
    print('Total files:', len(hs_files))
    # exit()

    symbol = 'graphene'
    # dump_Ylm(hs_files[0], symbol, 'datatest_', load_data=None)
    dump_Ylm_mp(hs_files, nproc=12, symbol=symbol, dataset_dir=dataset)
    print('Time used:', time.time() - t1)


if __name__ == '__main__':
    main()


