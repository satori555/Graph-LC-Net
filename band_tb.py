from ase.atoms import Atoms
import numpy as np
import torch
from math import floor
from scipy.linalg import eigvalsh
import matplotlib.pyplot as plt
from matplotlib import gridspec
from openmx_io import read_dat
import time, copy


class TB_model(object):
    def __init__(self, lat=None, pos=None, data=None, hs_file=None):
        self._lat = np.array(lat).astype(np.float32)
        self._reciprocal = np.array(lat.reciprocal()).astype(np.float32) * 2 * np.pi
        self._pos = np.array(pos, dtype=np.float32)
        self.num_orb = 13
        self.num_nbr = 85
        self.Rn = []
        self.ham = []
        self.ham_pred = []
        self.olp = []
        if hs_file:
            self.get_ham_olp(hs_file)
        if data:
            self.get_data(data)

    def get_ham_olp(self, file='HS.out'):
        with open(file) as f:
            content = f.readlines()
        idx = 0

        pos = []  # 全局坐标
        nbr = []  # 邻居
        edge_index_list = []  # 边
        edge_dist = []  # 每条边的长度

        while not content[idx].startswith('Kohn-Sham Hamiltonian'):
            idx += 1
        while content[idx + 1].strip():
            tmp = content[idx + 1].split()  # 读取global index
            idx_i, idx_j = int(tmp[1][6:]), int(tmp[4][8:-1])

            tmp = content[idx + 2].split()  # 读取坐标信息
            dist = float(tmp[1])
            vec_i = [float(d) * 0.529177249 + 0.5 for d in tmp[3:6]]  # 单位换成Ang
            vec_j = [float(d) * 0.529177249 + 0.5 for d in tmp[7:]]  # 全部加0.5防止原子跑出盒子

            r = self.cart2dir(self._lat, vec_j).tolist()
            r.extend([idx_i, idx_j])
            self.Rn.append([floor(d) for d in r])  # self.Rn保存每一条边的超胞和原子index

            if vec_i in pos:
                i = pos.index(vec_i)
            else:
                pos.append(vec_i)
                nbr.append([])
                i = len(pos) - 1

            if vec_j in pos:
                j = pos.index(vec_j)
            else:
                pos.append(vec_j)
                nbr.append([])
                j = len(pos) - 1

            edge_index_list.append([j, i])  # 添加一条从j指向i的边
            edge_dist.append([dist])  # 记录距离

            nbr[i].append([j, dist])  # 节点i增加一个邻居，距离用来排序

            # 读取hamiltonian
            ham_ij = []
            for i in range(self.num_orb):
                h = [float(d) for d in content[idx + 3 + i].split()]
                ham_ij.append(h)
            self.ham.append(ham_ij)
            idx += self.num_orb + 2

        while not content[idx].startswith('Overlap'):
            idx += 1
        while content[idx + 1].strip():
            # 读取Overlap
            olp_ij = []
            for i in range(self.num_orb):
                o = [float(d) for d in content[idx + 3 + i].split()]
                olp_ij.append(o)
            self.olp.append(olp_ij)
            idx += self.num_orb + 2

        nbr = [sorted(l, key=lambda x: x[-1]) for l in nbr]  # sorted by egde length
        nbr = [[x[0] for x in y] for y in nbr]  # remove distances

        for i, n in enumerate(nbr):
            nbr[i] = n[:self.num_nbr]  # 保留num_nbr个邻居
            for j in n[self.num_nbr:]:
                idx_rm = edge_index_list.index([j, i])  # 去掉对应的边
                edge_index_list.pop(idx_rm)
                edge_dist.pop(idx_rm)
                self.Rn.pop(idx_rm)
                self.ham.pop(idx_rm)
                self.olp.pop(idx_rm)

    def get_data(self, data):
        edge_list = data.edge_index.t().tolist()
        # natom = data.nbr.shape[0]
        # data.pos += 0.5  # 防止原子跑出盒子
        # assert (0 <= data.pos[:natom]).all()
        # for i in range(3):
        #     assert (data.pos[:natom, i] < self._lat[i, i]).all()

        for j, i in edge_list:
            vec_j, vec_i = data.pos[j], data.pos[i]

            r = self.cart2dir(self._lat, vec_j).tolist()
            r.extend([i, data.pbc_index[j]])
            self.Rn.append([floor(d) for d in r])  # self.Rn保存每一条边的超胞和原子index
        self.ham_pred = data.ham_pred
        self.ham = data.ham  # / 27.2113845
        self.olp = data.olp

    def solve(self, k_list=None, pred=True):
        ham = self.ham_pred if pred else self.ham
        Rn_list = list(set(tuple(tmp[:3]) for tmp in self.Rn))  # 找到所有Rn

        nkp = len(k_list)
        k_list = copy.deepcopy(k_list)
        for i in range(nkp):
            k_list[i] = np.sum(k_list[i] * self._reciprocal, axis=0)

        nhs = self._pos.shape[0] * self.num_orb  # H和S矩阵的size
        h = np.zeros((len(Rn_list), nhs, nhs), dtype=np.float32)  # h[Rn][ia][jb]
        s = np.zeros_like(h, dtype=np.float32)  # s[Rn][ia][jb]
        ret_eval = np.zeros((nkp, nhs), dtype=np.float32)

        for idx in range(len(ham)):
            # 找到这条边的(Rn, i, j)在h矩阵中的位置
            r0, r1, r2, i, j = self.Rn[idx]
            n = Rn_list.index((r0, r1, r2))
            ia = (i - 1) * self.num_orb
            jb = (j - 1) * self.num_orb
            for ii in range(self.num_orb):
                for jj in range(self.num_orb):
                    h[n][ia + ii][jb + jj] = ham[idx][ii][jj]
                    s[n][ia + ii][jb + jj] = self.olp[idx][ii][jj]

        norm_lat = np.linalg.norm(self._lat, axis=1)
        Rn_list_c = [Rn * norm_lat for Rn in Rn_list]  # Rn的实际长度
        for i, k in enumerate(k_list):
            H = np.zeros((nhs, nhs), dtype=np.complex64)  # H[ia][jb]
            S = np.zeros_like(H, dtype=np.complex64)  # s[ia][jb]
            for n, Rn in enumerate(Rn_list_c):
                fac = np.exp(1j * np.dot(Rn, k))
                H += fac * h[n]
                S += fac * s[n]
            v = np.sort(eigvalsh(H, S).real)
            ret_eval[i, :] = v[:]
        return ret_eval

    def k_path(self, kpts, nk):
        # 参考 http://www.physics.rutgers.edu/pythtb/usage.html

        k_list = np.array(kpts)
        n_nodes = k_list.shape[0]
        lat_per = np.copy(self._lat)
        k_metric = np.linalg.inv(np.dot(lat_per, lat_per.T))

        k_node = np.zeros(n_nodes, dtype=np.float32)
        for n in range(1, n_nodes):
            dk = k_list[n] - k_list[n - 1]
            dklen = np.sqrt(np.dot(dk, np.dot(k_metric, dk)))
            k_node[n] = k_node[n - 1] + dklen

        node_index = [0]
        for n in range(1, n_nodes - 1):
            frac = k_node[n] / k_node[-1]
            node_index.append(int(round(frac * (nk - 1))))
        node_index.append(nk - 1)

        k_dist = np.zeros(nk, dtype=np.float32)
        k_vec = np.zeros((nk, 3), dtype=np.float32)

        k_vec[0] = k_list[0]
        for n in range(1, n_nodes):
            n_i = node_index[n - 1]
            n_f = node_index[n]
            kd_i = k_node[n - 1]
            kd_f = k_node[n]
            k_i = k_list[n - 1]
            k_f = k_list[n]
            for j in range(n_i, n_f + 1):
                frac = float(j - n_i) / float(n_f - n_i)
                k_dist[j] = kd_i + frac * (kd_f - kd_i)
                k_vec[j] = k_i + frac * (k_f - k_i)
        return k_vec, k_dist, k_node

    @staticmethod
    def cart2dir(bases, data):
        b = np.array(data).T
        A = np.array(bases).T
        A_I = np.linalg.inv(A)
        return np.matmul(A_I, b).T


def get_band(tb=None, fermi=0.0, fig_name=None):
    k_path = [[0.5, 0.25, 0.75], [0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]
    label = (r'W', r'$\Gamma $', r'$X$')
    (k_vec, k_dist, k_node) = tb.k_path(k_path, 31)

    # evals = tb.solve(k_vec, pred) * 27.2113845 - fermi
    evals_pred = tb.solve(k_vec, pred=True) - fermi
    evals_true = tb.solve(k_vec, pred=False) - fermi

    if fig_name:
        print(f'Plotting band {fig_name} ...')
        fig, ax = plt.subplots()
        ax.set_xlim(k_node[0], k_node[-1])
        ax.set_xticks(k_node)
        ax.set_xticklabels(label, fontsize=15)
        for n in range(len(k_node)):
            ax.axvline(x=k_node[n], linewidth=0.5, color='k')
        ax.axhline(y=0, linestyle='--', color='k')

        ax.set_ylim((-2.5, 2.5))
        ax.set_yticks([-2.0, -1.0, 0.0, 1.0, 2.0])
        ax.set_yticklabels([-2.0, -1.0, 0.0, 1.0, 2.0], fontsize=15)
        ax.plot(k_dist, evals_pred, color='blue', linewidth=1.0)
        ax.plot(k_dist, evals_true, color='red', linewidth=1.0, dashes=[3, 3])
        fig.tight_layout()
        fig.savefig(fig_name)


def get_dos(tb=None, fermi=0.0, fig_name=None, pt_name=None):
    kmesh = (2, 2, 2)
    kpts = []
    for i in range(kmesh[0]):
        for j in range(kmesh[1]):
            for k in range(kmesh[2]):
                kpts.append([float(i) / float(kmesh[0]), float(j) / float(kmesh[1]), float(k) / float(kmesh[2])])

    evals_pred = tb.solve(kpts, pred=True).flatten() - fermi
    evals_true = tb.solve(kpts, pred=False).flatten() - fermi

    if pt_name:
        torch.save(evals_true, pt_name + '_true.pt')
        torch.save(evals_pred, pt_name + '_pred.pt')

    if fig_name:
        print(f'Plotting dos {fig_name}...')

        fig, ax = plt.subplots()
        hists, bins = np.histogram(evals_pred, 100, (-2.5, 2.5))
        ax.plot(bins[1:], hists, color='blue', linewidth=1.0)

        hists, bins = np.histogram(evals_true, 100, (-2.5, 2.5))
        ax.plot(bins[1:], hists, color='red', linewidth=1.0, dashes=[3, 3])

        ax.set_title("density of states")
        ax.set_xlabel("Band energy")
        ax.set_ylabel("Number of states")

        fig.tight_layout()
        fig.savefig(fig_name)


def get_band_dos(tb=None, fermi=0.0, fig_name=None):
    plt.rcParams['axes.linewidth'] = 2.0
    plt.rcParams['xtick.major.width'] = 2.0
    plt.rcParams['ytick.major.width'] = 2.0

    # band
    k_path = [[0.5, 0.25, 0.75], [0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]
    label = (r'W', r'$\Gamma $', r'$X$')
    (k_vec, k_dist, k_node) = tb.k_path(k_path, 31)
    evals_band_pred = tb.solve(k_vec, pred=True) - fermi
    evals_band_true = tb.solve(k_vec, pred=False) - fermi

    # dos
    kmesh = (4, 4, 4)
    kpts = []
    for i in range(kmesh[0]):
        for j in range(kmesh[1]):
            for k in range(kmesh[2]):
                kpts.append([float(i) / float(kmesh[0]), float(j) / float(kmesh[1]), float(k) / float(kmesh[2])])
    evals_dos_pred = tb.solve(kpts, pred=True).flatten() - fermi
    evals_dos_true = tb.solve(kpts, pred=False).flatten() - fermi

    if fig_name:
        ylim = (-2.5, 2.5)
        yts = [-2.0, -1.0, 0.0, 1.0, 2.0]
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        # band
        ax0 = plt.subplot(gs[0])
        ax0.set_xlim(k_node[0], k_node[-1])
        ax0.set_xticks(k_node)
        ax0.set_xticklabels(label, fontsize=18)
        for n in range(len(k_node)):
            ax0.axvline(x=k_node[n], linewidth=1.0, color='k')
        ax0.axhline(y=0, linestyle='--', color='k')
        ax0.set_ylim(ylim)
        ax0.set_yticks(yts)
        ax0.set_yticklabels(yts, fontsize=15)
        ax0.set_ylabel('Energy (eV)', fontsize=20)
        ax0.plot(k_dist, evals_band_pred, color='k', linewidth=1.5)
        ax0.plot(k_dist, evals_band_true, color='red', linewidth=1.5, dashes=[5, 5])
        # dos
        ax1 = plt.subplot(gs[1])
        hists, bins = np.histogram(evals_dos_pred, 100, (-2.5, 2.5))
        ax1.plot(hists, bins[1:], color='k', linewidth=1.5)
        hists, bins = np.histogram(evals_dos_true, 100, (-2.5, 2.5))
        ax1.plot(hists, bins[1:], color='red', linewidth=1.5, dashes=[5, 5])
        ax1.axhline(y=0, linestyle='--', color='k')
        ax1.set_yticks(yts)
        ax1.set_yticklabels((), fontsize=18)
        x_int = hists.max() // 2  # 希望dos横坐标有3个数字
        x_int = np.round(x_int-5, -1)
        xts = hists.max() // x_int * x_int
        xts = np.arange(0, xts+x_int, x_int)
        ax1.set_xticks(xts)
        ax1.set_xticklabels((), fontsize=18)
        # ax1.set_xlabel("DOS (a.u.)", fontsize=18)

        # fig.tight_layout()
        fig.subplots_adjust(left=0.12, bottom=0.12, wspace=0.1, hspace=0.1)
        fig.savefig(fig_name)


def get_fermi(file='si.out'):
    with open(file) as f:
        line = f.readline()
        while line:
            if line.find('Chemical potential') > -1:
                return float(line.split()[-1]) * 27.2113845
            line = f.readline()


def main(sc='222', idx=2, eval=True):
    if eval:
        prefix = f'/data/home/sumao/openmx_sige/supercell/{sc}/calc/{idx}/'
        data = torch.load(f'data_pred/sige_eval_{sc}_{idx}.pt')
    else:
        prefix = f'/data/home/sumao/openmx_sige/calc_{sc}_4.0_pert/{idx}/'
        data = torch.load(f'data_pred/sige_{sc}_pert_{idx}.pt')

    fermi = get_fermi(prefix + 'sige.out')
    print('Chemical potential:', fermi)

    atoms = read_dat(prefix + 'sige.dat#')
    tb = TB_model(lat=atoms.cell, pos=atoms.get_positions(), data=data)

    t1 = time.time()
    get_band_dos(tb, fermi, fig_name=f'band/{sc}.png')
    print('Time:', time.time() - t1)

    # get_band(tb, fermi, fig_name=f'band/{sc}_pred.png')
    # t2 = time.time()
    # print('Band time:', t2 - t1)

    # get_dos(tb, fermi, fig_name=f'band/dos_{sc}_pred.png')
    # get_dos(tb, fermi, pt_name=f'band/dos_{sc}')
    # print('Dos time:', time.time() - t1)


if __name__ == '__main__':
    main(sc='222', idx=2, eval=True)
    # cells = ['119', '221', '222', '224', '333', '444']
    # for cell in cells:
    #     main(sc=cell, idx=2, eval=True)
