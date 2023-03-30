"""
哈密顿量的转动，包括：
    实数球谐函数和复数球谐函数之间的变换
    转动矩阵
    Wigner D矩阵的计算。

表象变换：
    a' = S a
    H' = S H S^(-1)

OpenMX轨道顺序：
    s1, s2, px1, py1, pz1, px2, py2, pz2, dz^2, dx^2-y^2, dxy, dxz, dyz
     0,  0,   1,  -1,   0,   1,  -1,   0,    0,        2,  -2,   1,  -1

Last Modidied: 2022/2/22 By Mao Su.
"""
import numpy as np
import torch
from torch import exp, sin, cos, arccos, atan2

PI = 3.141592653589793
sqrt2 = 1.4142135623730951
sqrt38 = 0.6123724356957945

num_orb = 13

# 定义球谐函数real-complex变换矩阵
eye1 = np.eye(1, 1)
eye2 = np.eye(2, 2)

mat_p = np.array([
        [-1, -1j, 0],
        [ 1, -1j, 0],
        [ 0,   0, sqrt2]
    ]) / sqrt2

mat_d = np.array([
        [sqrt2, 0, 0, 0, 0],  # 0
        [0,  1,  1j,  0,   0],  # 2
        [0,  1, -1j,  0,   0],  # -2
        [0,  0,   0, -1, -1j],  # 1
        [0,  0,   0,  1, -1j]   # -1
    ]) / sqrt2

z13 = np.zeros((1, 3))
z23 = np.zeros((2, 3))
z33 = np.zeros((3, 3))
z85 = np.zeros((8, 5))

if num_orb == 4:
    r2c = np.block([[eye1, z13], [z13.T, mat_p]])
elif num_orb == 8:
    r2c = np.block([[eye2, z23, z23],
                    [z23.T, mat_p, z33],
                    [z23.T, z33, mat_p]])
elif num_orb == 13:
    r2c = np.block([[eye2, z23, z23],
                    [z23.T, mat_p, z33],
                    [z23.T, z33, mat_p]])
    r2c = np.block([[r2c, z85], [z85.T, mat_d]])
else:
    raise Exception('...')

c2r = np.conj(r2c)
r2c = r2c.T

c2r = torch.from_numpy(c2r)
r2c = torch.from_numpy(r2c)

# r2c = torch.as_tensor(r2c, dtype=torch.complex64).cuda()
# c2r = torch.as_tensor(c2r, dtype=torch.complex64).cuda()

def get_rotation_matrix(v1, v2):
    x = v1 / torch.linalg.norm(v1)
    y = torch.cross(v1, v2)
    y = y / torch.linalg.norm(y)
    z = torch.cross(x, y)
    z = z / torch.linalg.norm(z)
    R = torch.cat([x[None, :], y[None, :], z[None, :]])

    if abs(R[2][2]) > 1:
        R[2][2] = round(R[2][2].item())

    # Calculate Euler angles
    if R[2][2] == 1:
        a, b, c = atan2(R[1][0], R[1][1]), 0.0, 0.0
    elif R[2][2] == -1:
        a, b, c = -atan2(R[1][0], R[1][1]), PI, 0.0
    else:
        a, b, c = atan2(R[1][2], R[0][2]), arccos(R[2][2]), atan2(R[2][1], -R[2][0])
    a, b, c = torch.as_tensor([a, b, c])

    # Calculate Wigner D-matrix
    # p-orbital, m = 1, -1, 0
    d_p = torch.tensor([
        [0.5 * (1 + cos(b)),    0.5 * (1 - cos(b)),   -0.5 * sqrt2 * sin(b)],
        [0.5 * (1 - cos(b)),    0.5 * (1 + cos(b)),    0.5 * sqrt2 * sin(b)],
        [0.5 * sqrt2 * sin(b), -0.5 * sqrt2 * sin(b),  cos(b),             ]
    ])
    ms = torch.tensor([1, -1, 0])
    e_ima = exp(-1j * ms * a).unsqueeze(1).expand(3, 3)
    e_imc = exp(-1j * ms * c).expand(3, 3)
    D_p = e_ima * d_p * e_imc

    # d-orbital, m = 0, 2, -2, 1, -1
    d_d = torch.tensor([
        [ 0.5*(3*cos(b)**2-1), sqrt38*sin(b)**2,   sqrt38*sin(b)**2,      sqrt38*sin(2*b),            -sqrt38*sin(2*b)],
        [ sqrt38*sin(b)**2, 0.25*(1+cos(b))**2,    0.25*(1-cos(b))**2,   -0.5*sin(b)*(1+cos(b)),      -0.5*sin(b)*(1-cos(b))],
        [ sqrt38*sin(b)**2, 0.25*(1-cos(b))**2,    0.25*(1+cos(b))**2,    0.5*sin(b)*(1-cos(b)),       0.5*sin(b)*(1+cos(b))],
        [-sqrt38*sin(2*b), 0.5*sin(b)*(1+cos(b)), -0.5*sin(b)*(1-cos(b)), 0.5*(2*cos(b)**2+cos(b)-1),  0.5*(-2*cos(b)**2+cos(b)+1)],
        [ sqrt38*sin(2*b), 0.5*sin(b)*(1-cos(b)), -0.5*sin(b)*(1+cos(b)), 0.5*(-2*cos(b)**2+cos(b)+1), 0.5*(2*cos(b)**2+cos(b)-1)]
    ])
    ms = torch.tensor([0, 2, -2, 1, -1])
    e_ima = exp(-1j * ms * a).unsqueeze(1).expand(5, 5)
    e_imc = exp(-1j * ms * c).expand(5, 5)
    D_d = e_ima * d_d * e_imc

    # D = torch.eye(13, 13, dtype=torch.cfloat)
    D = torch.eye(num_orb, num_orb, dtype=torch.complex128)
    D[2:5, 2:5] = D_p
    D[5:8, 5:8] = D_p
    D[8:, 8:] = D_d
    return R, D


def local_ham(data):
    """
    把Hamiltonian转动到local坐标。
    """
    wigner_D = torch.as_tensor(data.wigner_D, dtype=torch.complex128)
    ham_c_d = c2r @ data.ham.type(torch.complex128) @ r2c
    ham_c_l = wigner_D @ ham_c_d @ torch.linalg.inv(wigner_D)
    ham_r_l = r2c @ ham_c_l @ c2r
    assert (ham_r_l.imag < 10e-8).all(), abs(ham_r_l.imag).max()
    return ham_r_l.real.type(torch.float32)


def dft_ham(data, ham_local):
    """
    把Hamiltonian转动到dft坐标。
    """
    wigner_D = torch.as_tensor(data.wigner_D, dtype=torch.complex128)
    ham_c_l2 = c2r @ ham_local.type(torch.complex128) @ r2c
    ham_c_d2 = torch.linalg.inv(wigner_D) @ ham_c_l2 @ wigner_D
    ham_r_d2 = r2c @ ham_c_d2 @ c2r
    assert (ham_r_d2.imag < 10e-8).all()
    return ham_r_d2.real


def save_pred(data, ham, file=None):  # 把哈密顿量保存到文件
    with open(file, 'w') as fw:
        ham = ham.view(-1, num_orb, num_orb)
        for idx, h in enumerate(ham):
            e_i, e_j = data.edge_index[0][idx].item(), data.edge_index[1][idx].item()
            dist = data.edge_dist[idx][0].item()
            fw.write(f'edge_index: {e_i} {e_j}  edge_dist: {dist}\n')
            for i in range(num_orb):
                s = ''
                for j in range(num_orb):
                    out = h[i][j].item()
                    s += f'{out:10.5f}'
                fw.write(s + '\n')
            fw.write('\n')


def test(file='../dataset/sige_eval_222/2.pt'):
    """
    用来测试：从HS.out文件读取Hamiltonian，然后进行旋转等操作。
    """
    data = torch.load(file)
    print(data)

    # save_pred(data, data.y, 'rotated2.dat')

    # 转动到DFT坐标，和原始文件比较
    hams_recover = dft_ham(data, data.y)
    diff = hams_recover - data.ham
    print('diff max:', diff.max())
    print('diff mean:', diff.mean())


if __name__ == '__main__':
    test()


