import matplotlib.pyplot as plt
import numpy as np
import torch
# from matplotlib import rcParams

plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0

def plot_a(name, sc):
    # 读取eval.log
    with open(f'{name}.log', 'r') as f:
        c = f.readlines()
    y = []
    for line in c:
        if line.startswith('file_name'):
            y.append(float(line.split()[-1]))
    y = np.array(y)
    yy = y[range(0, len(y)-1, 2)] * 1000
    yh = y[range(1, len(y), 2)] * 1000

    fig = plt.figure(figsize=(8, 6))
    ax0 = fig.add_subplot()
    for i in range(len(sc)):
        for j in range(10):
            ax0.scatter([i], yy[i * 10 + j], c='red', s=20, marker='o')

    # ax0.set_xlabel('Supercell', fontsize=18)
    # ax0.set_ylabel('MAE (meV)', fontsize=18)
    fig.subplots_adjust(left=0.12, bottom=0.12)
    fig.text(0.5, 0.02, 'Supercell (N×N×N)', ha='center', fontsize=20)
    fig.text(0.02, 0.5, 'MAE (meV)', va='center', rotation='vertical', fontsize=20)

    ax0.set_ylim((0.4, 1.0))
    ax0.set_xticks(range(0, len(sc)))
    ax0.set_xticklabels(sc, fontsize=15)
    yt = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    ax0.set_yticks(yt)
    ax0.set_yticklabels(yt, fontsize=15)

    # fig.tight_layout()
    fig.savefig(f'ttt_a.png')


def plot_b(name='pred.pt'):
    ticks = range(-15, 11, 5)
    # fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 6))
    # fig.subplots_adjust(left=0.12, bottom=0.12, wspace=0.1, hspace=0.1)
    fig, ax = plt.subplots()

    data = torch.load(name)
    true = data.ham.ravel().numpy()
    pred = data.ham_pred.ravel().numpy()
    mae = np.mean(abs(true - pred)) * 1000
    ax.plot([-15, 10], [-15, 10], dashes=[3, 3], c='black')
    ax.scatter(true, pred, s=5, c='red', label=f'{labels[i*2+j]}')
            # col.legend(fontsize=15, loc='lower right')

    col.set_xticks(ticks)
    col.set_xticklabels(ticks, fontsize=15)
    col.set_yticks(ticks)
    col.set_yticklabels(ticks, fontsize=15)

    fig.text(0.5, 0.02, 'DFT (eV)', ha='center', fontsize=20)
    fig.text(0.02, 0.5, 'Predicted (eV)', va='center', rotation='vertical', fontsize=20)

    # fig.tight_layout()
    fig.savefig(f'test.jpeg')


if __name__ == '__main__':
    # sc = ['222', '333', '444', '223', '224', '225', '229', '339', '555', '666']
    # plot_a(name='eval_1499', sc=sc)
    plot_b(name='data_pred/graphene_0.pt')
