# 分析误差分布
import torch
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

bins = [0.0, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 1.0, 10]


def main(files=None, fig_name=None):
    print(files)
    print(' d_min   d_max   ratio')
    diffs = [0] * (len(bins) - 1)
    nums = 0
    mae = 0
    for file in files:
        data = torch.load(file)
        # diff = (data.ham - data.ham_pred * 27.2113845).ravel().numpy()
        diff = (data.ham - data.ham_pred).ravel().numpy()
        nums += diff.shape[0]
        ad = abs(diff)
        mae += np.mean(ad)
        # print('Predicted MAE:', mae)

        # a, b = np.histogram(ad, bins=bins)
        # print(a/nums)

        for i in range(len(bins) - 1):
            s = (bins[i] < ad) & (ad < bins[i+1])
            diffs[i] += len(diff[s])

    for i in range(len(diffs)):
        print(f'{bins[i]:6.3f}  {bins[i+1]:6.3f}  {diffs[i]/nums:2.5%}')
    diffs = [d * 100 / nums for d in diffs]
    plt.bar(range(1, 9), diffs, align='edge', width=-0.9)
    plt.xticks(range(9), bins, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('ham_error (eV)', fontsize=12)
    plt.ylabel('ratio (%)', fontsize=12)
    for a, b, label in zip(range(9), diffs, diffs):
        plt.text(a, b, f'{label:.3f}%', fontsize=12)
    # plt.text()
    plt.text(6, 50, f'xxx\nMAE: {mae/len(files):.4f}', fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_name)


if __name__ == '__main__':
    files = glob('data_pred/c12_1.pt')
    main(files, fig_name='ttt.png')
