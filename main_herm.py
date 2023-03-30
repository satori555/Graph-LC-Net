import os, sys, time, argparse, copy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from glob import glob
import numpy as np
import torch
from torch_geometric.data import DataLoader
from data_loader.loaders import MyDataset, DatasetLDS
from model.model import Lcmp
from model.loss import FocalLoss
from utils.trans_ham import dft_ham, r2c, c2r


print(time.asctime(time.localtime(time.time())))
print('Command: python ', ' '.join(sys.argv))
print('Process id:', os.getpid(), '\n')

parser = argparse.ArgumentParser(description='PyTorch Test')

# dataset
parser.add_argument('--dataset_train', type=str, nargs='+', default=[
                                                                     './dataset/graphene/',
                                                                    ])
parser.add_argument('--dataset_val', type=str, nargs='+', default=[
                                                                     './dataset/graphene_eval/',
                                                                  ])
parser.add_argument('--sys_name', type=str, default='test')
parser.add_argument('--rev', action='store_true', default=False)

# hyperparameters
parser.add_argument('--node_dim', type=int, default=64)
parser.add_argument('--edge_dim', type=int, default=128)
parser.add_argument('--sph_harm_dim', type=int, default=64)
parser.add_argument('--out_dim', type=int, default=91)
parser.add_argument('--lds', action='store_true', default=False)

# training
parser.add_argument('--restart', action='store_true', default=False)
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--eval_dir', type=str, default='./data_pred/')
parser.add_argument('--init_model', type=str, default='test_final.pth')
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--lr_init', type=float, default=1e-3)
parser.add_argument('--lr_min', type=float, default=1e-5)
parser.add_argument('--decay_step', type=int, default=30)

# checkpoints
parser.add_argument('--ckpt', type=int, default=200)
parser.add_argument('--patience', type=int, default=300)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
if args.cuda:
    r2c = torch.as_tensor(r2c, dtype=torch.complex64).cuda()
    c2r = torch.as_tensor(c2r, dtype=torch.complex64).cuda()

if args.rev:
    args.sys_name = 'rev_' + args.sys_name
    args.init_model = 'rev_' + args.init_model

# y_mask = torch.ones(13, 13) > 0
y_mask = torch.triu(torch.ones(13, 13), diagonal=0) > 0
# y_mask = torch.eye(13) > 0

print('------args------')
for k, v in vars(args).items():
    print(k + ' = ' + str(v))
print('')


def train(model, data_iter, loss_fn, loss_mae, optimizer):
    sum_loss = 0
    sum_mae_loss = 0
    model.train()
    for data in data_iter:
        if args.cuda:
            data = data.cuda(non_blocking=True)
        data.ylm = data.ylm[0]
        out = model(data, args.rev)

        if args.rev:
            y_true = torch.where(data.rev.view(-1, 1, 1), data.y, torch.zeros_like(data.y))[:, y_mask]
        else:
            y_true = torch.where(~data.rev.view(-1, 1, 1), data.y, torch.zeros_like(data.y))[:, y_mask]

        if 'weights' in data.keys:
            loss = loss_fn(out, y_true, data.weights)
        else:
            loss = loss_fn(out, y_true)
        sum_loss += loss.item() / out.numel() * 2  # 乘以2是因为大约一般的矩阵元被重置为0了

        # 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss2 = loss_mae(out, y_true)
        sum_mae_loss += loss2.item() * 2
        # print(loss.item(), loss2.item())

    print(f'Train_Loss: {sum_loss/len(data_iter):.6g}  ', end='')
    print(f'Train_MAE_Loss: {sum_mae_loss/len(data_iter):.6g}  ', end='')
    return sum_mae_loss/len(data_iter)


def validate(model, val_loader, loss_fn):
    sum_loss = 0
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            if args.cuda:
                data = data.cuda(non_blocking=True)
            data.ylm = data.ylm[0]
            out = model(data, args.rev)
            if args.rev:
                y_true = torch.where(data.rev.view(-1, 1, 1), data.y, torch.zeros_like(data.y))[:, y_mask]
            else:
                y_true = torch.where(~data.rev.view(-1, 1, 1), data.y, torch.zeros_like(data.y))[:, y_mask]

            loss = loss_fn(out, y_true)
            sum_loss += loss.item() * 2
    print(f'Val_MAE_loss: {sum_loss/len(val_loader):.6g}  ', end='')
    return sum_loss/len(val_loader)


def evaluate(model=None, val_loader=None, filenames=None, loss_fn=None, dump_ham=False):
    pred_dir = args.eval_dir
    if not os.path.isdir(pred_dir):
        os.mkdir(pred_dir)
    model[0].eval()
    model[1].eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if args.cuda:
                data = data.cuda(non_blocking=True)
            data.ylm = data.ylm[0]
            # out = model(data).view(-1, 13, 13)
            out_0 = model[0](data, rev=False)
            out_1 = model[1](data, rev=True)

            y_true_0 = torch.where(~data.rev.view(-1, 1, 1), data.y, torch.zeros_like(data.y))[:, y_mask]
            y_true_1 = torch.where(data.rev.view(-1, 1, 1), data.y, torch.zeros_like(data.y))[:, y_mask]
            loss_0 = loss_fn(out_0, y_true_0).item()
            loss_1 = loss_fn(out_1, y_true_1).item()
            print('loss_0:', loss_0)
            print('loss_1', loss_1)
            # exit()

            pred_0 = torch.zeros_like(data.y)
            pred_1 = torch.zeros_like(data.y)
            pred_0[:, y_mask] = out_0
            pred_1[:, y_mask] = out_1
            pred_upper = pred_0 + pred_1

            # 把 rev 边的对角线改成 0，后续修改 rev 不要对角线：91 - 78
            # onsite 对角线除以 2 （有没有更好的方法？）
            mask_diag = torch.eye(13) > 0
            for jj, edge_attr in enumerate(pred_upper):
                if data.rev[jj]:
                    edge_attr[mask_diag] = 0.0
                if data.index_rev[jj] == jj:
                    edge_attr[mask_diag] /= 2.0

            pred_all = torch.zeros_like(pred_upper)
            mask = data.index_rev >= 0
            pred_all[mask] = pred_upper[mask] + torch.tril(pred_upper[data.index_rev][mask].transpose(1, 2), diagonal=0)

            loss = loss_fn(pred_all, data.y)
            print(f'file_name: {filenames[i]} Eval_MAE_Loss: {loss:.6g}')

            data = data.cpu()
            pred_all = pred_all.cpu()
            data.ham_pred = dft_ham(data=data, ham_local=pred_all)

            loss_ham = loss_fn(data.ham_pred, data.ham)
            print(f'file_name: {filenames[i]} Eval_MAE_Loss_ham: {loss_ham:.6g}')

            fn = filenames[i].split('/', 2)[-1].replace('/', '_')
            torch.save(data, pred_dir + fn)
            print('Data saved to:' + pred_dir + fn)

            if dump_ham:
                with open(pred_dir + fn + '.txt', 'w') as fw:
                    for idx, h in enumerate(data.ham_pred):
                        e_i, e_j = data.edge_index[0][idx].item(), data.edge_index[1][idx].item()
                        dist = data.edge_dist[idx][0].item()
                        fw.write(f'edge_index: {e_i} {e_j}  edge_dist: {dist}\n')
                        for ii in range(13):
                            s = ''
                            for jj in range(13):
                                out = h[ii][jj].item()
                                s += f'{out:10.5f}'
                            fw.write(s + '\n')
                        fw.write('\n')


def main():
    t1 = time.time()
    # data
    print('Preparing data...')
    files_train = []
    files_val = []
    for dirs in args.dataset_train:
        files = glob(os.path.join(dirs, '*.pt'))
        files_train.extend(files)
    for dirs in args.dataset_val:
        files = glob(os.path.join(dirs, '*.pt'))
        files_val.extend(files)

    files_val = files_val[:10]
    # print(files_val)

    if not os.path.isdir('tmp/'):
        os.mkdir('tmp/')
    if not os.path.isdir('checkpoints/'):
        os.mkdir('checkpoints/')

    np.save(f'tmp/{args.sys_name}_files_train.npy', files_train)
    np.save(f'tmp/{args.sys_name}_files_val.npy', files_val)
    np.savetxt(f'tmp/{args.sys_name}_files_train.txt', files_train, fmt='%s')
    np.savetxt(f'tmp/{args.sys_name}_files_val.txt', files_val, fmt='%s')

    if args.lds:
        train_data = DatasetLDS(root=None, file_name=f'tmp/{args.sys_name}_files_train.npy', weights='weights.npy')
    else:
        train_data = MyDataset(root=None, file_name=f'tmp/{args.sys_name}_files_train.npy')
    val_data = MyDataset(root=None, file_name=f'tmp/{args.sys_name}_files_val.npy')
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=6, pin_memory=args.cuda)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=args.cuda)
    print(f'Train: {len(files_train)}. Validate: {len(files_val)}.')

    # model
    lcmp = Lcmp(channels=args.node_dim, dim_e=args.edge_dim, dim_Y=args.sph_harm_dim, dim_ham=args.out_dim)
    if args.cuda:
        lcmp = lcmp.cuda()
    total_trainable_params = sum(p.numel() for p in lcmp.parameters() if p.requires_grad)
    print(lcmp)
    print(f'{total_trainable_params:,} total training parameters.', '\n')

    focal_loss = FocalLoss(reduce=False)
    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(lcmp.parameters(), lr=args.lr_init)
    # gamma = (args.lr_min / args.lr_init) ** (args.decay_step / args.epochs)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6,
                                                           patience=args.decay_step, min_lr=args.lr_min)

    init_epoch = 0
    if args.restart or args.evaluate:
        ckp = torch.load('checkpoints/' + args.init_model)
        init_epoch = ckp['epoch'] + 1
        lcmp.load_state_dict(ckp['net'])
        optimizer.load_state_dict(ckp['optimizer'])
        print(f"Load Model State from checkpoints/{args.init_model}.", '\n')

    # evaluate only
    if args.evaluate:
        lcmp_rev = copy.deepcopy(lcmp)
        ckp_rev = torch.load('checkpoints/rev_' + args.init_model)
        lcmp_rev.load_state_dict(ckp_rev['net'])
        evaluate(model=(lcmp, lcmp_rev), val_loader=val_loader, filenames=files_val, loss_fn=mae_loss, dump_ham=False)
        print(f'Finished! Elapsed time (sec): {time.time() - t1}.')
        return

    # train
    best_val_loss = 0.9
    best_model = None
    best_epoch = 0
    es = 0  # early stopping
    print('Start training...')
    for epoch in range(init_epoch, init_epoch + args.epochs):
        print(f'Epoch: {epoch}  ', end='')
        t_epoch = time.time()
        train_loss = train(lcmp, train_loader, focal_loss, mae_loss, optimizer)
        val_loss = validate(lcmp, val_loader, mae_loss)

        curr_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'lr: {curr_lr:.3e}  ', end='')
        print(f'time: {time.time() - t_epoch :.3f}')

        if (epoch+1) % args.ckpt == 0:
            state_dict = {
                'epoch': epoch,
                'net': lcmp.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state_dict, f'checkpoints/{args.sys_name}_{epoch}.pth')
            print(f"Save checkpoint to checkpoints/{args.sys_name}_{epoch}.pth")

        scheduler.step(val_loss)

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(lcmp)
            best_epoch = epoch
            es = 0
        else:
            es += 1
            if es > args.patience:
                print(f'Early stopping with best_val_loss: {best_val_loss}')
                break

    # save best model
    state_dict = {
        'epoch': best_epoch,
        'net': best_model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state_dict, f'checkpoints/{args.sys_name}_final.pth')
    print(f"Save Model State to checkpoints/{args.sys_name}_final.pth")
    print(f'Finished! Elapsed time (sec): {time.time()-t1}.')


if __name__ == '__main__':
    main()

