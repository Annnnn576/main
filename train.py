import numpy as np
import argparse, os
import random
import sklearn.metrics
import torch
import faiss
from faiss import normalize_L2
import scipy
import scipy.stats
import datasets
from torch.utils.data import DataLoader
import time
from metrics import *
from model import *
from loss.supcon_loss import *
from loss.loss import *
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('always')


# Training settings
parser = argparse.ArgumentParser(description="IMDN")
parser.add_argument("--batch_size", type=int, default=16, help="training batch size")
parser.add_argument("-nEpochs", type=int, default=300, help="number of epochs to train")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning Rate. Default=2e-4")

parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--using_lp', action='store_true')

parser.add_argument('--low_dim', type=int, default=128, help='Size of contrastive learning embedding')
parser.add_argument("--datasets", type=str, default="pascal")
parser.add_argument("--split_seed", type=int, default=1)

parser.add_argument("--number_class", type=int, default=20, help="number of class")
parser.add_argument("--input_dim", type=int, default=768)
parser.add_argument("--text_dim", type=int, default=768)

parser.add_argument("--alpha", type=float, default=0.2, help="Unknown Labels parameter")
parser.add_argument("--beta_pos", type=float, default=0.9, help="positive pseudo-label parameter")
parser.add_argument("--lam", type=float,default=10)

parser.add_argument("--gamma", type=int, default=0.01, help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda")
parser.add_argument("--resume", default="", type=str, help="path to checkpoint")

parser.add_argument("--root", type=str, default="./training_data", help='dataset directory')
parser.add_argument("--n_train", type=int, default=800, help="number of training set")
parser.add_argument("--n_val", type=int, default=1, help="number of validation set")

parser.add_argument("--scale", type=int, default=2, help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=192, help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1, help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3, help="number of color channels to use")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained models")
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--isY", action="store_true", default=True)
parser.add_argument("--ext", type=str, default='.npy')
parser.add_argument("--phase", type=str, default='train')

args = parser.parse_args()
print(args)
torch.backends.cudnn.benchmark = True
seed = args.seed
print("Ramdom Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')
torch.cuda.empty_cache()

print("===> Loading datasets")
trainset = datasets.get_data(args, 'train')
testset = datasets.get_data(args, 'val')
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)

print("===> Building models")
args.is_train = True
model = Model()
criterion = Supcon_loss(args)
criterion_ce = MyLoss(args)

print("===> Setting GPU")
if cuda:
    model = model.to(device)
    criterion = criterion.to(device)

print("===> Setting Optimizer")
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def adjust_learning_rate(optimizer, epoch): #, it
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    if epoch >= 15:
        lr = 1e-6
    if epoch >=20:
        lr = 5e-7

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def build_graph(X, k=10, args=None):

    X = X.astype('float32')
    d = X.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatIP(res, d, flat_config)  # build the index

    normalize_L2(X)
    index.add(X)
    N = X.shape[0]
    Nidx = index.ntotal

    c = time.time()
    D, I = index.search(X, k + 1)
    elapsed = time.time() - c

    # Create the graph
    D = D[:, 1:] ** 3
    I = I[:, 1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (k, 1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T

    # Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D

    return Wn

def estimating_label_correlation_matrix(Y_P):
    num_class = Y_P.shape[1]
    n = Y_P.shape[0]

    R = np.zeros((num_class, num_class))
    for i in range(num_class):
        for j in range(num_class):
            if i == j:
                R[i][j] = 0
            else:
                if np.sum(Y_P[:, i]) == 0 and np.sum(Y_P[:, j]) == 0 :
                    R[i][j] = 1e-5 # avoid divide zero error
                else:
                    R[i][j] = Y_P[:, i].dot(Y_P[:, j]) / (Y_P[:, i].sum() + Y_P[:, j].sum())
    D_1_2 = np.diag(1. / np.sqrt(np.sum(R, axis=1)))
    L = D_1_2.dot(R).dot(D_1_2)
    L = np.nan_to_num(L)

    return L

def label_propagation(args, Wn, L, Y_pred, Y_pred_1, Y_P_train): #L,

    alpha = 0.01
    eta = 0.01
    beta = 0.01
    gamma = args.gamma  # learning rate
    Z = Y_P_train

    Z_g = torch.from_numpy(Z).float().detach().cuda()
    Y_pred_g = torch.from_numpy(Y_pred).float().detach().cuda()
    L_g = torch.from_numpy(L).float().detach().cuda()

    with torch.no_grad():
        for i in range(200):
            W_matmul_Z_g = torch.from_numpy(Wn.dot(Z_g.cpu().numpy())).detach().cuda()
            grad = 2 * alpha * (Z_g - W_matmul_Z_g) + 1 * (Z_g - Y_pred_g) + beta * (Z_g - Z_g @ L_g)
            Z_g = Z_g - gamma * grad
            Y_pred_g = Z_g

    Z = Z_g.detach().cpu().numpy()

    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    Z = min_max_scaler.fit_transform(Z)

    torch.cuda.empty_cache()

    return Z

def train(epoch, results, results_1, Y_lp_np):
    model.train()
    losses = 0
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    for it, (data, data_aug, train_labels, label_obs, pseudo_label, index) in enumerate(trainloader):
        data, data_aug, label, label_obs = data.to(device), data_aug.to(device), train_labels.to(device), label_obs.to(device)

        optimizer.zero_grad()
        feat, logits = model(data)
        feat1, logits_1 = model(data_aug)
        pred = torch.sigmoid(logits)
        pred_1 = torch.sigmoid(logits_1)
        feats[index.cpu().detach().numpy().tolist()] = feat.cpu().detach().numpy()
        feats_1[index.cpu().detach().numpy().tolist()] = feat.cpu().detach().numpy()
        results[index.cpu().detach().numpy().tolist()] = pred.cpu().detach().numpy()
        results_1[index.cpu().detach().numpy().tolist()] = pred_1.cpu().detach().numpy()

        pred_label = torch.where(pred > 0.6, torch.tensor(1), torch.tensor(0))
        pred_label = pred_label.to(torch.int64)
        label_obs = label_obs.to(torch.int64)
        label = label.to(torch.int64)
        est_label = pred_label | label_obs

        Y_lp = torch.from_numpy(Y_lp_np[index, :]).float().detach().cuda()
        pseudo_label = torch.where(Y_lp > 0.8, torch.tensor(1), torch.tensor(0))
        pseudo_label = pseudo_label.to(torch.int64)
        pseudo_label = pseudo_label | label_obs
        if epoch > 2:
            loss_supcon = criterion(feat, feat1, est_label)
        else:
            loss_supcon = criterion(feat, feat1, label_obs)
        if epoch > 2:
            loss_ce = criterion_ce(pred, label_obs, Y_lp)
            loss_ce_1 = criterion_ce(pred_1, label_obs, Y_lp)
        else:
            loss_ce = criterion_ce(pred, label_obs)
            loss_ce_1 = criterion_ce(pred_1, label_obs)

        loss = 0.5*loss_supcon + loss_ce + loss_ce_1
        loss.backward()
        optimizer.step()
        losses += loss.item()

    loss = losses / len(trainloader)
    return loss, loss_ce, results, results_1, feats, feats_1

def valid():
    model.eval()
    pred_list = []
    label_list = []
    for train_data, data_aug, train_labels, label_obs, pseudo_label, index in testloader:
        img, label = train_data.to(device), train_labels.to(device)

        _, logits = model(img)
        pred = torch.sigmoid(logits)
        pred = pred.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        pred_list.append(pred)
        label_list.append(label)

    pred_list = np.concatenate(pred_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)

    metric = compute_metrics(pred_list, label_list)
    print("===> Valid. avg_pre: {:.4f} | rec_at_3: {:.4f}".format(metric['map'], metric['rec_at_3']))#/id

def compute_metrics(y_pred, y_true):
    '''
    Given predictions and labels, compute a few metrics.
    '''

    num_examples = len(y_true)
    num_classes = args.number_class

    results = {}
    average_precision_list = []
    # y_pred = np.array(y_pred)
    # y_true = np.array(y_true)
    y_true = np.array(y_true == 1, dtype=np.float32)  # convert from -1 / 1 format to 0 / 1 format
    for j in range(num_classes):
        average_precision_list.append(compute_avg_precision(y_true[:, j], y_pred[:, j]))

    results['map'] = 100.0 * float(np.mean(average_precision_list))

    for k in [1, 3, 5]:
        rec_at_k = np.array([compute_recall_at_k(y_true[i, :], y_pred[i, :], k) for i in range(num_examples)])
        prec_at_k = np.array(
            [compute_precision_at_k(y_true[i, :], y_pred[i, :], k) for i in range(num_examples)])
        results['rec_at_{}'.format(k)] = np.mean(rec_at_k)
        results['prec_at_{}'.format(k)] = np.mean(prec_at_k)
        results['top_{}'.format(k)] = np.mean(prec_at_k > 0)

    return results


def save_checkpoint(epoch):
    model_folder = "checkpoint_x{}/".format(args.scale)
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

def append_to_file(args, element):
    with open('./psnr/x{}.txt'.format(args.scale), 'a') as f:
        f.write(f"{element}\n")


print("===> Training")
print_network(model)
results = np.zeros((len(trainloader.dataset), args.number_class), dtype=np.float32)
results_1 = np.zeros((len(trainloader.dataset), args.number_class), dtype=np.float32)
feats = np.zeros((len(trainloader.dataset), 768), dtype=np.float32)
feats_1 = np.zeros((len(trainloader.dataset), 768), dtype=np.float32)
Y_pred_np = trainset.label_obs
Y_pred_np_1 = trainset.label_obs
Y_lp_np = trainset.label_obs
Y_P_train = trainset.label_obs
# L = estimating_label_correlation_matrix(Y_P_train)
for epoch in range(0, args.nEpochs):

    if epoch > 0:
        Wn = build_graph(feats, k=10)
        L = estimating_label_correlation_matrix(trainset.pseudo_label)
        Y_lp_np = label_propagation(args, Wn, L, Y_pred_np, Y_pred_np_1, Y_P_train)  #L,
        Y_lp_np[Y_lp_np > 1] = 1

    trainset.label_update(Y_lp_np, aug=True)
    adjust_learning_rate(optimizer, epoch)
    begin_epoch = time.time()
    losses, loss_ce, pred, pred_1, feats, feats_1 = train(epoch, results, results_1, Y_lp_np) #
    trainset.label_update(pred, aug=False)
    Y_pred_np = pred
    Y_pred_np_1 = pred_1

    print('Epoch {0:3d} | Time {1:5d} | loss {2:4f} | loss_ce {3:4f} ' .format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        losses,
        loss_ce,
    )
    )
    valid()
