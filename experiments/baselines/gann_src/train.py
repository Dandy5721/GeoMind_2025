import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD
from tqdm import tqdm
from torchinfo import summary
import numpy as np
import pandas as pd
from torch import nn
from dataset import RtFCDataset
from MMS_RNN import distance_loss
from utils import cluster_score, plot_epochs, AverageMeter
import warnings
from MMS_RNN import SPD_GBMS_RNN, Grouping
from SPD_CNN import SPDConv, SPDActivate, SPDDiag, Normalize, SPDAttention, SPDAvgPool
from SPD_CNN.optimizer import MyOptimizer

warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class MSNet(nn.Module):
    def __init__(self):
        super(MSNet, self).__init__()
        self.layers = nn.Sequential(
            SPDAttention(in_channels=1, kernel_size=5),
            SPDConv(1, 4, 5),
            Normalize(p='fro', dim=(-2, -1)),

            SPDAttention(in_channels=4, kernel_size=5),
            SPDActivate(activate_func='exp'),
            SPDConv(4, 8, 5),
            Normalize(p='fro', dim=(-2, -1)),

            SPDAttention(in_channels=8, kernel_size=5),
            SPDActivate(activate_func='exp'),

            # SPDAvgPool(2, 2),

            SPDDiag(),
            SPD_GBMS_RNN(0.1),
            SPD_GBMS_RNN(0.1),
            SPD_GBMS_RNN(0.1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == '__main__':
    input_dir = 'data/demo/bolds'
    label_dir = 'data/demo/labels/label.csv'
    output_dir = 'demo'

    num_classes = 3
    window = 30
    batch_size = 32
    num_workers = 8
    scan = None
    seed = 0
    transpose = False

    resume = None
    resume_log = None

    use_cuda = False
    save_result = True
    save_test_result = True
    total_epochs = 5
    log_columns = [
        'epoch',
        'train_loss',
        'train_acc',
        'test_loss',
        'test_acc',
        'val_loss',
        'val_acc',
    ]

    device = torch.device('cuda:0' if use_cuda else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    print('\n############### window = %d ###############\n' % window)

    if scan:
        result_path = os.path.join(
            'train_results', output_dir, '{}/window={}'.format(scan, window)
        )
        test_result_path = os.path.join(
            'test_results', output_dir, '{}/window={}'.format(scan, window)
        )
    else:
        result_path = os.path.join(
            'train_results', output_dir, 'window={}'.format(window)
        )
        test_result_path = os.path.join(
            'test_results', output_dir, 'window={}'.format(window)
        )

    models_save_path = os.path.join(result_path, 'models_save')

    num_sample = len([path for path in os.listdir(input_dir)])
    num_train = num_sample // 10 * 6
    num_val = (num_sample - num_train) // 2
    num_test = num_sample - num_train - num_val

    train_dataset = RtFCDataset(
        input_dir,
        label_dir,
        slice=slice(num_train),
        window=window,
        normalize=1e-3,
        transpose=transpose,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=num_workers
    )

    val_dataset = RtFCDataset(
        input_dir,
        label_dir,
        slice=slice(num_train, num_train + num_val),
        window=window,
        normalize=1e-3,
        transpose=transpose,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )

    test_dataset = RtFCDataset(
        input_dir,
        label_dir,
        slice=slice(-num_test, None),
        window=window,
        normalize=1e-3,
        transpose=transpose,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )

    model = MSNet()
    model.to(device)

    print(result_path)
    if save_test_result:
        os.makedirs(test_result_path, exist_ok=True)

    optimizer = SGD(model.parameters(), lr=0.005, weight_decay=1e-5, momentum=0.9)

    if resume:
        print('resume: ' + resume)
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['opt'])
        model.load_state_dict(checkpoint['model'])

    os.makedirs(models_save_path, exist_ok=True)

    if save_result:
        pd.DataFrame(columns=log_columns).to_csv(
            os.path.join(result_path, 'log.csv'), mode='a', index=False
        )
        with open(os.path.join(result_path, 'net.txt'), 'w') as f:
            size = train_dataset[0][0].shape[-1]
            a = torch.randn(8, 1, size, size)
            a = a @ a.transpose(-2, -1) + 1e-4 * torch.eye(size)
            model_str = str(summary(model, input_data=a, device=device))
            f.write(model_str)

    criterion = distance_loss
    optimizer = MyOptimizer(optimizer)

    def train(data_loader):
        model.train()
        losses = AverageMeter()
        purities = AverageMeter()
        nmis = AverageMeter()
        bar = tqdm(enumerate(data_loader), total=len(data_loader))
        for _, (inputs, targets) in bar:
            inputs = inputs.squeeze()
            targets = targets.squeeze()

            loader = DataLoader(
                TensorDataset(inputs, targets),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
            outputs_list = []
            targets_list = []
            for inputs, targets in loader:
                inputs = inputs.to(device).unsqueeze(-3)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                outputs_list.append(outputs)
                targets_list.append(targets)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                losses.update(loss.item())

            outputs = torch.cat(outputs_list, 0)
            targets = torch.cat(targets_list, 0)
            clustering = Grouping(n_clusters=num_classes, n_jobs=-1)
            labels_pred = clustering.fit(outputs.cpu().detach()).labels_
            purity, _, nmi = cluster_score(targets.cpu(), labels_pred)
            purities.update(purity)
            nmis.update(nmi)

            bar.set_description(
                f'Loss: {losses.avg:.4f} | purity: {purities.avg:.4f} | nmi: {nmis.avg:.4f}'
            )
        return losses.avg, purities.avg

    @torch.no_grad()
    def test(data_loader):
        model.eval()
        losses = AverageMeter()
        purities = AverageMeter()
        nmis = AverageMeter()
        bar = tqdm(enumerate(data_loader), total=len(data_loader))
        all_purity = []
        labels = []
        for batch_idx, (inputs, targets) in bar:
            inputs = inputs.squeeze()
            targets = targets.squeeze()

            loader = DataLoader(
                TensorDataset(inputs, targets),
                batch_size=inputs.shape[0],
                shuffle=False,
                num_workers=num_workers,
            )
            outputs_list = []
            targets_list = []
            for inputs, targets in loader:
                inputs = inputs.to(device).unsqueeze(-3)
                targets = targets.to(device)

                outputs = model(inputs)
                outputs_list.append(outputs)
                targets_list.append(targets)
                loss = criterion(outputs, targets)

                losses.update(loss.item())

            outputs = torch.cat(outputs_list, 0)
            targets = torch.cat(targets_list, 0)
            clustering = Grouping(n_clusters=num_classes, n_jobs=-1)
            labels_pred = clustering.fit(outputs.cpu().detach()).labels_
            purity, _, nmi = cluster_score(targets.cpu(), labels_pred)

            purities.update(purity)
            nmis.update(nmi)
            labels.append(labels_pred)
            all_purity.append(purity)

            bar.set_description(
                f'Loss: {losses.avg:.4f} | purity: {purities.avg:.4f} | nmi: {nmis.avg:.4f}'
            )

        return losses.avg, purities.avg, labels, all_purity

    train_losses = []
    train_accs = []

    test_losses = []
    test_accs = []

    val_losses = []
    val_accs = []

    epochs = []

    start_epoch = 1

    if resume_log:
        print('resume_log: ' + resume_log)
        log = pd.read_csv(resume_log)
        train_losses, train_accs = (
            log['train_loss'].to_list(),
            log['train_acc'].to_list(),
        )
        test_losses, test_accs = log['test_loss'].to_list(), log['test_acc'].to_list()
        val_losses, val_accs = log['val_loss'].to_list(), log['val_acc'].to_list()
        epochs = log['epoch'].to_list()
        best_acc = max(test_accs)
        start_epoch = max(epochs) + 1

        for i, epoch in enumerate(epochs):
            print('\nEpoch: %d' % epoch)
            print('Loss: %.4f | purity: %.4f' % (train_losses[i], train_accs[i]))
            print('Loss: %.4f | purity: %.4f' % (test_losses[i], test_accs[i]))
            print('Loss: %.4f | purity: %.4f' % (val_losses[i], val_accs[i]))

    for epoch in range(start_epoch, start_epoch + total_epochs):
        print('\nEpoch: %d' % epoch)
        train_loss, train_acc = train(train_loader)
        test_loss, test_acc, _, _ = test(test_loader)
        val_loss, val_acc, _, _ = test(val_loader)

        epochs.append(epoch)

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        test_losses.append(test_loss)
        test_accs.append(test_acc)

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if save_result:
            plot_epochs(
                os.path.join(result_path, 'loss.svg'),
                [train_losses, test_losses, val_losses],
                epochs,
                xlabel='epoch',
                ylabel='loss',
                legends=['train', 'test', 'val'],
                max=False,
            )
            plot_epochs(
                os.path.join(result_path, 'acc.svg'),
                [train_accs, test_accs, val_accs],
                epochs,
                xlabel='epoch',
                ylabel='accuracy',
                legends=['train', 'test', 'val'],
            )

            pd.DataFrame(
                [[epoch, train_loss, train_acc, test_loss, test_acc, val_loss, val_acc]]
            ).to_csv(
                os.path.join(result_path, 'log.csv'),
                mode='a',
                index=False,
                header=False,
            )
            torch.save(
                {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'opt': optimizer.state_dict(),
                },
                os.path.join(models_save_path, "%d_%.4f.pt" % (epoch, test_acc)),
            )
