import numpy as np
import argparse, time
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
import os
import random
import torch.nn as nn
import json

from model import Transformer_Based_Model
from dataset import *


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class Logit_Loss(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, tau=2.0):
        super(Logit_Loss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, z_s, z_t):
        y_s = (z_s / self.tau).softmax(dim=1)
        y_t = (z_t / self.tau).softmax(dim=1)
        inter_loss = self.tau ** 2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau ** 2 * intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss


class Feature_Loss(nn.Module):
    def __init__(self, temp=1.0):
        super(Feature_Loss, self).__init__()
        self.t = temp

    def forward(self, other_embd, text_embd):
        text_embd = F.normalize(text_embd, p=2, dim=1)
        other_embd = F.normalize(other_embd, p=2, dim=1)
        target = torch.matmul(text_embd, text_embd.transpose(0, 1))
        x = torch.matmul(text_embd, other_embd.transpose(0, 1))
        log_q = torch.log_softmax(x / self.t, dim=1)
        p = torch.softmax(target / self.t, dim=1)
        return F.kl_div(log_q, p, reduction='batchmean')


def CE_Loss(args, pred_outs, logit_t, hidden_s, hidden_t, labels):
    ori_loss = nn.CrossEntropyLoss()(pred_outs, labels)
    logit_loss = Logit_Loss().cuda()(pred_outs, logit_t)
    return ori_loss + logit_loss


def train_or_eval_model(model, data_loader, args, epoch=0, optimizer=None, scheduler=None, train=False):
    losses, preds, labels, masks = [], [], [], []
    losses_a_kd, losses_v_kd = [], []

    assert (not train) or (optimizer is not None)

    model.cuda()
    ce_loss = nn.CrossEntropyLoss()

    if train:
        model.train()
    else:
        model.eval()

    for data in data_loader:
        if train:
            optimizer.zero_grad()

        text, audio, video, audio_kd, video_kd, z_m, qmask, umask, label = [d.cuda() for d in data[:-1]]
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        t_logit, a_logit, v_logit, t_hidden, a_hidden, v_hidden = model(
            text, audio_kd, video_kd, z_m, umask, qmask, lengths
        )

        umask_bool = umask.bool()
        labels_ = label[umask_bool]

        logit_t = t_logit[umask_bool]
        logit_a = a_logit[umask_bool]
        logit_v = v_logit[umask_bool]

        hidden_t = t_hidden[umask_bool]
        hidden_a = a_hidden[umask_bool]
        hidden_v = v_hidden[umask_bool]

        loss_kd_a = CE_Loss(args, logit_a, logit_t, hidden_a, hidden_t, labels_)
        loss_kd_v = CE_Loss(args, logit_v, logit_t, hidden_v, hidden_t, labels_)

        loss_val = ce_loss(logit_t, labels_) + args.kd_a * loss_kd_a + args.kd_v * loss_kd_v

        pred_ = torch.argmax(logit_t, dim=1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss_val.item())
        losses_a_kd.append(loss_kd_a.item())
        losses_v_kd.append(loss_kd_v.item())

        if train:
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            scheduler.step()

    if preds != []:
        preds = np.concatenate(preds)
        masks = np.concatenate(masks)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), float('nan'), float('nan')

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_loss_a_kd = round(np.sum(losses_a_kd) / len(losses_a_kd), 4)
    avg_loss_v_kd = round(np.sum(losses_v_kd) / len(losses_v_kd), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, avg_loss_a_kd, avg_loss_v_kd


def _SaveModel(model, save_path, model_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, model_name))


def save_labels_and_preds(labels, preds, filename):
    data = {
        'labels': labels.tolist(),
        'preds': preds.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def model_train(model, optimizer, scheduler, train_loader, dev_loader, test_loader, args):
    best_valid_fscore = None
    best_epoch = -1
    early_stop_counter = 0

    for epoch in range(args.epochs):
        start = time.time()

        train_loss, train_acc, _, _, _, train_fscore, train_loss_a_kd, train_loss_v_kd = \
            train_or_eval_model(model, train_loader, args, epoch, optimizer, scheduler, True)

        valid_loss, valid_acc, _, _, _, valid_fscore, valid_loss_a_kd, valid_loss_v_kd = \
            train_or_eval_model(model, dev_loader, args, epoch)

        test_loss, test_acc, label, pred, _, test_fscore, test_loss_a_kd, test_loss_v_kd = \
            train_or_eval_model(model, test_loader, args, epoch)

        print(
            f'epoch: {epoch}, '
            f'train_loss: {train_loss}, train_acc: {train_acc}, train_fscore: {train_fscore}, '
            f'valid_loss: {valid_loss}, valid_acc: {valid_acc}, valid_fscore: {valid_fscore}, '
            f'test_loss: {test_loss}, test_acc: {test_acc}, test_fscore: {test_fscore}, '
            f'time: {round(time.time() - start, 2)}s'
        )
        print(
            f'epoch: {epoch}, '
            f'train_loss_a_kd: {train_loss_a_kd}, train_loss_v_kd: {train_loss_v_kd}, '
            f'valid_loss_a_kd: {valid_loss_a_kd}, valid_loss_v_kd: {valid_loss_v_kd}, '
            f'test_loss_a_kd: {test_loss_a_kd}, test_loss_v_kd: {test_loss_v_kd}'
        )

        if best_valid_fscore is None or valid_fscore > best_valid_fscore:
            best_valid_fscore = valid_fscore
            best_epoch = epoch
            early_stop_counter = 0

            _SaveModel(model, './save_model', args.save_name)
            save_labels_and_preds(label, pred, f'save_model/{args.save_json}')

            print(f'saved new best model at epoch {epoch} with valid_fscore={valid_fscore}')
        else:
            early_stop_counter += 1
            print(f'no improvement. early_stop_counter={early_stop_counter}/{args.early_stop_patience}')

        if early_stop_counter >= args.early_stop_patience:
            print(f'early stopping at epoch {epoch}. best_epoch={best_epoch}, best_valid_fscore={best_valid_fscore}')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for training.')
    parser.add_argument('--l2', type=float, default=1e-6, help='l2 regularization weight.')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training.')
    parser.add_argument('--seed', type=int, default=2024, help='random seed for training.')
    parser.add_argument('--epochs', type=int, default=30, help='epoch for training.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate.')
    parser.add_argument('--hidden_dim', type=int, default=768, help='hidden dimension.')
    parser.add_argument('--n_head', type=int, default=8, help='number of heads.')
    parser.add_argument('--temp', type=float, default=2.0, help='temperature for contrastive learning.')
    parser.add_argument('--clsNum', type=int, default=7, help='number of classes.')
    parser.add_argument('--train', type=bool, default=True, help='whether to train the model.')
    parser.add_argument('--kd_a', type=float, default=0.01, help='KD weight for audio branch.')
    parser.add_argument('--kd_v', type=float, default=0.08, help='KD weight for video branch.')
    parser.add_argument('--early_stop_patience', type=int, default=3, help='early stopping patience.')
    parser.add_argument('--save_name', type=str, default='multimodal_fusion_best.bin', help='best model file name.')
    parser.add_argument('--save_json', type=str, default='multimodal_fusion_best.json', help='best prediction json file name.')
    parser.add_argument('--zm_root', type=str, default='./omni_zm_real', help='root dir of generated z_m vectors')
    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)

    train_path = './feature/first_stage_train_features.pkl'
    dev_path = './feature/first_stage_dev_features.pkl'
    test_path = './feature/first_stage_test_features.pkl'

    # 这里开始真正把 zm_root 传给 dataset
    train_dataset = MELD_MM_Dataset(train_path, zm_root=args.zm_root)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=train_dataset.collate_fn
    )

    dev_dataset = MELD_MM_Dataset(dev_path, zm_root=args.zm_root)
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=dev_dataset.collate_fn
    )

    test_dataset = MELD_MM_Dataset(test_path, zm_root=args.zm_root)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=test_dataset.collate_fn
    )

    model = Transformer_Based_Model(args)
    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))

    if not args.train:
        model.load_state_dict(torch.load('./model/first_stage_model.pth'))

    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    model_train(model, optimizer, scheduler, train_loader, dev_loader, test_loader, args)