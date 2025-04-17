# coding: utf-8

import os
import sys
import pickle
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from sklearn.model_selection import train_test_split

# ============ 1. 解析命令行参数 ============
def get_args():
    parser = argparse.ArgumentParser(
        description="Train LeNet or CNN+Transformer on MNIST or EMNIST (various splits) and evaluate on test set"
    )
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Dataset type: mnist or emnist')
    parser.add_argument('--emnist_split', type=str, default='mnist',
                        help='For EMNIST: specify split: mnist, balanced, letters, bymerge, or byclass')
    parser.add_argument('--model', type=str, default='lenet',
                        help='Model type: lenet or transformer')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0022,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes. If not provided, defaults: for mnist/emnist-mnist=10, balanced=47, letters=37, bymerge=47, byclass=62')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='Directory to save model files')
    # 添加噪声相关参数：噪声标准差和噪声类型（目前只支持高斯噪声）
    parser.add_argument('--noise_std', type=float, default=0.0,
                        help='Standard deviation for Gaussian noise to be added to test images')
    parser.add_argument('--noise_type', type=str, default='gaussian',
                        help='Type of noise to add to test images (currently only supports gaussian)')
    
    args = parser.parse_args()
    # 根据不同数据集默认类别数（你也可以在命令行手动指定）
    if args.num_classes is None:
        if args.dataset.lower() == 'mnist':
            args.num_classes = 10
        elif args.dataset.lower() == 'emnist':
            split = args.emnist_split.lower()
            if split == 'mnist':
                args.num_classes = 10
            elif split == 'balanced':
                args.num_classes = 47
            elif split == 'letters':
                args.num_classes = 37
            elif split == 'bymerge':
                args.num_classes = 47
            elif split == 'byclass':
                args.num_classes = 62
            else:
                raise ValueError("Unsupported EMNIST split: " + args.emnist_split)
        else:
            raise ValueError("Please specify --num_classes for dataset: " + args.dataset)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    return args

# ============ 2. 数据加载与预处理 ============
def load_mnist_data():
    """
    假设 MNIST 数据存放在 /scratch/gilbreth/hu953/cs57800/dataset/MNIST 下，
    包含 traindata.pkl, trainlabel.npy, testdata.pkl, testlabel.npy
    """
    mnist_dir = '/scratch/gilbreth/hu953/cs57800/dataset/MNIST'
    with open(os.path.join(mnist_dir, 'traindata.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    train_label = np.load(os.path.join(mnist_dir, 'trainlabel.npy'))
    
    with open(os.path.join(mnist_dir, 'testdata.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    test_label = np.load(os.path.join(mnist_dir, 'testlabel.npy'))
    
    return (train_data, train_label), (test_data, test_label)

def load_emnist_data(split_type='mnist'):
    """
    根据 split_type 加载 EMNIST 数据：
      - "mnist": 使用 emnist-mnist-train.csv / emnist-mnist-test.csv
      - "balanced": 使用 emnist-balanced-train.csv / emnist-balanced-test.csv
      - "letters": 使用 emnist-letters-train.csv / emnist-letters-test.csv
      - "bymerge": 使用 emnist-bymerge-train.csv / emnist-bymerge-test.csv
      - "byclass": 使用 emnist-byclass-train.csv / emnist-byclass-test.csv

    CSV 格式：第一列为 label，其余 784 列为像素（28x28）。
    """
    emnist_dir = '/scratch/gilbreth/hu953/cs57800/dataset/EMNIST'
    split_type = split_type.lower()
    if split_type == 'mnist':
        train_csv = os.path.join(emnist_dir, 'emnist-mnist-train.csv')
        test_csv  = os.path.join(emnist_dir, 'emnist-mnist-test.csv')
    elif split_type == 'balanced':
        train_csv = os.path.join(emnist_dir, 'emnist-balanced-train.csv')
        test_csv  = os.path.join(emnist_dir, 'emnist-balanced-test.csv')
    elif split_type == 'letters':
        train_csv = os.path.join(emnist_dir, 'emnist-letters-train.csv')
        test_csv  = os.path.join(emnist_dir, 'emnist-letters-test.csv')
    elif split_type == 'bymerge':
        train_csv = os.path.join(emnist_dir, 'emnist-bymerge-train.csv')
        test_csv  = os.path.join(emnist_dir, 'emnist-bymerge-test.csv')
    elif split_type == 'byclass':
        train_csv = os.path.join(emnist_dir, 'emnist-byclass-train.csv')
        test_csv  = os.path.join(emnist_dir, 'emnist-byclass-test.csv')
    else:
        raise ValueError("Unsupported EMNIST split: " + split_type)

    train_data, train_label = [], []
    with open(train_csv, 'r') as f:
        for line in f:
            row = line.strip().split(',')
            label = int(row[0])
            pixels = np.array(row[1:], dtype=np.uint8)
            img = pixels.reshape(28, 28)
            train_data.append(img)
            train_label.append(label)
    
    test_data, test_label = [], []
    with open(test_csv, 'r') as f:
        for line in f:
            row = line.strip().split(',')
            label = int(row[0])
            pixels = np.array(row[1:], dtype=np.uint8)
            img = pixels.reshape(28, 28)
            test_data.append(img)
            test_label.append(label)

    return (train_data, np.array(train_label)), (test_data, np.array(test_label))

def transform_emnist(img):
    """
    对于 EMNIST，转置+左右翻转，使其方向与 MNIST 一致
    """
    img = np.transpose(img)
    img = np.fliplr(img)
    return img

def paddingsize(img, do_transform=False):
    """
    将图像统一调整为 64x64
    如果 do_transform=True，则先对图像进行转置+翻转（用于 EMNIST）
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if do_transform:
        img = transform_emnist(img)
    row, col = img.shape
    if row < 64:
        img = np.vstack([img, np.zeros([64 - row, col])])
    elif row > 64:
        img = img[0:64, :]
    if col < 64:
        img = np.hstack([img, np.zeros([64, 64 - col])])
    elif col > 64:
        img = img[:, 0:64]
    img = np.expand_dims(img, axis=0)  # (1, 64, 64)
    return img

def build_dataloader(dataset_type, batch_size, emnist_split='mnist'):
    """
    根据 dataset_type 加载训练数据，预处理后返回训练集和验证集的 DataLoader
    """
    if dataset_type.lower() == 'mnist':
        (train_data, train_label), _ = load_mnist_data()
        do_transform = False
    elif dataset_type.lower() == 'emnist':
        (train_data, train_label), _ = load_emnist_data(split_type=emnist_split)
        do_transform = True
    else:
        raise ValueError("Unsupported dataset type: {}".format(dataset_type))
    
    img_sets = []
    for i in range(len(train_data)):
        img_sets.append(paddingsize(train_data[i], do_transform=do_transform))
    img_sets = np.array(img_sets)  # (N, 1, 64, 64)
    labels = np.array(train_label)
    
    train_X, vali_X, train_y, vali_y = train_test_split(
        img_sets, labels, test_size=0.1, random_state=0
    )
    
    train_dataset = Data.TensorDataset(torch.FloatTensor(train_X),
                                       torch.LongTensor(train_y))
    val_dataset = Data.TensorDataset(torch.FloatTensor(vali_X),
                                     torch.LongTensor(vali_y))
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def build_test_dataloader(dataset_type, batch_size, emnist_split='mnist'):
    """
    根据 dataset_type 加载测试数据，预处理后返回测试集的 DataLoader
    """
    if dataset_type.lower() == 'mnist':
        (_, _), (test_data, test_label) = load_mnist_data()
        do_transform = False
    elif dataset_type.lower() == 'emnist':
        (_, _), (test_data, test_label) = load_emnist_data(split_type=emnist_split)
        do_transform = True
    else:
        raise ValueError("Unsupported dataset type: {}".format(dataset_type))
    
    test_imgs = []
    for i in range(len(test_data)):
        test_imgs.append(paddingsize(test_data[i], do_transform=do_transform))
    test_imgs = np.array(test_imgs)
    test_label = np.array(test_label)
    
    test_dataset = Data.TensorDataset(torch.FloatTensor(test_imgs),
                                      torch.LongTensor(test_label))
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# ============ 3. 定义模型结构 ============
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 13 * 13, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CNN_Transformer(nn.Module):
    def __init__(self, num_classes=10, transformer_layers=2, nhead=4, d_model=256):
        super(CNN_Transformer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.feature_size = 16 * 13 * 13
        self.fc_before_transformer = nn.Linear(self.feature_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc_before_transformer(x)
        x = x.unsqueeze(0)              # (1, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)                # (batch_size, d_model)
        return self.classifier(x)

# ============ 4. 训练逻辑（含 Early Stopping） ============
def train_model(args):
    # 当数据集为 emnist 时，将使用 args.emnist_split 参数
    train_loader, val_loader = build_dataloader(args.dataset, args.batch_size, emnist_split=args.emnist_split)
    
    num_classes = args.num_classes

    if args.model.lower() == 'lenet':
        net = LeNet(num_classes=num_classes)
    elif args.model.lower() == 'transformer':
        net = CNN_Transformer(num_classes=num_classes)
    else:
        raise ValueError("Unsupported model type: {}".format(args.model))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print("[Epoch %d/%d] Iter [%d/%d] Loss: %.4f" % (
                    epoch + 1, args.epochs, i + 1, len(train_loader), loss.item()
                ))
        net.eval()
        val_losses = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                v_loss = loss_func(outputs, labels)
                val_losses.append(v_loss.item())
        avg_val_loss = np.mean(val_losses)
        print("Epoch [%d/%d] Validation Loss: %.4f" % (epoch + 1, args.epochs, avg_val_loss))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(net.state_dict(), os.path.join(args.save_dir, f"{args.model}_best.pkl"))
        else:
            patience_counter += 1
            print("No improvement for %d epoch(s)." % patience_counter)
            if patience_counter >= args.patience:
                print("Early stopping triggered at epoch %d" % (epoch + 1))
                break

    torch.save(net.state_dict(), os.path.join(args.save_dir, f"{args.model}.pkl"))
    print("Training complete!")
    return net

def test_model(args, net):
    """
    在测试集上评估模型，输出准确率等指标。
    如果指定了噪声参数，则在测试时对图像加入噪声。
    """
    test_loader = build_test_dataloader(args.dataset, args.batch_size, emnist_split=args.emnist_split)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            # 如果指定了噪声标准差，则对测试图像加入噪声（这里仅实现了高斯噪声）
            if args.noise_std > 0:
                if args.noise_type.lower() == 'gaussian':
                    noise = torch.randn_like(images) * args.noise_std
                    images = images + noise
                    # 根据数据集保持与训练时一致的像素范围
                    if args.dataset.lower() == 'emnist':
                        images = torch.clamp(images, 0.0, 255.0)
                    else:
                        images = torch.clamp(images, 0.0, 1.0)
                else:
                    raise ValueError("Unsupported noise type: " + args.noise_type)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print("Test Accuracy: {:.2%}".format(accuracy))
    return accuracy

# ============ 5. 主程序入口 ============
if __name__ == "__main__":
    args = get_args()
    print("===== Training Config =====")
    print("Dataset:       ", args.dataset)
    if args.dataset.lower() == 'emnist':
        print("EMNIST Split:  ", args.emnist_split)
    print("Model:         ", args.model)
    print("BatchSize:     ", args.batch_size)
    print("Epochs:        ", args.epochs)
    print("LR:            ", args.lr)
    print("Patience:      ", args.patience)
    print("NumClasses:    ", args.num_classes)
    print("SaveDir:       ", args.save_dir)
    print("Noise Std:     ", args.noise_std)
    print("Noise Type:    ", args.noise_type)
    print("===========================")
    
    net = train_model(args)
    test_model(args, net)
