from cleanlab.classification import LearningWithNoisyLabels
from sklearn.linear_model import LogisticRegression
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import ToTensor

import torchvision

from sklearn.base import BaseEstimator

from tqdm import tqdm

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.expansion*planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out

class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10):
		super(ResNet, self).__init__()
		self.in_planes = 64

		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
		self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
		self.linear = nn.Linear(512*block.expansion, num_classes)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out


def ResNet18():
	return ResNet(BasicBlock, [2,2,2,2])

def main(**kwargs):
    for key, value in kwargs.items():
        print(key + ': '+ str(value))
    globals().update(**kwargs)

    torch.manual_seed(seed)

    class MLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.classifier = nn.Sequential(
                # 第1引数：input
                # 第2引数：output
                nn.Flatten(),
                nn.Linear(dim, 400),
                # メモリを節約出来る
                nn.ReLU(inplace=True),
                nn.Linear(400, 200),
                nn.ReLU(inplace=True),
                nn.Linear(200, 100),
                nn.ReLU(inplace=True),
                nn.Linear(100, 10)
            )
        def forward(self, x):
            output = self.classifier(x)
            return output


    # 学習用関数
    def train(loader_train, model_obj, optimizer, loss_fn, device, total_epoch, epoch, shape):
        model_obj.train()  # モデルを学習モードに変更

        # ミニバッチごとに学習
        for data, targets in loader_train:

            data = data.to(device)  # GPUを使用するため，to()で明示的に指定
            targets = targets.to(device)  # 同上

            optimizer.zero_grad()  # 勾配を初期化
            outputs = model_obj(data)  # 順伝播の計算
            loss = loss_fn(outputs, targets)  # 誤差を計算

            loss.backward()  # 誤差を逆伝播させる
            optimizer.step()  # 重みを更新する

        #print('Epoch [%d/%d], Loss: %.4f' % (epoch, total_epoch, loss.item()))

    from torch.utils.data import Dataset, DataLoader

    #class NpDataset(Dataset):
    #  def __init__(self, array):
    #    self.array = array
    #  def __len__(self): return len(self.array)
    #  def __getitem__(self, i): return self.array[i]

    class MLPModel(BaseEstimator): # Inherits sklearn base classifier
        def __init__(self, dataset_shape, device, batch_size, epoch, **kwargs):
            dim = dataset_shape[0]
            for x in dataset_shape[1:]:
                dim *= x
            self._mlp = MLP(dim)
            self.device = device
            self.batch_size = batch_size
            self.epoch = epoch
            self.dataset_shape = (-1,) + dataset_shape

        def fit(self, X, y, sample_weight=None):
            self._mlp.to(self.device)
#            data = [(X[i,:],y[i]) for i in range(len(X))]
            #data = np.dstack((X,y))
            #data = np.fromiter(zip(X,y), np.float32)
            dataset = torch.utils.data.TensorDataset(X, y)

            loader = DataLoader(dataset, batch_size=self.batch_size)

            # 7. 損失関数を定義
            loss_fn = nn.CrossEntropyLoss()

            # 8. 最適化手法を定義（ここでは例としてAdamを選択）
            from torch import optim
            optimizer = optim.Adam(self._mlp.parameters(), lr=0.01)
            for epoch in tqdm(range(self.epoch)):
                train(loader, self._mlp, optimizer, loss_fn, self.device, self.epoch, epoch, self.dataset_shape)
            self._mlp.to('cpu')

        def predict(self, X):
            self._mlp.to(self.device)
            with torch.no_grad():
                return self._mlp.forward(torch.tensor(X.reshape(self.dataset_shape)).to(self.device)).cpu()

        def predict_proba(self, X):
            self._mlp.to(self.device)
            with torch.no_grad():
                return self._mlp.forward(torch.tensor(X.reshape(self.dataset_shape)).to(self.device)).cpu()

        def score(self, X, y, sample_weight=None):
            raise NotImplementedError()

    class ResNetModel(BaseEstimator): # Inherits sklearn base classifier
        def __init__(self, dataset_shape, device, batch_size, epoch, **kwargs):
            shape = (-1,) + dataset_shape
            self._net = ResNet18()
            self.device = device
            self.batch_size = batch_size
            self.epoch = epoch
            self.dataset_shape = shape

        def fit(self, X, y, sample_weight=None):
            self._net.to(self.device)
#            data = [(X[i,:],y[i]) for i in range(len(X))]
            #data = np.dstack((X,y))
            #data = np.fromiter(zip(X,y), np.float32)
#            loader = DataLoader(data, batch_size=self.batch_size)
            X = torch.tensor(X.reshape(self.dataset_shape))
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(X, y)

            loader = DataLoader(dataset, batch_size=self.batch_size)

            # 7. 損失関数を定義
            loss_fn = nn.CrossEntropyLoss()

            # 8. 最適化手法を定義（ここでは例としてAdamを選択）
            from torch import optim
            optimizer = optim.Adam(self._net.parameters(), lr=0.01)
            for epoch in tqdm(range(self.epoch)):
                train(loader, self._net, optimizer, loss_fn, self.device, self.epoch, epoch, self.dataset_shape)
            #self._net.to('cpu')

        def predict(self, X):
            X = torch.tensor(X).reshape(self.dataset_shape)
            dataset = torch.utils.data.TensorDataset(X)

            loader = DataLoader(dataset, batch_size=self.batch_size)
            with torch.no_grad():
                #self._net.to(self.device)
                # ミニバッチごとに学習
                ret = []
                for (data,) in loader:
                    ret.append(self._net.forward(data.to(self.device)))

            ret = torch.cat(ret)
            return ret.to('cpu')

        def predict_proba(self, X):
            X = torch.tensor(X).reshape(self.dataset_shape)
            dataset = torch.utils.data.TensorDataset(X)

            loader = DataLoader(dataset, batch_size=self.batch_size)
            with torch.no_grad():
                #self._net.to(self.device)
                # ミニバッチごとに学習
                ret = []
                for (data,) in loader:
                    ret.append(self._net.forward(data.to(self.device)))

            ret = torch.cat(ret)
            return ret.to('cpu')

        def score(self, X, y, sample_weight=None):
            raise NotImplementedError()


    from torchvision import datasets

    if dataset == 'MNIST':
        trainset = datasets.MNIST('../data',
                        train=True,
                        download=True)
        testset = datasets.MNIST('../data',
                        train=False,
                        download=True)

        train_dataset_length = trainset.data.shape[0]
        train_dataset_shape = trainset.data.shape[1:]
        test_dataset_length = testset.data.shape[0]
        test_dataset_shape = testset.data.shape[1:]

        train_dataset_array = ((trainset.data.numpy().reshape(train_dataset_length, -1))/255.0).astype(np.float32)
        train_class_array = trainset.targets.numpy()
        test_dataset_array = ((testset.data.numpy().reshape(test_dataset_length, -1))/255.0).astype(np.float32)
        test_class_array = testset.targets.numpy()

    elif dataset == 'CIFAR10':
        trainset = datasets.CIFAR10('../data',
                        train=True,
                        download=True, transform=torchvision.transforms.ToTensor())
        testset = datasets.CIFAR10('../data',
                        train=False,
                        download=True, transform=torchvision.transforms.ToTensor())

        train_dataset_length = trainset.data.shape[0]
        #train_dataset_shape = trainset.data[0].shape
        test_dataset_length = testset.data.shape[0]
        #test_dataset_shape = testset.data[0].shape

        train_dataset_array = next(iter(DataLoader(trainset, batch_size=len(trainset))))[0].numpy()
        train_dataset_shape = train_dataset_array[0].shape
        train_dataset_array = train_dataset_array.reshape((train_dataset_length, -1))
        train_class_array = trainset.targets
        test_dataset_array = next(iter(DataLoader(testset, batch_size=len(testset))))[0].numpy()
        test_dataset_shape = test_dataset_array[0].shape
        test_dataset_array = test_dataset_array.reshape((test_dataset_length, -1))
        test_class_array = testset.targets

    else:
        raise NotImplemented('unknown dataset name: ' + dataset)

    from sklearn.metrics import classification_report

    # Wrap around any classifier. Yup, you can use sklearn/pyTorch/Tensorflow/FastText/etc.

    from cleanlab.noise_generation import generate_noisy_labels

    noise_matrix = [
        # 0 -> 6 or 8
        [ 0.8,  0.0,  0.0,  0.0,  0.0,  0.0,  0.1,  0.0,  0.1,  0.0],
        # 1 -> 7,9
        [ 0.0,  0.8,  0.0,  0.0,  0.0,  0.0,  0.0,  0.1,  0.1,  0.0],
        # 2 -> 3,5
        [ 0.0,  0.0,  0.8,  0.1,  0.0,  0.1,  0.0,  0.0,  0.0,  0.0],
        # 3 -> 2,5
        [ 0.0,  0.0,  0.1,  0.8,  0.0,  0.1,  0.0,  0.0,  0.0,  0.0],
        # 4 -> 1, 9
        [ 0.0,  0.1,  0.0,  0.0,  0.8,  0.0,  0.0,  0.0,  0.0,  0.1],
        # 5 -> 2,3
        [ 0.0,  0.1,  0.1,  0.0,  0.0,  0.8,  0.0,  0.0,  0.0,  0.0],
        # 6 -> 0, 8
        [ 0.1,  0.0,  0.0,  0.0,  0.0,  0.0,  0.8,  0.0,  0.1,  0.0],
        # 7 -> 1, 9
        [ 0.0,  0.1,  0.0,  0.0,  0.0,  0.0,  0.0,  0.8,  0.0,  0.1],
        # 8 -> 0, 6
        [ 0.1,  0.0,  0.0,  0.0,  0.0,  0.0,  0.1,  0.0,  0.8,  0.0],
        # 9 -> 4, 7
        [ 0.0,  0.0,  0.0,  0.0,  0.1,  0.0,  0.0,  0.1,  0.0,  0.8],
    ]
    for i in range(len(noise_matrix)):
        for j in range(len(noise_matrix[i])):
            if noise_matrix[i][j] == 0.8:
                noise_matrix[i][j] = 1-noise_prob*2
            if noise_matrix[i][j] == 0.1:
                noise_matrix[i][j] = noise_prob
    train_targets = train_class_array
    train_class_array = generate_noisy_labels(train_class_array, noise_matrix)
    print("==== ノイズマトリックス ====")
    print(noise_matrix)

    print("==== Noisy Labelの評価 ====")
    from sklearn.metrics import confusion_matrix
    print("正解ラベル-ノイズラベルのconfusion_matrix:")
    print(confusion_matrix(train_targets, train_class_array))
    print("正解ラベル-ノイズラベルのclassification_report:")
    print(classification_report(train_targets, train_class_array))
    # 真のエラーインデックス
    err = np.invert(train_targets==train_class_array)

    # 単純なMLPモデルでconfident learningを行う
    if model_kind == 'MLP':
        model = MLPModel(dataset_shape=train_dataset_shape, **kwargs)
    elif model_kind == 'ResNet18':
        model = ResNetModel(dataset_shape=train_dataset_shape, **kwargs)

    model.fit(train_dataset_array, train_class_array)
    torch.cuda.empty_cache()
    predicted_test_labels = model.predict(test_dataset_array)
    torch.cuda.empty_cache()

    print("==== Noisy Labelでの通常の学習による推定 ====")
    y_pred = torch.argmax(predicted_test_labels, dim=1)
    print(classification_report(test_class_array, y_pred))

    if model_kind == 'MLP':
        model = MLPModel(dataset_shape=train_dataset_shape, **kwargs)
    elif model_kind == 'ResNet18':
        model = ResNetModel(dataset_shape=train_dataset_shape, **kwargs)

    from cleanlab.latent_estimation import estimate_py_noise_matrices_and_cv_pred_proba
    lnl = LearningWithNoisyLabels(clf=model, seed=seed, cv_n_folds=cv_n_folds)
    lnl.fit(X=train_dataset_array, s=train_class_array)
    torch.cuda.empty_cache()
    # Estimate the predictions you would have gotten by training with *no* label errors.
    predicted_test_labels = lnl.predict(test_dataset_array)
    torch.cuda.empty_cache()

    print("==== Noisy LabelのConfident Learningによる学習結果の評価 ====")
    y_pred = torch.argmax(predicted_test_labels, dim=1)
    print(confusion_matrix(test_class_array, y_pred))
    print(classification_report(test_class_array, y_pred))

    est_py, est_nm, est_inv, confident_joint, psx = estimate_py_noise_matrices_and_cv_pred_proba(
        X = train_dataset_array, s=train_class_array, clf=model, seed=seed, cv_n_folds=cv_n_folds
    )

    #y_pred = torch.argmax(psx, dim=1)
    y_pred = np.argmax(psx, axis=1)

    print("==== Confident Learningによるエラーマトリックス推定時に得られた結果の評価 ====")
    print(confusion_matrix(train_targets, y_pred))
    print(classification_report(train_targets, y_pred))

    print(confident_joint)
    np.set_printoptions(suppress=True, precision=2)
    print(est_nm)

    print("==== Confident Learningによるエラーインデックスの評価 ====")
    from cleanlab.pruning import get_noise_indices
    est_err = get_noise_indices(s=train_class_array, psx=psx, inverse_noise_matrix=est_inv, confident_joint=confident_joint)
    print(est_err)
    print(classification_report(err, est_err))

    import csv

    # 推定結果とエラーインデックスのcsvへの出力
    with open(output, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['true_label', 'noise_label', 'estimated_error', 'true_error'])
        out = list(zip(*[train_targets, train_class_array, est_err, err]))
        writer.writerows(out)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='cleanlab-test program.')
    parser.add_argument('--seed', type=int, nargs=1,
                        help='random seed.', default=0)
    parser.add_argument('--dataset', type=str, nargs=1,
                        help='dataset.[MNIST or CIFAR10]', default='CIFAR10')
    parser.add_argument('--model_kind', type=str, nargs=1,
                        help='dataset.[MLP or ResNet18]', default='ResNet18')
    parser.add_argument('--noise_prob', type=float, nargs=1,
                        help='noise probability.', default=0.1)
    parser.add_argument('--output', type=str, nargs=1,
                        help='output csv filename.', default='./noise_estimate.csv')
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, nargs=1,
                        help='pytorch device.', default=dev)

    parser.add_argument('--cv_n_folds', type=int, nargs=1,
                        help='The number of cross-validation folds used to compute', default=5)

    parser.add_argument('--batch_size', type=int, nargs=1,
                        help='batch size.', default=512)
    parser.add_argument('--epoch', type=int, nargs=1,
                        help='epoch.', default=10)

    args = parser.parse_args()
    main(**vars(args))
