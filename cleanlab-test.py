from cleanlab.classification import LearningWithNoisyLabels
from sklearn.linear_model import LogisticRegression
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.base import BaseEstimator

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
    def train(loader_train, model_obj, optimizer, loss_fn, device, total_epoch, epoch):
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

        def fit(self, X, y, sample_weight=None):
            self._mlp.to(self.device)
            data = [(X[i,:],y[i]) for i in range(len(X))]
            #data = np.dstack((X,y))
            #data = np.fromiter(zip(X,y), np.float32)
            loader = DataLoader(data, batch_size=self.batch_size)

            # 7. 損失関数を定義
            loss_fn = nn.CrossEntropyLoss()

            # 8. 最適化手法を定義（ここでは例としてAdamを選択）
            from torch import optim
            optimizer = optim.Adam(self._mlp.parameters(), lr=0.01)
            for epoch in range(self.epoch):
                train(loader, self._mlp, optimizer, loss_fn, self.device, self.epoch, epoch)

        def predict(self, X):
            self._mlp.to(self.device)
            with torch.no_grad():
                return self._mlp.forward(torch.tensor(X).to(self.device)).cpu()

        def predict_proba(self, X):
            self._mlp.to(self.device)
            with torch.no_grad():
                return self._mlp.forward(torch.tensor(X).to(self.device)).cpu()

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
                        download=True)
        testset = datasets.CIFAR10('../data',
                        train=False,
                        download=True)

        train_dataset_length = trainset.data.shape[0]
        train_dataset_shape = trainset.data.shape[1:]
        test_dataset_length = testset.data.shape[0]
        test_dataset_shape = testset.data.shape[1:]

        train_dataset_array = ((trainset.data.reshape(train_dataset_length, -1))/255.0).astype(np.float32)
        train_class_array = trainset.targets
        test_dataset_array = ((testset.data.reshape(test_dataset_length, -1))/255.0).astype(np.float32)
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
                noise_matrix[i][j] = noise_prob*2
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
    model = MLPModel(dataset_shape=train_dataset_shape, **kwargs)

    model.fit(train_dataset_array, train_class_array)
    predicted_test_labels = model.predict(test_dataset_array)

    print("==== Noisy Labelでの通常の学習による推定 ====")
    y_pred = torch.argmax(predicted_test_labels, dim=1)
    print(classification_report(test_class_array, y_pred))

    model = MLPModel(dataset_shape=train_dataset_shape, **kwargs)

    from cleanlab.latent_estimation import estimate_py_noise_matrices_and_cv_pred_proba
    lnl = LearningWithNoisyLabels(clf=model, seed=seed, cv_n_folds=cv_n_folds)
    lnl.fit(X=train_dataset_array, s=train_class_array)
    # Estimate the predictions you would have gotten by training with *no* label errors.
    predicted_test_labels = lnl.predict(test_dataset_array)

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
                        help='dataset.[MNIST or CIFAR10]', default='MNIST')
    parser.add_argument('--noise_prob', type=float, nargs=1,
                        help='noise probability.', default=0.1)
    parser.add_argument('--output', type=str, nargs=1,
                        help='output csv filename.', default='./noise_estimate.csv')
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, nargs=1,
                        help='pytorch device.', default=dev)

    parser.add_argument('--cv_n_folds', type=int, nargs=1,
                        help='The number of cross-validation folds used to compute', default=10)

    parser.add_argument('--batch_size', type=int, nargs=1,
                        help='batch size.', default=2048)
    parser.add_argument('--epoch', type=int, nargs=1,
                        help='epoch.', default=10)

    args = parser.parse_args()
    main(**vars(args))
