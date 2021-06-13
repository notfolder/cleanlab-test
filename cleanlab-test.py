from cleanlab.classification import LearningWithNoisyLabels
from sklearn.linear_model import LogisticRegression
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.base import BaseEstimator

torch.manual_seed(0)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            # 第1引数：input
            # 第2引数：output
            nn.Linear(28 * 28, 400),
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

    print('Epoch [%d/%d], Loss: %.4f' % (epoch, total_epoch, loss.item()))

from torch.utils.data import Dataset, DataLoader

#class NpDataset(Dataset):
#  def __init__(self, array):
#    self.array = array
#  def __len__(self): return len(self.array)
#  def __getitem__(self, i): return self.array[i]

class MLPModel(BaseEstimator): # Inherits sklearn base classifier
    def __init__(self):
        self._mlp = MLP()

    def fit(self, X, y, sample_weight=None):
        self._mlp.to('cuda:0')
        data = [(X[i,:],y[i]) for i in range(len(X))]
        #data = np.dstack((X,y))
        #data = np.fromiter(zip(X,y), np.float32)
        loader = DataLoader(data, batch_size=1024)

        # 7. 損失関数を定義
        loss_fn = nn.CrossEntropyLoss()

        # 8. 最適化手法を定義（ここでは例としてAdamを選択）
        from torch import optim
        optimizer = optim.Adam(self._mlp.parameters(), lr=0.01)
        for epoch in range(10):
            train(loader, self._mlp, optimizer, loss_fn, 'cuda:0', 10, epoch)

    def predict(self, X):
        self._mlp.to('cuda:0')
        with torch.no_grad():
            return self._mlp.forward(torch.tensor(X).to('cuda:0')).cpu()

    def predict_proba(self, X):
        self._mlp.to('cuda:0')
        with torch.no_grad():
            return self._mlp.forward(torch.tensor(X).to('cuda:0')).cpu()

    def score(self, X, y, sample_weight=None):
        raise NotImplementedError()


from torchvision import datasets

trainset = datasets.MNIST('../data',
                train=True,
                download=True)
testset = datasets.MNIST('../data',
                train=False,
                download=True)

from sklearn.metrics import classification_report

# Wrap around any classifier. Yup, you can use sklearn/pyTorch/Tensorflow/FastText/etc.
train_dataset_array = ((trainset.data.numpy().reshape(60000, 28*28))/255.0).astype(np.float32)
train_class_array = trainset.targets.numpy()
test_dataset_array = ((testset.data.numpy().reshape(10000, 28*28))/255.0).astype(np.float32)

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
train_targets = train_class_array
train_class_array = generate_noisy_labels(train_class_array, noise_matrix)

print("==== Noisy Labelの評価 ====")
from sklearn.metrics import confusion_matrix
print(confusion_matrix(train_targets, train_class_array))
print(classification_report(train_targets, train_class_array))
err = np.invert(train_targets==train_class_array)
print(err)

model = MLPModel()

model.fit(train_dataset_array, train_class_array)
predicted_test_labels = model.predict(test_dataset_array)

print("==== Noisy Labelの通常の学習による推定 ====")
y_pred = torch.argmax(predicted_test_labels, dim=1)
print(classification_report(testset.targets, y_pred))

model = MLPModel()

from cleanlab.latent_estimation import estimate_py_noise_matrices_and_cv_pred_proba
lnl = LearningWithNoisyLabels(clf=model, seed=0)
lnl.fit(X=train_dataset_array, s=train_class_array)
# Estimate the predictions you would have gotten by training with *no* label errors.
predicted_test_labels = lnl.predict(test_dataset_array)

print("==== Noisy LabelのConsistent Learningによる推定 ====")
y_pred = torch.argmax(predicted_test_labels, dim=1)
print(classification_report(testset.targets, y_pred))

est_py, est_nm, est_inv, confident_joint, psx = estimate_py_noise_matrices_and_cv_pred_proba(
    X = train_dataset_array, s=train_class_array, clf=model, seed=0
)

#y_pred = torch.argmax(psx, dim=1)
y_pred = np.argmax(psx, axis=1)

print("==== Consistent Learningによるエラーマトリックス推定 ====")
print(confusion_matrix(trainset.targets, y_pred))
print(classification_report(trainset.targets, y_pred))

print(confident_joint)
np.set_printoptions(suppress=True, precision=2)
print(est_nm)

print("==== Consistent Learningによるエラーインデックス ====")
from cleanlab.pruning import get_noise_indices
est_err = get_noise_indices(s=train_class_array, psx=psx, inverse_noise_matrix=est_inv, confident_joint=confident_joint)
print(est_err)
print(classification_report(err, est_err))

import csv

with open('./noise_estimate.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['true_label', 'noise_label', 'estimated_error', 'true_error'])
    out = list(zip(*[trainset.targets.tolist(), train_class_array, est_err, err]))
    writer.writerows(out)
