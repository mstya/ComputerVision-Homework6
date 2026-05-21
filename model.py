import os

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn, relu
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

class FilesUtil:
    def __init__(self):
        self.files = []
        self.labels = []

    def load_data(self, path):
        for ci, class_name in enumerate(os.listdir(path)):
            class_dir = os.path.join(path, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    fpath = os.path.join(class_dir, fname)
                    self.files.append(fpath)
                    self.labels.append(ci)
        return self.files

class LandDataset(Dataset):
    def __init__(self, data_path):
        fileUtil = FilesUtil()
        fileUtil.load_data(data_path)
        self.files = fileUtil.files
        self.labels = fileUtil.labels

        self.transform = Compose([
            Resize((64, 64)),
            ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = self.transform(Image.open(self.files[i]))
        return img, self.labels[i]

    def load_data(self, path):
        for ci, class_name in enumerate(os.listdir(path)):
            class_dir = os.path.join(path, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    fpath = os.path.join(class_dir, fname)
                    self.files.append(fpath)
                    self.labels.append(ci)

class DatasetLoader:
    def __init__(self, data_path):
        self.full_dataset = LandDataset(data_path)
        datasets = random_split(self.full_dataset, [0.7, 0.15, 0.15])
        self.train_dataset = datasets[0]
        self.val_dataset = datasets[1]
        self.test_dataset = datasets[2]
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=len(self.val_dataset), shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(12288, 512)
        self.layer2 = nn.Linear(512, 128)
        self.layer3 = nn.Linear(128, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.flatten(1)
        x = self.layer1(x)
        x = relu(x)
        #x = self.dropout(x)
        x = self.layer2(x)
        x = relu(x)
        #x = self.dropout(x)
        return self.layer3(x)

class Model:
    def __init__(self, path, epoch=10, learning_rate=0.001, momentum=0.9):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.loader = DatasetLoader(path)
        self.model = NN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=momentum)
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        losses = []
        accuracies = []
        for epoch in range(self.epoch):
            total_loss = 0.0
            self.model.train()
            for X, Y in self.loader.train_loader:
                # self.model.zero_grad()
                self.optimizer.zero_grad()
                y_pred = self.model(X)
                loss = self.criterion(y_pred, Y)
                loss.backward()
                # with torch.no_grad():
                #     for param in self.model.parameters():
                #         param -= self.learning_rate * param.grad
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.loader.train_loader)
            losses.append(avg_loss)

            print(f"Epoch [{epoch + 1:2d}/{self.epoch}]  Avg Loss per batch: {avg_loss:.4f}")

            self.model.eval()
            with torch.no_grad():
                for X, Y in self.loader.val_loader:
                    y_val_pred = self.model(X)
                    classI = torch.argmax(y_val_pred, dim=1)
                    correctAnswSum = (classI == Y).sum() / len(Y) * 100
                    accuracies.append(correctAnswSum)
                    print("Val dataset accuracy %: ", correctAnswSum.item())

        plt.plot(losses, label='Training Loss')
        plt.title('Model Loss Over Time')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.plot(accuracies, label='Accuracy')
        plt.title('Model Accuracy Over Time')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def test(self):
        print("Final testing...")
        accuracies = []
        for i, (X, Y) in enumerate(self.loader.test_loader):
            y_val_pred = self.model(X)
            classI = torch.argmax(y_val_pred, dim=1)

            correctAnswSum = (classI == Y).sum() / len(Y) * 100
            accuracies.append(correctAnswSum)
            print("Final %: ", correctAnswSum.item())

    def inference(self, file):
        img = Compose([
            Resize((64, 64)),
            ToTensor()
        ])(Image.open(file))
        img = torch.unsqueeze(img, 0)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(img)
            return torch.argmax(pred, dim=1)

if __name__ == "__main__":
    path = "./Aerial_Landscapes"
    settings = {}

    model = Model(path, epoch=10, learning_rate=0.001, momentum=0.9)
    model.train()
    model.test()

    # Беру всі зображення із датасету, перемішую їх і беру 60 штук. Далі проганяю їх через inference
    # там відображаю 4 плота із класифікованими зображеннями.
    util = FilesUtil()
    files = util.load_data(path)
    np.random.shuffle(files)
    files = files[:60]
    classes = [[], [], [], []]
    for index, path in enumerate(files):
        clazz = model.inference(path)
        classes[clazz].append(path)

    for clazz in classes:
        plt.figure(figsize=(10, 10))
        for index, file in enumerate(clazz):
            plt.subplot(5, 5, index + 1)
            img = Image.open(file)
            img = np.array(img)
            plt.imshow(img)
            plt.axis('off')
        plt.show()


#######################################################################################################################
# Update                                                        Loss        Accuracy on val set  Comments
######################################################################################################################
# Batch GD 10 epoch,            LR=0.1                          1.2862      38.5417%             Замало епох явно
# Batch GD 10 epoch,            LR=0.01                         1.3261      25.2083%             Заради цікавості дивлюсь задаю менший LR
# Batch GD 100 epoch,           LR=0.01                         1.1552      37.2917%             Збільшую кількість епох, навчання плавне, але повільно
# Batch GD 100 epoch,           LR=0.1                          1.1552      47.5000%             Learning Rate завеликий, loss стрибає
# Mini-Batch (32) GD 100 epoch, LR=0.1                          1.3824      25%                  після епохи ~55 loss різко злетів до 1.38 і застиг — а accuracy впав до 25%. Точність випадкова. gradient explosion?
# Mini-Batch (32) GD 100 epoch, LR=0.001                        0.5390      77.7083%             Loss плавно спадає, можна ще збільшити кількість епох
# Mini-Batch (32) GD 300 epoch, LR=0.001                        0.3247      80%                  val accuracy вийшла на плато приблизно на 80% після епохи ~50, хоча train loss продовжує падати.
# Mini-Batch (32) SGD 300 epoch, LR=0.001                       0.3003      83.1250              оновлюваги ваги за допомоги PyTorch SGD optimizer.
# Mini-Batch (32) Momentum 300 epoch, LR=0.001                  0.0067      80.6250              Додав Momentum, сильно збільшилась швидкість навчання, але точність в результаті так сама в середньому. Loss майже на 0, але Точність на валідаційному наборі 80%, схоже тут overfitting.
# Mini-Batch (32) Momentum 300 epoch, Dropout 0.5, LR=0.001     0.3010      82.0833              Dropout не допомогім
# Mini-Batch (32) RMSprop  300 epoch, Dropout 0.5, LR=0.001     0.1693      83.1250              RMSprop
# Mini-Batch (32) Adam     300 epoch, LR=0.001                  0.0366      80.8333              Adam
# Залашаю Mini-Batch із Momentum, із ним результати найстабільніші. 80% на тестовому наборі, схоже це потолок на поточних даних і на поточній архітектурі.
# Далі треба переходити на CNN.