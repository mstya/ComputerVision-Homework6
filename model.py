import csv
import os

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
        class_dirs = sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))
        for ci, class_name in enumerate(class_dirs):
            class_dir = os.path.join(path, class_name)
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
        class_dirs = sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))
        for ci, class_name in enumerate(class_dirs):
            class_dir = os.path.join(path, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    fpath = os.path.join(class_dir, fname)
                    self.files.append(fpath)
                    self.labels.append(ci)

CLASS_NAMES = ['Agriculture', 'City', 'Dessert', 'Forest']

class DatasetLoader:
    def __init__(self, data_path):
        self.full_dataset = LandDataset(data_path)
        datasets = random_split(self.full_dataset, [0.8, 0.1, 0.1])
        self.train_dataset = datasets[0]
        self.val_dataset = datasets[1]
        self.test_dataset = datasets[2]
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=len(self.val_dataset), shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)

    def test_file_paths(self):
        return [self.full_dataset.files[i] for i in self.test_dataset.indices]

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

    def test(self, csv_path='error_analysis.csv'):
        print("Final testing...")
        self.model.eval()
        file_paths = self.loader.test_file_paths()
        with torch.no_grad():
            X, Y = next(iter(self.loader.test_loader))
            logits = self.model(X)
            predicted = torch.argmax(logits, dim=1)

        correct = (predicted == Y).sum().item()
        print(f"Final accuracy: {correct / len(Y) * 100:.2f}%")

        errors_per_class = [0] * len(CLASS_NAMES)
        rows = []
        img_number = 1
        for idx in range(len(Y)):
            true_cls = Y[idx].item()
            pred_cls = predicted[idx].item()
            if pred_cls == true_cls:
                continue
            one_hot = [1 if c == pred_cls else 0 for c in range(len(CLASS_NAMES))]
            rows.append([img_number, true_cls] + one_hot + [''])
            errors_per_class[true_cls] += 1
            img_number += 1

        total_errors = sum(errors_per_class)
        if total_errors > 0:
            error_pcts = [f"{(e / total_errors * 100):.1f}%" for e in errors_per_class]
        else:
            error_pcts = ['0.0%'] * len(CLASS_NAMES)

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Image #', 'OrigClass'] + CLASS_NAMES + ['Comments'])
            writer.writerows(rows)
            writer.writerow(['Error%', ''] + error_pcts + [''])

        print(f"Error analysis saved to {csv_path} ({total_errors} misclassified out of {len(Y)})")

        base, ext = os.path.splitext(csv_path)
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            cls_rows = [r for r in rows if r[1] == cls_idx]
            cls_errors = len(cls_rows)
            if cls_errors > 0:
                cls_pcts = [f"{(sum(r[2 + c] for r in cls_rows) / cls_errors * 100):.1f}%" for c in range(len(CLASS_NAMES))]
            else:
                cls_pcts = ['0.0%'] * len(CLASS_NAMES)
            cls_path = f"{base}_{cls_name.lower()}{ext}"
            with open(cls_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Image #', 'OrigClass'] + CLASS_NAMES + ['Comments'])
                writer.writerows(cls_rows)
                writer.writerow(['Error%', ''] + cls_pcts + [''])
            print(f"  {cls_name}: {cls_errors} errors → {cls_path}")

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

    model = Model(path, epoch=50, learning_rate=0.001, momentum=0.9)
    model.train()
    model.test()

    # Беру всі зображення із датасету, перемішую їх і беру 60 штук. Далі проганяю їх через inference
    # там відображаю 4 плота із класифікованими зображеннями.
    # util = FilesUtil()
    # files = util.load_data(path)
    # np.random.shuffle(files)
    # files = files[:60]
    # classes = [[], [], [], []]
    # for index, path in enumerate(files):
    #     clazz = model.inference(path)
    #     classes[clazz].append(path)
    #
    # for clazz in classes:
    #     plt.figure(figsize=(10, 10))
    #     for index, file in enumerate(clazz):
    #         plt.subplot(5, 5, index + 1)
    #         img = Image.open(file)
    #         img = np.array(img)
    #         plt.imshow(img)
    #         plt.axis('off')
    #     plt.show()


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
# Залишаю Mini-Batch із Momentum, із ним результати найстабільніші. 80% на тестовому наборі, схоже це потолок на поточних даних і на поточній архітектурі.
# Далі треба переходити на CNN.

# Додав augment.py який на основі кожного зображення генерує 3 додаткові, завдяку повороту зображення. Для фото із супутників працює добре.
# Завдяки додатковим даним вийшли підняти точність до 88.20%.

# Також додав генерацію репортів по помилкам по кожному із класів для додаткового аналізу проблемних зображень.