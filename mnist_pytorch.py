import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

x_train = train_dataset.data.numpy()
y_train = train_dataset.targets.numpy()
x_test = test_dataset.data.numpy()
y_test = test_dataset.targets.numpy()

print(f"训练集样本数: {x_train.shape[0]}")
print(f"测试集样本数: {x_test.shape[0]}")
print(f"图像尺寸: {x_train.shape[1]}×{x_train.shape[2]}")
print(f"训练集标签分布: {np.bincount(y_train)}")
print(f"测试集标签分布: {np.bincount(y_test)}")

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.savefig('mnist_samples.png')
plt.show()

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

baseline_model = BaselineModel().to(device)
cnn_model = CNNModel().to(device)

criterion = nn.CrossEntropyLoss()
baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

print("基准模型结构:")
print(baseline_model)
print("\nCNN模型结构:")
print(cnn_model)

def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}
    
    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            total += inputs.size(0)
        
        train_acc = running_correct / total
        train_loss = running_loss / total
        
        val_acc, val_loss = evaluate(model, test_loader, criterion)
        
        history['accuracy'].append(train_acc)
        history['loss'].append(train_loss)
        history['val_accuracy'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return history

def evaluate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            total += inputs.size(0)
    
    return running_correct / total, running_loss / total

print("\n" + "="*60)
print("训练基准模型（逻辑回归）")
print("="*60)
history_baseline = train(baseline_model, train_loader, criterion, baseline_optimizer, epochs=10)

print("\n" + "="*60)
print("训练CNN模型")
print("="*60)
history_cnn = train(cnn_model, train_loader, criterion, cnn_optimizer, epochs=10)

baseline_test_acc, baseline_test_loss = evaluate(baseline_model, test_loader, criterion)
cnn_test_acc, cnn_test_loss = evaluate(cnn_model, test_loader, criterion)

print("\n" + "="*60)
print("模型性能对比表")
print("="*60)
print(f"{'模型':<20} {'测试准确率':<15} {'测试损失':<15}")
print("-"*60)
print(f"{'基准模型(逻辑回归)':<20} {baseline_test_acc:<15.4f} {baseline_test_loss:<15.4f}")
print(f"{'CNN模型':<20} {cnn_test_acc:<15.4f} {cnn_test_loss:<15.4f}")
print("="*60)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_baseline['accuracy'], label='基准模型训练准确率')
plt.plot(history_baseline['val_accuracy'], label='基准模型验证准确率')
plt.plot(history_cnn['accuracy'], label='CNN模型训练准确率')
plt.plot(history_cnn['val_accuracy'], label='CNN模型验证准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('准确率对比曲线')

plt.subplot(1, 2, 2)
plt.plot(history_baseline['loss'], label='基准模型训练损失')
plt.plot(history_baseline['val_loss'], label='基准模型验证损失')
plt.plot(history_cnn['loss'], label='CNN模型训练损失')
plt.plot(history_cnn['val_loss'], label='CNN模型验证损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('损失对比曲线')

plt.tight_layout()
plt.savefig('accuracy_loss_curve.png')
plt.show()

cnn_model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = cnn_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.numpy())

y_pred = np.array(y_pred)
y_true = np.array(y_true)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[str(i) for i in range(10)],
            yticklabels=[str(i) for i in range(10)])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('CNN模型混淆矩阵')
plt.savefig('confusion_matrix.png')
plt.show()

errors = np.where(y_pred != y_true)[0]

plt.figure(figsize=(12, 12))
for i in range(min(16, len(errors))):
    idx = errors[i]
    plt.subplot(4, 4, i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"真实: {y_true[idx]}, 预测: {y_pred[idx]}")
    plt.axis('off')
plt.savefig('error_samples.png')
plt.show()

print("\n训练完成！结果已保存为图片文件。")