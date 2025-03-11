import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch.optim as optim
import fire

from sklearn.model_selection import train_test_split
from sklearn import preprocessing



class ClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(ClassificationModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  
        )
       

    def forward(self, x):
        return self.fc(x)

def pre_process(test_size):
    train_data = torch.load("./train_data.pt") 
    train_label = torch.load("./train_label.pt") 

    train_data, test_data, train_label, test_label = train_test_split(
    train_data, train_label, test_size=test_size, random_state=42
    )

    scaler = preprocessing.StandardScaler()
    train_data = torch.tensor(scaler.fit_transform(train_data)).float()
    test_data = torch.tensor(scaler.transform(test_data)).float()
    import joblib
    joblib.dump(scaler, './scaler.pkl')


    train_label = train_label.float()
    test_label = test_label.float()
    return train_data, train_label, test_data, test_label

def main(
    test_size: float = 0.2,
    batch_size: int = 1024,
    num_epochs: int = 5000,
    threshold_value: float = 0.65
    ):
    train_data, train_label, test_data, test_label = pre_process(test_size)
    input_size = train_data.shape[1] 
    model = ClassificationModel(input_size)

    criterion = nn.BCELoss()  
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)

    train_dataset = TensorDataset(train_data, train_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_accuracies = []
    test_accuracies = []
    for epoch in range(num_epochs):
        for batch_data, batch_labels in train_loader:
            outputs = model(batch_data).squeeze() 
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        model.eval()
        with torch.no_grad():
            train_outputs = model(train_data).squeeze()
            train_predictions = (train_outputs > 0.5).float()
            train_accuracy = (train_predictions == train_label).sum().item() / train_label.size(0)

            test_outputs = model(test_data).squeeze()
            test_predictions = (test_outputs > 0.5).float()
            test_accuracy = (test_predictions == test_label).sum().item() / test_label.size(0)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, "
            f"Train Accuracy: {train_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%", flush=True)
        if test_accuracy >= threshold_value and train_accuracy >= threshold_value:
            torch.save(model.state_dict(), f"./best_{epoch}_{test_accuracy}_classification_model.pth")
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label="Train Accuracy", marker='o')
    plt.plot(test_accuracies, label="Test Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("./test.png")





if __name__ == "__main__":
    fire.Fire(main)








