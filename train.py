import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from models.cbam_resnet import CBAMResNet18
from utils.dataset import get_dataloaders

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100.0 * correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100.0 * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Perangkat Komputasi: {device}")

    seeds = [0, 1, 2]
    epochs = 30
    batch_size = 64
    lr = 0.01

    for seed in seeds:
        print(f"\n=================== Pelatihan Seed {seed} ===================")
        set_seed(seed)
        train_loader, val_loader, test_loader = get_dataloaders(seed=seed, batch_size=batch_size)

        model = CBAMResNet18(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            t_elapsed = time.time() - t0

            print(f"Epoch [{epoch:02d}/{epochs}] ({t_elapsed:.1f}s) | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f">> Hasil Akhir Seed {seed} - Akurasi Test: {test_acc:.2f}%")

if __name__ == "__main__":
    main()