import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from utils.paths import resolve_data_dir

def get_dataloaders(data_dir=None, batch_size=64, img_size=128, seed=0, num_workers=2):
    data_dir = resolve_data_dir(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    for directory in (train_dir, val_dir):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Folder dataset tidak ditemukan: {directory}. "
                "Jalankan python -m utils.download_data dari root proyek."
            )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    if full_train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("Kelas pada folder train dan val harus sama.")

    targets = full_train_dataset.targets
    train_idx, internal_val_idx = train_test_split(
        range(len(targets)),
        test_size=0.10,
        random_state=seed,
        stratify=targets
    )

    train_subset = Subset(full_train_dataset, train_idx)
    val_dataset_base = datasets.ImageFolder(root=train_dir, transform=val_transform)
    internal_val_subset = Subset(val_dataset_base, internal_val_idx)

    loader_options = {"batch_size": batch_size, "num_workers": num_workers,
                      "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_subset, shuffle=True, **loader_options)
    val_loader = DataLoader(internal_val_subset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    return train_loader, val_loader, test_loader
