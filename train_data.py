from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets

from augmentation import get_transforms

USE_TRAIN_SUBSET_ONLY = True

TRAIN_SAMPLES_PER_CLASS = 50

def get_train_dataset_loader(
    data_dir,
    batch_size,
    generator_train,
):
    assert USE_TRAIN_SUBSET_ONLY, "USE_TRAIN_SUBSET_ONLY must be True"

    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=USE_TRAIN_SUBSET_ONLY,  # True
        download=True,
        transform=get_transforms(train=True),
    )

    if TRAIN_SAMPLES_PER_CLASS is not None:
        targets = train_dataset.targets  # length 50_000
        num_classes = 100
        indices = []
        counts = {c: 0 for c in range(num_classes)}

        for idx, y in enumerate(targets):
            c = int(y)
            if counts[c] < TRAIN_SAMPLES_PER_CLASS:
                indices.append(idx)
                counts[c] += 1
            if all(counts[c] >= TRAIN_SAMPLES_PER_CLASS for c in range(num_classes)):
                break

        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator_train,
    )

    return train_dataset, train_loader