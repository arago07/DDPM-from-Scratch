from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loader(batch_size=64, train=True, root="./data", num_workers=0):
    
    """
    MNIST Dataloader.
    
    Args:
        batch_size: 배치 사이즈
        train: True -학습 데이터, False -테스트 데이터
        root: MNIST 데이터셋이 저장될 경로
        num_workers: DataLoader의 num_workers(multiprocessing) 파라미터
    
    Returns:
        torch DataLoader
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = datasets.MNIST(
        root=root,
        train=train,
        download=True,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,  # shuffle only for training
        num_workers=num_workers,
        drop_last=True, # drop the last incomplete batch
    )
    return loader

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=== MNIST data loading test ===\n")

    loader = get_mnist_loader(batch_size=64)
    print(f"Dataset size: {len(loader.dataset):,}")
    print(f"Batch count:  {len(loader):,}")

    # 첫 배치
    images, labels = next(iter(loader))
    print(f"\nBatch shape:  {tuple(images.shape)}")
    print(f"Image dtype:  {images.dtype}")
    print(f"Value range:  [{images.min():.3f}, {images.max():.3f}]   (≈ [-1, 1] 이어야 함)")
    print(f"Labels (first 8): {labels[:8].tolist()}")

    # 8개 시각화 (2x4 grid)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap='gray', vmin=-1, vmax=1)
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis('off')
    plt.suptitle("MNIST samples (normalized to [-1, 1])")
    plt.tight_layout()
    plt.savefig("mnist_sample.png", dpi=100, bbox_inches='tight')
    print("\nSaved: mnist_sample.png")


