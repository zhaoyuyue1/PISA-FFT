
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.fft_gated import FFTPermeabilityPredictorPatchPhysics
from data.patch_loader import PermeabilityDataset, create_dataloaders
from trainers.runner import train_model, count_parameters, measure_end_to_end_latency

def execute_pipeline(seed=42, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    patch_size = 56  # this value is fixed
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.225, 0.224]),
        ToTensorV2()
    ])

    dataset = PermeabilityDataset(
        image_dir='/absolute/path/to/image/folder',
        csv_file='/absolute/path/to/metadata.csv',
        patch_size=patch_size,
        transform=transform
    )

    train_loader, valid_loader, test_loader, valid_dataset, train_dataset = create_dataloaders(
        dataset, seed=seed, batch_size=batch_size
    )

    model = FFTPermeabilityPredictorPatchPhysics(
        patch_size=patch_size,
        embed_dim=96,
        num_heads=8,
        depth=12
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}, Trainable: {trainable_params:,}")

    avg_latency = measure_end_to_end_latency(
        model, device, valid_dataset if len(valid_dataset) > 0 else train_dataset,
        warmup=5, runs=30
    )
    print(f"End-to-end average latency: {avg_latency:.2f} ms/sample")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    output_path = f'training_output/fft_patchphysics_bs{batch_size}_ps{patch_size}'

    test_r2, test_rmse, test_medare, test_mse = train_model(
        model=model,
        output_path=output_path,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        num_epochs=100,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        patience=20
    )

    print(f"Final Test -> R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MedARE: {test_medare:.2f}%, MSE: {test_mse:.4f}")
