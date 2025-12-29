
def train():
    """Original training with MS-SSIM loss and distributed training"""
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    torch.cuda.empty_cache()
    torch.autograd.detect_anomaly(True)
    rank, world_size, local_rank = setup_distributed()
    DEVICE = torch.device(f"cuda:{local_rank}")

    # Only print on main process
    is_main = rank == 0
    run = None
    # Initialize wandb only on main process
    if is_main:
        wandb.login(key=os.getenv("WANDB_KEY"))
        run = wandb.init(
            entity="jenslundsgaard7-uw-madison",
            project="IVF-Training",
            config={
                "learning_rate": 0.005,
                "architecture": "Conv LSTM Autoencoder",
                "dataset": "https://zenodo.org/records/7912264",
                "epochs": 10,
                "world_size": world_size,
                "loss": "MS-SSIM + L1",
            },
        )

        login(os.getenv("HF_KEY"))
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        print(DEVICE)

    model = Model()
    if os.path.exists("model_weights.pth"):
        try:
            checkpoint = torch.load("model_weights.pth", map_location=DEVICE, weights_only=True)
            model.load_state_dict(checkpoint)
            if is_main:
                print("Loaded model weights")
        except Exception as e:
            if is_main:
                print(f"Error loading weights: {e}")
                torch.save(model.state_dict(), "model_weights.pth")
    else:
        if is_main:
            torch.save(model.state_dict(), "model_weights.pth")
    model = model.to(DEVICE)

    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
