import wandb
from train_ae import train_lstm





if __name__ == "__main__":
    import sys
    import argparse

    if len(sys.argv) > 1 and sys.argv[1] == "lstm":
        mode = sys.argv[1]
        parser = argparse.ArgumentParser(description="Train ConvLSTM Autoencoder with Ablation Studies")
        parser.add_argument("mode", type=str, help="Training mode")

        # loss ablation arguments
        parser.add_argument("--loss-type", type=str, default="l1", choices=["l1", "mse"],
                          help="Reconstruction loss type: l1 or mse (default: l1)")
        parser.add_argument("--ms-ssim-weight", type=float, default=0.5,
                          help="Weight for MS-SSIM loss (default: 0.5, set to 0 to disable)")
        parser.add_argument("--rec-weight", type=float, default=0.5,
                          help="Weight for reconstruction loss (default: 0.5, set to 0 to disable)")
        parser.add_argument("--temporal-weight", type=float, default=0.1,
                          help="Weight for temporal smoothness loss (default: 0.1, set to 0 to disable)")
        #model layer ablations
        parser.add_argument("--dropout-rate", type=float, default=0.1,
                          help="Dropout rate (default: 0.1, set to 0 to disable)")
        parser.add_argument("--no-lstm", action="store_true",
                          help="Disable ConvLSTM (no temporal modeling)")
        parser.add_argument("--no-residual", action="store_true",
                          help="Disable residual connections")
        parser.add_argument("--no-batchnorm", action="store_true",
                          help="Disable batch normalization")
        parser.add_argument("--name", type=str, default="", help="model name")
        parser.add_argument("--size", type=int, default=4096, help="lat dimensions")

        parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
        parser.add_argument("--epochs", type=int, default=25, help="epochs")
        parser.add_argument("--warm-restarts", action="store_true", help="turn on warm restarts lr scheduling, default is cosine annealing decreasing over the the whole run")
        args = parser.parse_args()

        train_lstm(
            loss_type=args.loss_type,
            ms_ssim_weight=args.ms_ssim_weight,
            rec_weight=args.rec_weight,
            temporal_weight=args.temporal_weight,
            dropout_rate=args.dropout_rate,
            use_lstm=not args.no_lstm,
            use_residual=not args.no_residual,
            use_batchnorm=not args.no_batchnorm,
            model_name = args.name,
            latent_size = args.size,
            lr=args.lr,
            epochs=args.epochs,
            warm_restarts=args.warm_restarts

