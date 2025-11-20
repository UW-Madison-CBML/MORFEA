"""
Test script: Ensure model can run correctly
No actual training, only verifies forward pass, loss computation, etc.
"""
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model import ConvLSTMAutoencoder
from losses import reconstruction_loss, temporal_smoothness_loss


def test_model():
    """Test model forward pass"""
    print("=" * 60)
    print("Testing ConvLSTM Autoencoder")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Model parameters
    batch_size = 4
    seq_len = 20
    H, W = 128, 128
    
    # Create model
    print("1. Creating model...")
    model = ConvLSTMAutoencoder(
        seq_len=seq_len,
        input_channels=1,
        encoder_hidden_dim=256,
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        use_classifier=True,
        num_classes=2
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}\n")
    
    # Create random input
    print("2. Creating random input...")
    x = torch.randn(batch_size, seq_len, 1, H, W).to(device)
    print(f"   Input shape: {x.shape}\n")
    
    # Forward pass
    print("3. Forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"   Reconstruction shape: {output['reconstruction'].shape}")
    print(f"   z_seq shape: {output['z_seq'].shape}")
    print(f"   z_last shape: {output['z_last'].shape}")
    if 'logits' in output:
        print(f"   Logits shape: {output['logits'].shape}\n")
    
    # Verify shapes
    assert output['reconstruction'].shape == x.shape, \
        f"Reconstruction shape mismatch: {output['reconstruction'].shape} vs {x.shape}"
    assert output['z_seq'].shape[0] == batch_size, "Batch size mismatch"
    assert output['z_seq'].shape[1] == seq_len, "Sequence length mismatch"
    print("   ✓ Shape checks passed\n")
    
    # Test loss computation
    print("4. Testing loss computation...")
    rec_loss, rec_details = reconstruction_loss(
        output['reconstruction'], x,
        l1_weight=0.5,
        ms_ssim_weight=0.5
    )
    smooth_loss = temporal_smoothness_loss(output['z_seq'], weight=0.1)
    
    print(f"   Reconstruction loss: {rec_loss.item():.4f}")
    print(f"     - L1: {rec_details['l1_loss']:.4f}")
    print(f"     - MS-SSIM: {rec_details['ms_ssim_loss']:.4f}")
    print(f"     - MS-SSIM value: {rec_details['ms_ssim_value']:.4f}")
    print(f"   Smoothness loss: {smooth_loss.item():.4f}\n")
    
    # Test backward pass
    print("5. Testing backward pass...")
    model.train()
    # Recompute loss (ensure in train mode)
    output_train = model(x)
    rec_loss_train, _ = reconstruction_loss(
        output_train['reconstruction'], x,
        l1_weight=0.5,
        ms_ssim_weight=0.5
    )
    smooth_loss_train = temporal_smoothness_loss(output_train['z_seq'], weight=0.1)
    total_loss = rec_loss_train + smooth_loss_train
    total_loss.backward()
    
    # Check gradients
    has_grad = any(p.grad is not None for p in model.parameters())
    print(f"   Gradients computed: {has_grad}\n")
    
    # Test encode/decode separately
    print("6. Testing encode/decode separately...")
    z_seq, z_last = model.encode(x)
    x_rec = model.decode(z_seq)
    assert x_rec.shape == x.shape, "Decode shape mismatch"
    print("   ✓ Encode/decode test passed\n")
    
    # Test different batch sizes
    print("7. Testing different batch sizes...")
    model.eval()  # Use eval mode to avoid BatchNorm issues with batch_size=1
    for bs in [1, 2, 8]:
        x_test = torch.randn(bs, seq_len, 1, H, W).to(device)
        with torch.no_grad():
            output_test = model(x_test)
        assert output_test['reconstruction'].shape[0] == bs, \
            f"Batch size {bs} failed"
    print("   ✓ Different batch sizes work\n")
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nModel is ready for training on CHTC H200 GPU!")
    print("\nNext steps:")
    print("1. Upload to GitHub")
    print("2. Use CHTC setup scripts to run on H200")
    print("3. Monitor training with chtc_monitor.sh")


if __name__ == "__main__":
    test_model()

