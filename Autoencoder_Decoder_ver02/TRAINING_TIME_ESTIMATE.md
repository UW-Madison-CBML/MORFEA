# Training Time Estimate

## Configuration
- **Epochs**: 50
- **Batch size**: 8
- **Sequence length**: 20
- **GPU**: H200 (very fast!)
- **Model**: ConvLSTM Autoencoder (13.4M parameters)

## Time Breakdown

### Per Epoch Estimate
- **Best case** (small dataset, ~100 samples): ~5-10 minutes/epoch
- **Typical case** (medium dataset, ~500-1000 samples): ~10-20 minutes/epoch
- **Worst case** (large dataset, ~2000+ samples): ~20-30 minutes/epoch

### Total Training Time

| Dataset Size | Per Epoch | Total (50 epochs) |
|--------------|-----------|-------------------|
| Small (~100) | 5-10 min | **4-8 hours** |
| Medium (~500-1000) | 10-20 min | **8-17 hours** |
| Large (~2000+) | 20-30 min | **17-25 hours** |

## Most Likely Scenario

Based on typical IVF datasets:
- **Per epoch**: ~15 minutes
- **Total training**: **~12-15 hours**

## Factors Affecting Time

### Faster:
- ✅ H200 GPU (very powerful)
- ✅ Batch size 8 (good balance)
- ✅ Sequence length 20 (not too long)
- ✅ Model size 13.4M (reasonable)

### Slower:
- ⚠️ Large dataset size
- ⚠️ 50 epochs (comprehensive training)
- ⚠️ MS-SSIM loss (more computation than simple MSE)

## Checkpoints

- Checkpoints saved every **5 epochs**
- So you'll have checkpoints at: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
- Can stop early if needed (use checkpoint from epoch 45, etc.)

## Real-Time Monitoring

You can check progress anytime:
```bash
# See current epoch
tail -n 20 logs/train_2650361_0.out | grep -i epoch

# Estimate remaining time
# If at epoch 10/50 and took 2 hours:
# Remaining: (50-10)/10 * 2 hours = 8 hours
```

## Quick Estimate Formula

```
Total time ≈ (Current epoch / Elapsed time) * (50 - Current epoch)
```

Example:
- At epoch 10, elapsed 2 hours
- Remaining: (10/2) * (50-10) = 5 * 40 = 200 minutes = ~3.3 hours
- Total: ~5.3 hours





