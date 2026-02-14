#!/bin/bash

echo "============================================================"
echo "============================================================"
echo ""
echo ""
echo ""
echo "   cd \"/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02\""
echo "   scp export_all_frame_latents_direct.py rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/"
echo ""
echo "   python3 export_all_frame_latents_direct.py \\"
echo "     --checkpoint /staging/groups/bhaskar_group/rho9/checkpoints/checkpoint_epoch_50.pt \\"
echo "     --data_root /staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset \\"
echo "     --output /staging/groups/bhaskar_group/rho9/model_latents/v1_baseline/latents_all_frames_direct.npz"
echo ""
echo ""
echo "============================================================"








