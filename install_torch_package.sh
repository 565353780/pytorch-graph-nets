export TORCH_VERSION_='1.9.0'
export VISION_VERSION_='0.10.0'
export AUDIO_VERSION_='0.9.0'
export CUDA_VERSION_='cu111'

pip install torch==$TORCH_VERSION_+$CUDA_VERSION_ torchvision==$VISION_VERSION_+$CUDA_VERSION_ torchaudio==$AUDIO_VERSION_ -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-$TORCH_VERSION_+$CUDA_VERSION_.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-$TORCH_VERSION_+$CUDA_VERSION_.html
pip install torch-geometric
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-$TORCH_VERSION_+$CUDA_VERSION_.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-$TORCH_VERSION_+$CUDA_VERSION_.html

