conda create -n pk_SS python=3.8
conda activate pk_SS
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets seqeval numpy scikit-learn evaluate accelerate -U
