pip3 install "jax[tpu]"
pip3 install flax
pip3 install einops
pip3 install tensorboardX
pip3 install google-cloud-storage
pip3 install orbax-checkpoint
pip3 install gcsfs
pip3 install tensorflow # -cpu # needed for io I guess
pip3 install soundfile
pip3 install tqdm

pip3 uninstall orbax-checkpoint -y
pip3 install orbax-checkpoint==0.11.28