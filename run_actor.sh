export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python train_rlpd_simplified.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path=first_run \
    --actor \
