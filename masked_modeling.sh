CUDA_VISIBLE_DEVICES=1 python3 mutex/main_masked_modeling.py \
        benchmark_name=rw_h2r \
        policy.task_spec_modalities=img_vid \
        policy.add_mim=False policy.add_mgm=False policy.add_mrm=True \
        policy.add_mfm=True policy.add_maim=False policy.add_magm=False \
        folder=./dataset \
        hydra.run.dir=experiments/mutex \
        data=h2r 
