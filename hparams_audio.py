import tensorflow as tf

default_params = tf.contrib.training.HParams(

    # Audio
    sample_rate=16000,
    num_sil_samples=80,

    # Encoder
    encoder_num_hiddens=128,
    encoder_num_layers=6,

    # WaveNet
    quantization_levels=256,
    n_logistic_mix=8,
    local_condition_label_dim=64,  # == embedding_dim
    use_global_condition=True,
    global_condition_cardinality=108,
    speaker_id_offset=225,
    filter_width=3,
    residual_channels=64,
    dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
               1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    skip_channels=128,
    out_channels=3 * 8,  # 3*n_logistic_mix
    use_causal_conv_bias=True,
    use_filter_gate_bias=True,
    use_output_bias=True,
    use_skip_bias=True,
    use_postprocessing1_bias=True,
    use_postprocessing2_bias=True,

    max_output_length=80000,

    # VQVAE
    embedding_dim=64,
    num_embeddings=512,
    commitment_cost=0.25,

    # Training
    batch_size=1,
    learning_rate=3e-4,
    save_summary_steps=100,
    save_checkpoints_steps=500,
    keep_checkpoint_max=200,
    keep_checkpoint_every_n_hours=1,
    log_step_count_steps=1,
    approx_min_target_length=100,
    suffle_buffer_size=64,
    batch_bucket_width=50,
    batch_num_buckets=50,
    interleave_cycle_length_cpu_factor=1.0,
    interleave_cycle_length_min=4,
    interleave_cycle_length_max=16,
    interleave_buffer_output_elements=200,
    interleave_prefetch_input_elements=200,
    prefetch_buffer_size=4,

    # Validation
    num_evaluation_steps=32,
    eval_start_delay_secs=3600,  # 1h: disable time based evaluation
    eval_throttle_secs=86400,  # 24h: disable time based evaluation

    # Misc
    logfile="log.txt",
)


def hparams_debug_string(hparams):
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
