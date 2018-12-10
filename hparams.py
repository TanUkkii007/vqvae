import tensorflow as tf

default_params = tf.contrib.training.HParams(

    # Encoder
    encoder_num_hiddens=128,
    encoder_num_residual_hiddens=32,
    encoder_num_residual_layers=2,

    # Decoder
    decoder_num_hiddens=128,
    decoder_num_residual_hiddens=32,
    decoder_num_residual_layers=2,

    embedding_dim=64,
    num_embeddings=512,
    commitment_cost=0.25,

    # VectorQuantizer
    vector_quantizer="VectorQuantizer",
    # EMA
    ema_decay=0.99,
    ema_epsilon=1e-5,
    # EM
    sampling_count=10,

    # Training
    batch_size=32,
    learning_rate=3e-4,
    save_summary_steps=100,
    save_checkpoints_steps=500,
    keep_checkpoint_max=200,
    keep_checkpoint_every_n_hours=1,
    log_step_count_steps=1,
    shuffle_buffer_size=4,

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
