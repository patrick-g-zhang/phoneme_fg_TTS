import tensorflow as tf
from text import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=2000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=True,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight', "decoder.prenet.layers.0.linear_layer.weight",
                       "decoder.prenet.layers.1.linear_layer.weight", "decoder.attention_rnn.weight_ih"],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        # training_files='datasets/blizzard2013/blz13_audio_text_train_filelist.txt',
        # validation_files='datasets/blizzard2013/blz13_audio_text_val_filelist.txt',
        training_files='datasets/blizzard2013/full_blz13_audio_text_less_9_more_4train_filelist.txt',
        validation_files='datasets/blizzard2013/full_blz13_audio_text_less_9_more_4val_filelist.txt',
        text_cleaners=['english_cleaners'],
        cmudict_path="data/cmu_dictionary",

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=16000,
        filter_length=1024,
        hop_length=200,  # 12.5ms
        win_length=800,  # 50ms
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        n_spks=69 + 1,
        spks_embedding_dim=128,
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=128,
        max_decoder_steps=10000,  # 12.5 ms
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        # Reference encoder
        ref_enc_filters=[32, 32, 64, 64, 128, 128],
        ref_enc_size=[3, 3],
        ref_enc_strides=[2, 2],
        ref_enc_pad=[1, 1],
        ref_enc_gru_size=128,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=32,
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams