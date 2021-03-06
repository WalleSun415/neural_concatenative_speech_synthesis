import tensorflow as tf


_pad = '_'
_punctuation = '!\'(),.:;?'
_special = '-'
_letters = 'abcdefghijklmnopqrstuvwxyz '

# Export all symbols:
symbols = list(_special) + list(_punctuation)
letters = [_pad] + list(_letters)


def load_hparams():
    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=5000,
        exp_path = "runs/exp12",
        seed=1234,
        glued_num=1,
        model_save_path="NeuralConcate.pth",

    ################################
        # Data Parameters             #
        ################################
        training_files='filelists/temp_colab_audio_text_train_filelist.txt',
        validation_files='filelists/colab_audio_text_val_filelist.txt',
        training_files_base='filelists/colab_audio_text_train_filelist.txt',

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(letters),
        symbols_embedding_dim=512,

        # Audio Encoder Parameters
        encoder_rnn_dim=512,
        prenet_dim=512,
        audio_kernel_size=11, # audio
        audio_stride=11,
        decoder_kernel_size=5,  # text
        text_stride=5,
        # Text Decoder parameters
        decoder_rnn_dim=1024,

        # Mel Decoder parameters
        mel_decoder_rnn_dim=512,
        rnn_dropout=0.0,
        ################################
        # Optimization Hyperparameters #
        ################################
        batch_size=16,
        learning_rate=1e-3,
    )

    return hparams
