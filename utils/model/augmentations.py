import tensorflow as tf
import numpy as np


class MelSpectrogramAugmenter(tf.keras.layers.Layer):
    """
    Applies pink and white noise augmentation directly in the mel-spectrogram domain using TensorFlow.

    Args:
        sr: Sampling rate of the audio (required for frequency calculations).
        n_mels: Number of Mel bands.
        f_min: Minimum frequency for Mel filterbank.
        f_max: Maximum frequency for Mel filterbank.
        pink_noise_factor: Factor to control the intensity of the pink noise.
        white_noise_factor: Factor to control the intensity of the white noise.
    """

    def __init__(self, sr, f_min, f_max, pink_noise_factor=0.05, white_noise_factor=0.02, p_pink=1, p_white=1, name='spect_noise'):
        super().__init__(name=name)
        self.sr = sr
        self.n_mels = None
        self.f_min = f_min
        self.f_max = f_max
        self.pink_noise_factor = pink_noise_factor
        self.white_noise_factor = white_noise_factor
        self.p_pink = p_pink
        self.p_white = p_white

    def build(self, input_shape):
        actual_shape = input_shape[1:]
        dims = len(actual_shape)
        self.n_mels = actual_shape[0]

        import librosa

        # Calculate and store mel frequencies once
        mel_bins = librosa.mel_frequencies(n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max)

        # Calculate and store pink noise PSD once
        nyquist_rate = self.sr / 2
        num_bins = int(nyquist_rate / 1000)  # Adjust based on desired frequency resolution
        freqs = np.linspace(0.0, nyquist_rate, num_bins)
        freqs = np.maximum(freqs, 1e-6)  # Avoid division by zero
        pink_noise_psd = 1 / np.sqrt(freqs)
        mel_pink_noise_psd_numpy = np.interp(mel_bins, freqs, pink_noise_psd)

        assert len(mel_pink_noise_psd_numpy.shape) <= dims
        while len(mel_pink_noise_psd_numpy.shape) < dims:
            mel_pink_noise_psd_numpy = mel_pink_noise_psd_numpy[..., None]

        self.mel_pink_noise_psd = self.add_weight(
            shape=mel_pink_noise_psd_numpy.shape,
            initializer=lambda shape, dtype=None: mel_pink_noise_psd_numpy if dtype is None else tf.cast(mel_pink_noise_psd_numpy, dtype=dtype),
            trainable=False,
            name='mel_pink_noise_psd'
        )

        # self.zeros = self.add_weight(shape=actual_shape, initializer='zeros', trainable=False)

    def get_noise(self, shape, p=1., dtype=None):
        noise = tf.random.normal(shape, dtype=dtype, mean=0, stddev=1)
        if p < 1.:
            noise = tf.cast(tf.random.uniform(shape[0], maxval=1.) <= p, dtype=noise.dtype) * noise
        return noise

    def call(self, inputs, training=None):
        """
        Applies pink and white noise augmentation to the input mel-spectrogram.

        Args:
            inputs: Input mel-spectrogram tensor.

        Returns:
            Augmented mel-spectrogram tensor.
        """
        if not training:
            return inputs

        # Generate pink noise
        noise = self.get_noise(tf.shape(inputs), p=self.p_pink, dtype=inputs.dtype)
        pink_noise = noise * self.mel_pink_noise_psd[None]

        # Generate white noise
        white_noise = self.get_noise(tf.shape(inputs), p=self.p_white, dtype=inputs.dtype)

        # Add noise to the mel-spectrogram
        augmented_mel = inputs + self.pink_noise_factor * pink_noise + self.white_noise_factor * white_noise

        return augmented_mel
