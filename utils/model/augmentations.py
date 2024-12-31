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

    def __init__(self, sr, n_mels, f_min, f_max, pink_noise_factor=0.05, white_noise_factor=0.02, name='spect_noise'):
        super().__init__(name=name)
        import librosa
        self.sr = sr
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.pink_noise_factor = pink_noise_factor
        self.white_noise_factor = white_noise_factor

        # Calculate and store mel frequencies once
        mel_bins = tf.constant(librosa.mel_frequencies(n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max),
                                    dtype=tf.float32)

        # Calculate and store pink noise PSD once
        nyquist_rate = self.sr / 2
        num_bins = int(nyquist_rate / 1000)  # Adjust based on desired frequency resolution
        freqs = np.linspace(0.0, nyquist_rate, num_bins)
        freqs = np.maximum(freqs, 1e-6)  # Avoid division by zero
        pink_noise_psd = 1 / np.sqrt(freqs)
        self.mel_pink_noise_psd_numpy = np.interp(mel_bins, freqs, pink_noise_psd)

        # Interpolate pink noise PSD to mel-frequencies
        self.mel_pink_noise_psd = None

    def build(self, input_shape):
        current_mel_shape_len = len(self.mel_pink_noise_psd_numpy.shape)
        assert current_mel_shape_len <= len(input_shape)
        while current_mel_shape_len < len(input_shape):
            self.mel_pink_noise_psd_numpy = self.mel_pink_noise_psd_numpy[..., None]
            current_mel_shape_len = len(self.mel_pink_noise_psd_numpy.shape)
        self.mel_pink_noise_psd = self.add_weight(
            shape=self.mel_pink_noise_psd_numpy.shape,
            initializer=lambda shape, dtype=None: tf.constant(self.mel_pink_noise_psd) if dtype is None else tf.cast(tf.constant(self.mel_pink_noise_psd, dtype=dtype)),
            trainable=False,
        )

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
        pink_noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=1.0, dtype=inputs.dtype)
        pink_noise *= self.mel_pink_noise_psd[None]

        # Generate white noise
        white_noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=1.0, dtype=inputs.dtype)

        # Add noise to the mel-spectrogram
        augmented_mel = inputs + self.pink_noise_factor * pink_noise + self.white_noise_factor * white_noise

        return augmented_mel
