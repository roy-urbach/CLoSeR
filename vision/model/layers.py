from utils.model.layers import SplitPathways
import tensorflow as tf


class SplitPathwaysVision(SplitPathways):
    def __init__(self, num_patches, token_per_path=False, n=2, d=0.5, intersection=True, fixed=False,
                 seed=0, class_token=True, pathway_to_cls=None, gaussian_mask=False, gaussian_std=4, image_size=32, **kwargs):
        super(SplitPathwaysVision, self).__init__(num_signals=num_patches, token_per_path=token_per_path, n=n, d=d,
                                                  intersection=intersection, fixed=fixed, seed=seed,
                                                  class_token=class_token, pathway_to_cls=pathway_to_cls, **kwargs)
        self.num_patches = num_patches
        self.num_patches_per_path = self.num_signals_per_path
        self.gaussian_mask = gaussian_mask
        self.gaussian_std = gaussian_std
        self.image_size = image_size

    @staticmethod
    def sample_gaussian_mask(image_size, samples, center=None, std=4, seed=None, flat=True):
        import numpy as np
        if seed is not None:
            np.random.seed(seed)

        if center is None:
            center = np.random.choice(image_size, 2)
        if isinstance(center, np.ndarray):
            center = center.tolist()

        i0, j0 = center

        # Create coordinate grid
        x = np.arange(image_size)
        y = np.arange(image_size)
        xx, yy = np.meshgrid(x, y)

        # Compute 2D Gaussian PDF over the grid
        gaussian = np.exp(-((xx - i0) ** 2 + (yy - j0) ** 2) / (2 * std ** 2))
        gaussian /= gaussian.sum()  # Normalize to make it a probability distribution

        # Flatten
        probs = gaussian.ravel()
        indices = np.arange(image_size * image_size)

        # Sample without replacement
        sampled_flat = np.random.choice(indices, size=samples, replace=False, p=probs)
        if flat:
            return sampled_flat

        else:
            # Convert back to (row, col) pairs
            sampled_coords = np.stack([(sampled_flat / image_size).astype(sampled_flat.dtype),
                                       np.mod(sampled_flat, image_size)], axis=-1)
            return sampled_coords

    def get_indices(self, indices=None):
        if self.indices is not None and indices is None and self.gaussian_mask:
            indices = tf.stack([self.sample_gaussian_mask(self.image_size, self.num_signals_per_path,
                                                          std=self.gaussian_std) + self.shift
                                for _ in range(self.n)], axis=-1)
            if self.fixed:
                self.indices = indices

        return super().get_indices(indices=indices)
