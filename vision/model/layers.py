from utils.model.layers import SplitPathways
import tensorflow as tf


class SplitPathwaysVision(SplitPathways):
    def __init__(self, num_patches, token_per_path=False, n=2, d=0.5, intersection=True, fixed=False,
                 seed=0, class_token=True, pathway_to_cls=None, contiguous=False, rows=None, cols=None,
                 gaussian_mask=False, gaussian_std=4, image_size=32, **kwargs):
        self.contiguous = contiguous
        self.rows = rows
        self.cols = cols
        if contiguous:
            assert rows and cols, "if contiguous, must also recieve rows and cols"
        self.gaussian_mask = gaussian_mask
        self.gaussian_std = gaussian_std
        self.image_size = image_size
        super(SplitPathwaysVision, self).__init__(num_signals=num_patches, token_per_path=token_per_path, n=n, d=d,
                                                  intersection=intersection, fixed=fixed, seed=seed,
                                                  class_token=class_token, pathway_to_cls=pathway_to_cls, **kwargs)
        self.num_patches = num_patches
        self.num_patches_per_path = self.num_signals_per_path

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

    @staticmethod
    def sample_contiguous_mask(patches, rows=8, cols=8, center=None, seed=None):
        import numpy as np
        if center is None:
            center_row = np.random.choice(rows)
            center_col = np.random.choice(cols)
            center = np.array(center_row, center_col)
        locs = np.indices([rows+2, cols+2])
        dist_from_center = np.linalg.norm(center[..., None, None] - locs, axis=0)
        if seed:
            np.random.seed(seed)
        dist_from_center_noisy = dist_from_center + np.random.rand(rows + 2, cols + 2) * 0.5
        from scipy.signal import convolve2d
        smoothed_dist_from_center = convolve2d(dist_from_center_noisy, np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 12, mode='valid')
        assert smoothed_dist_from_center.shape == (rows, cols)

        return np.argsort(smoothed_dist_from_center.flatten())[:patches]

    def get_indices(self, indices=None):
        if self.indices is None:
            if self.contiguous:
                import numpy as np
                centers = np.indices([self.rows, self.cols]).reshape(2, -1)[np.random.choice(np.arange(self.rows*self.cols))]
                indices = tf.stack([self.sample_contiguous_mask(self.num_signals_per_path, self.rows, self.cols, center=centers[:, i]) + self.shift
                                    for i in range(self.n)], axis=-1)
                if self.fixed:
                    self.indices = indices

            if self.indices is not None and indices is None and self.gaussian_mask:
                indices = tf.stack([self.sample_gaussian_mask(self.image_size, self.num_signals_per_path,
                                                              std=self.gaussian_std) + self.shift
                                    for _ in range(self.n)], axis=-1)
                if self.fixed:
                    self.indices = indices

        return super().get_indices(indices=indices)
