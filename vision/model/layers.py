from vision.utils.tf_utils import serialize



@serialize
class SplitPathwaysVision(tf_layers.Layer):
    def __init__(self, num_patches, token_per_path=False, n=2, d=0.5, intersection=True, fixed=False,
                 seed=0, class_token=True, pathway_to_cls=None, **kwargs):
        super(SplitPathwaysVision, self).__init__(**kwargs)
        assert intersection or (n*int(num_patches * d)) <= num_patches
        self.n = n
        self.seed = seed
        self.fixed = fixed
        self.num_patches = num_patches
        self.token_per_path = token_per_path
        if pathway_to_cls is not None:
            if isinstance(pathway_to_cls, str):
                pathway_to_cls = eval(pathway_to_cls)
            self.pathway_to_cls = tf.constant(pathway_to_cls)
        if class_token and pathway_to_cls is None:
            if not token_per_path:
                self.pathway_to_cls = tf.zeros(n, dtype=tf.int32)
            else:
                self.pathway_to_cls = tf.range(n, dtype=tf.int32)
        self.num_patches_per_path = int(num_patches * d)
        self.intersection = intersection
        self.class_token = class_token
        self.shift = (tf.reduce_max(self.pathway_to_cls) + 1) if class_token else 0
        self.indices = None
        if fixed:
            set_seed(self.seed)
            self.get_indices()

    def get_config(self):
        return dict(**super().get_config(), n=self.n, seed=self.seed, fixed=self.fixed,
                    num_patches=self.num_patches, intersection=self.intersection,
                    shift=self.shift, class_token=self.class_token, pathway_to_cls=self.pathway_to_cls)

    def get_indices(self):
        if self.indices is None:
            if self.intersection:
                indices = tf.stack(
                    [tf.random.shuffle(tf.range(self.shift, self.num_patches + self.shift))[:self.num_patches_per_path]
                     for _ in range(self.n)],
                    axis=-1)
            else:
                indices = tf.reshape(
                    tf.random.shuffle(tf.range(self.shift,
                                               self.num_patches + self.shift))[:self.num_patches_per_path * self.n],
                    (-1, self.n))

            if self.fixed:
                self.indices = indices
        else:
            indices = self.indices

        # everyone gets the class token
        if self.class_token:
            # cls_tokens_to_add = tf.range(self.n, dtype=indices.dtype)[None] if self.token_per_path else tf.zeros(
            #     (1, self.n),
            #     dtype=indices.dtype)
            indices = tf.concat([self.pathway_to_cls[None], indices], axis=0)
        return indices

    def call(self, inputs, training=False):
        if not training:
            set_seed(self.seed)

        indices = self.get_indices()

        return tf.gather(inputs, indices, axis=-2, batch_dims=0)