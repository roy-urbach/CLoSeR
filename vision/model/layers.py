from utils.model.layers import SplitPathways
from vision.utils.tf_utils import serialize


@serialize
class SplitPathwaysVision(SplitPathways):
    def __init__(self, num_patches, token_per_path=False, n=2, d=0.5, intersection=True, fixed=False,
                 seed=0, class_token=True, pathway_to_cls=None, **kwargs):
        super(SplitPathwaysVision, self).__init__(num_features=num_patches, token_per_path=token_per_path, n=n, d=d,
                                                  intersection=intersection, fixed=fixed, seed=seed,
                                                  class_token=class_token, pathway_to_cls=pathway_to_cls, **kwargs)
        self.num_patches = num_patches
        self.num_patches_per_path = self.num_features_per_path
