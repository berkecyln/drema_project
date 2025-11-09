import torch
import tqdm

from drema.scene.gaussian_model import GaussianModel
from drema.gaussian_splatting_utils.general_utils import build_rotation
from drema.gaussian_splatting_utils.sh_utils import SH2RGB
from drema.utils.utils import build_quaternion


class InteractiveGaussianModel(GaussianModel):
    
    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)


    def filter_by_mask(self, mask):
        self._xyz = self._xyz[mask]
        self._features_dc = self._features_dc[mask]
        self._features_rest = self._features_rest[mask]
        self._scaling = self._scaling[mask]
        self._rotation = self._rotation[mask]
        self._opacity = self._opacity[mask]


    def filter_by_color(self, threshold=0.01):
        
        rgb = SH2RGB(self._features_dc.squeeze(1))
        color_norm = torch.norm(rgb, dim=1)

        # filter based on the distance between the color and the origin (black)
        mask = color_norm > threshold

        self._xyz = self._xyz[mask]
        self._features_dc = self._features_dc[mask]
        self._features_rest = self._features_rest[mask]
        self._scaling = self._scaling[mask]
        self._rotation = self._rotation[mask]
        self._opacity = self._opacity[mask]

        return mask

    def get_color(self):
        return SH2RGB(self._features_dc.squeeze(1))

    def add(self, gaussian_model):
        if len(self._xyz) == 0:
            self._xyz = gaussian_model._xyz
            self._features_dc = gaussian_model._features_dc
            self._features_rest = gaussian_model._features_rest
            self._scaling = gaussian_model._scaling
            self._rotation = gaussian_model._rotation
            self._opacity = gaussian_model._opacity
        else:
            self._xyz = torch.cat((self._xyz, gaussian_model._xyz))
            self._features_dc = torch.cat((self._features_dc, gaussian_model._features_dc))
            self._features_rest = torch.cat((self._features_rest, gaussian_model._features_rest))
            self._scaling = torch.cat((self._scaling, gaussian_model._scaling))
            self._rotation = torch.cat((self._rotation, gaussian_model._rotation))
            self._opacity = torch.cat((self._opacity, gaussian_model._opacity))

    def translate(self, t, mask=None):
        if mask is None:
            self._xyz += t
        else:
            self._xyz[mask] += t

    def scale(self, s, mask=None):
        if mask is None:
            self._xyz *= s
            self._scaling += torch.log(s)
        else:
            mask = mask.bool()
            self._xyz[mask] *= s
            self._scaling[mask] += torch.log(s)

    def rotate_abs(self, r_mat, mask=None):
        if mask is None:
            mask = torch.ones(len(self._xyz), device="cuda", dtype=bool)

        self._xyz[mask] = torch.matmul(r_mat, self._xyz[mask].t()).t()

        # compute covariance
        new_rotation = build_rotation(self._rotation[mask])
        new_rotation = r_mat @ new_rotation
        self._rotation[mask] = build_quaternion(new_rotation)

    def rotate(self, r_mat, mask=None, whole_mask=None, m=None):
        # TODO: Rotation should affects the sh color too

        mask = mask.bool().squeeze()
        if m is None:
            if whole_mask is not None:
                whole_mask = whole_mask.bool().squeeze()
                m = self._xyz[whole_mask].mean(dim=0)
            else:
                m = self._xyz[mask].mean(dim=0)
                
        self._xyz[mask] -= m
        self._xyz[mask] = torch.matmul(r_mat[0], self._xyz[mask].t()).t()
        self._xyz[mask] += m

        # compute covariance
        new_rotation = build_rotation(self._rotation[mask])
        new_rotation = r_mat[0] @ new_rotation
        self._rotation[mask] = build_quaternion(new_rotation)

    def rotate_covariance(self, r_mat, mask=None):
        if mask is None:
            mask = torch.ones(self._xyz.shape[0], device="cuda", dtype=bool)

        # compute covariance
        new_rotation = build_rotation(self._rotation[mask])
        new_rotation = r_mat @ new_rotation
        self._rotation[mask] = build_quaternion(new_rotation)


    def get_close_gaussians(self, Y, K, mask=None):
        #BN: it was 2000 but I reduced since I got CUDA out of memory
        batch_size = 1000  # Process X in smaller batches
        all_topk_indices = []
        all_topk_distances = []
        if mask is None:
            mask = torch.ones(self._xyz.size(0), device="cuda", dtype=bool)

        for i in tqdm.tqdm(range(0, self._xyz[mask].size(0), batch_size)):
            X_batch = self._xyz[mask][i:i + batch_size]

            # Compute distances for the current batch
            distances = torch.cdist(X_batch, Y)

            # Find top K nearest neighbors
            topk_distances, topk_indices = torch.topk(distances, K, largest=False)

            all_topk_distances.append(topk_distances)
            all_topk_indices.append(topk_indices)

        # Concatenate results
        final_topk_distances = torch.cat(all_topk_distances, dim=0)
        final_topk_indices = torch.cat(all_topk_indices, dim=0)

        return final_topk_distances, final_topk_indices

    def set_require_grad(self, flag=False):

        self._xyz.requires_grad = flag
        self._features_dc.requires_grad = flag
        self._features_rest.requires_grad = flag
        self._opacity.requires_grad = flag
        self._scaling.requires_grad = flag
        self._rotation.requires_grad = flag

    def clone(self):
        """
        Clone the current Gaussian model
        :return: a new InteractiveGaussianModel
        """
        cloned = InteractiveGaussianModel(self.max_sh_degree)
        cloned._xyz = self._xyz.clone()
        cloned._features_dc = self._features_dc.clone()
        cloned._features_rest = self._features_rest.clone()
        cloned._scaling = self._scaling.clone()
        cloned._rotation = self._rotation.clone()
        cloned._opacity = self._opacity.clone()

        return cloned