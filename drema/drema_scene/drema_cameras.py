from drema.scene.cameras import Camera


class DepthCamera(Camera):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid, data_device, depth, cx=None, cy=None):

        super().__init__(colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid, data_device=data_device, cx=cx, cy=cy)
        self.depth = depth
