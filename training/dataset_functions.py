import tensorflow as tf


class DatasetTransformer:
    def __init__(self, config):
        self.INCLUDE_MASK = config.INCLUDE_MASK
        self.LABEL_HEIGHT = config.LABEL_HEIGHT
        self.LABEL_WIDTH = config.LABEL_WIDTH
        self.IMAGE_SIZE = config.IMAGE_SIZE

        self.PAF_GAUSSIAN_SIGMA_SQ = config.PAF_GAUSSIAN_SIGMA_SQ
        self.KPT_HEATMAP_GAUSSIAN_SIGMA_SQ = config.KPT_HEATMAP_GAUSSIAN_SIGMA_SQ

        self.PAF_NUM_FILTERS = config.PAF_NUM_FILTERS
        self.HEATMAP_NUM_FILTERS = config.HEATMAP_NUM_FILTERS

        self.JOINTS_DEF = config.JOINTS_DEF
        self.JOINTS_SIDES = config.JOINTS_SIDES
        self.KEYPOINTS_SIDES = config.KEYPOINTS_SIDES

        self.CONTRAST_RANGE = config.CONTRAST_RANGE
        self.BRIGHTNESS_RANGE = config.BRIGHTNESS_RANGE
        self.HUE_RANGE = config.HUE_RANGE
        self.SATURATION_RANGE = config.SATURATION_RANGE

        # for parsing TFrecords files
        self.feature_description = {
                'id'       : tf.io.FixedLenFeature([1], tf.int64),
                'image_raw': tf.io.FixedLenFeature([], tf.string),
                'size'     : tf.io.FixedLenFeature([2], tf.int64),
                'kpts'     : tf.io.FixedLenFeature([], tf.string),
                'joints'   : tf.io.FixedLenFeature([], tf.string),
                'mask'     : tf.io.FixedLenFeature([], tf.string)
                }
        self.init_grid()

    def init_grid(self):
        y_grid = tf.linspace(0.0, 1.0, self.LABEL_HEIGHT)
        x_grid = tf.linspace(0.0, 1.0, self.LABEL_WIDTH)
        yy, xx = tf.meshgrid(y_grid, x_grid, indexing='ij')  # indexing is a must, otherwise, it's just bizarre!
        self.grid = tf.stack((yy, xx), axis=-1)

    @tf.function
    def read_tfrecord(self, serialized_example):
        """Transforms a single data element from the raw TFrecord storage format to a dict of tensors"""
        parsed = tf.io.parse_single_example(serialized_example, self.feature_description)

        idd = parsed['id']
        image_raw = parsed['image_raw']
        size = parsed['size']

        kpts = tf.io.parse_tensor(parsed['kpts'], tf.float32)
        joints = tf.io.parse_tensor(parsed['joints'], tf.float32)
        mask = tf.io.parse_tensor(parsed['mask'], tf.float32)
        mask = tf.ensure_shape(mask, ([self.LABEL_HEIGHT, self.LABEL_WIDTH]))
        mask = tf.expand_dims(mask, axis=-1)  # required to concat

        kpts = tf.RaggedTensor.from_tensor(kpts)
        joints = tf.RaggedTensor.from_tensor(joints)

        return {"id": idd, "image_raw": image_raw, "size": size, "kpts": kpts, "joints": joints, "mask": mask}

    @tf.function
    def keypoints_spots_vloop(self, kpts_tensor):
        """This transforms the keypoint coords coming from the dataset into gaussian spots label tensor
        this version of the function works via a nested loop vs the other version which uses map_fn.
        *does not support batched input
        :param kpts_tensor - must be a tf.RaggedTensor of shape (num of keypoints(17 for coco),number of persons,3)
        :return tf.Tensor of shape (IMAGE_HEIGHT,IMAGE_WIDTH,num of keypoints(17 for coco)) where each point """
        kpts_tensor = kpts_tensor.to_tensor()
        results = tf.TensorArray(tf.float32, size=kpts_tensor.shape[0])
        for i in tf.range(kpts_tensor.shape[0]):
            kpts_layer = kpts_tensor[i]
            total_dist = tf.ones((self.LABEL_HEIGHT, self.LABEL_WIDTH), dtype=tf.float32)

            for kpt in kpts_layer:
                if kpt[2] == tf.constant(0.0):
                    continue
                # must add condition to deal with zeros
                ortho_dist = self.grid - kpt[0:2]
                spot_dist = tf.linalg.norm(ortho_dist, axis=-1)
                total_dist = tf.math.minimum(spot_dist, total_dist)

            results = results.write(i, total_dist)
        raw = tf.exp((-(results.stack() ** 2) / self.KPT_HEATMAP_GAUSSIAN_SIGMA_SQ))
        result = tf.where(raw < 0.001, 0.0, raw)

        result = tf.transpose(result, (1, 2, 0))  # must transpose to match the model output
        result = tf.ensure_shape(result, ([self.LABEL_HEIGHT, self.LABEL_WIDTH, self.HEATMAP_NUM_FILTERS]))
        return result

    @tf.function
    def keypoints_spots_vmapfn(self, kpts_tensor):
        """This transforms the keypoint coords coming from the dataset into gaussian spots label tensor
        this version of the function works via a map_fn.
        *does not support batched input
        :param kpts_tensor - must be a tf.RaggedTensor of shape (num of keypoints(17 for coco),n,3) where n is the number of persons
        :return tf.Tensor of shape (num of keypoints(17 for coco),IMAGE_HEIGHT,IMAGE_WIDTH)"""
        kpts_tensor = kpts_tensor.to_tensor()  # seems to be mandatory for map_fn
        all_dists = tf.map_fn(self.keypoints_layer,
                              kpts_tensor)  # ,parallel_iterations=20) for cpu it has no difference, maybe for gpu it will

        raw = tf.exp((-(all_dists ** 2) / self.KPT_HEATMAP_GAUSSIAN_SIGMA_SQ))
        result = tf.where(raw < 0.001, 0.0, raw)

        result = tf.transpose(result, (1, 2, 0))  # must transpose to match the model output
        result = tf.ensure_shape(result, ([self.LABEL_HEIGHT, self.LABEL_WIDTH, self.HEATMAP_NUM_FILTERS]), name="kpts_ensured_shape")
        return result

    @tf.function
    def keypoints_layer(self, kpts_layer):
        """This transforms a single layer of keypoints (such as 3 keypoints of type 'right shoulder')
        the keypoint_distance creates an array of the distances from each keypoint
        and this reduces them to a single array by the  of the distances.
        :param kpts_layer must be a tf.Tensor of shape (n,3)"""
        layer_dists = tf.map_fn(self.keypoint_distance, kpts_layer)
        return tf.math.reduce_min(layer_dists, axis=0)

    @tf.function
    def keypoint_distance(self, kpt):
        """This transforms a single keypoint into an array of the distances from the keypoint
        :param kpt must be tf.Tensor of shape (x,y,a) where a is either 0,1,2 for missing,invisible and visible"""
        if kpt[2] == tf.constant(0.0):
            return tf.ones((self.LABEL_HEIGHT, self.LABEL_WIDTH), dtype=tf.float32)  # maximum distance in case of empty kpt, not ideal but meh
        else:
            ortho_dist = self.grid - kpt[0:2]
            return tf.linalg.norm(ortho_dist, axis=-1)

    @tf.function
    def joints_PAFs(self, joints_tensor):
        """This transforms the joints coords coming from the dataset into vector fields label tensor
        *does not support batched input
        :param joints_tensor - must be a tf.RaggedTensor of shape (num of joints(19 for coco),number of persons,3)
        :return tf.Tensor of shape (IMAGE_HEIGHT,IMAGE_WIDTH,num of joints(19 for coco)*2)"""
        joints_tensor = joints_tensor.to_tensor()  # seems to be mandatory for map_fn
        all_pafs = tf.map_fn(self.layer_PAF,
                             joints_tensor)
        # ,parallel_iterations=20) for cpu it has no difference, maybe for gpu it will
        # this must be executed in the packing order, to produce the layers in the right order
        result = tf.stack(all_pafs)

        result = tf.where(abs(result) < 0.001, 0.0, result)  # stabilize numerically

        # must transpose to fit the label (NJOINTS,LABEL_HEIGHT, LABEL_WIDTH, 2) to
        # [LABEL_HEIGHT, LABEL_WIDTH,PAF_NUM_FILTERS=NJOINTS*2]
        result = tf.transpose(result, [1, 2, 0, 3])
        result_y = result[..., 0]
        result_x = result[..., 1]
        result = tf.concat((result_y, result_x), axis=-1)

        result = tf.ensure_shape(result, ([self.LABEL_HEIGHT, self.LABEL_WIDTH, self.PAF_NUM_FILTERS]), name="paf_ensured_shape")
        return result

    @tf.function
    def layer_PAF(self, joints):
        """ Makes a combined PAF for all joints of the same type
        and reduces them to a single array by averaging the vectors out
        *does not support batched input
        :param joints must be a tf.Tensor of shape (n,5)
        :return a tensor of shape (LABEL_HEIGHT, LABEL_WIDTH, 2)"""
        layer_PAFS = tf.map_fn(self.single_PAF, joints)
        combined = tf.math.reduce_sum(layer_PAFS, axis=0)  # averages the vectors out to combine the fields in case they intersect
        return combined

    @tf.function
    def single_PAF(self, joint):
        """ Makes a single vector valued PAF (part affinity field) array
        *does not support batched input
        :param joint a 1D tensor of (x1,y1,x2,y2,visibility)
        :return a tensor of shape (LABEL_HEIGHT, LABEL_WIDTH, 2)
        """
        jpts = tf.reshape(joint[0:4], (2, 2))  # reshape to ((x1,y1),(x2,y2))
        if joint[4] == tf.constant(0.0) or tf.reduce_all(jpts[1] - jpts[0] == 0.0):
            return tf.zeros((self.LABEL_HEIGHT, self.LABEL_WIDTH, 2), dtype=tf.float32)  # in case of empty joint
        else:
            # this follows the OpenPose paper of generating the PAFs
            vector_full = jpts[1] - jpts[0]  # get the joint vector
            vector_length = tf.linalg.norm(vector_full)  # get joint length
            vector_hat = vector_full / vector_length  # get joint unit vector
            normal_vector = tf.stack((-vector_hat[1], vector_hat[0]))

            vectors_from_begin = self.grid - jpts[0]  # get grid of vectors from first joint point
            vectors_from_end = self.grid - jpts[1]  # get grid of vectors from second joint point

            projections = tf.tensordot(vectors_from_begin, vector_hat, 1)  # get projection on the joint unit vector
            n_projections = tf.tensordot(vectors_from_begin, normal_vector, 1)  # get projection on the joint normal unit vector

            dist_from_begin = tf.linalg.norm(vectors_from_begin, axis=-1)  # get distances from the beginning, and end
            dist_from_end = tf.linalg.norm(vectors_from_end, axis=-1)

            begin_gaussian_mag = tf.exp((-(dist_from_begin ** 2) / self.PAF_GAUSSIAN_SIGMA_SQ))  # compute gaussian bells
            end_gaussian_mag = tf.exp((-(dist_from_end ** 2) / self.PAF_GAUSSIAN_SIGMA_SQ))
            normal_gaussian_mag = tf.exp((-(n_projections ** 2) / self.PAF_GAUSSIAN_SIGMA_SQ))

            limit = (0 <= projections) & (projections <= vector_length)  # cutoff the joint before beginning and after end
            limit = tf.cast(limit, tf.float32)
            bounded_normal_gaussian_mag = normal_gaussian_mag * limit  # bound the normal distance by the endpoints

            max_magnitude = tf.math.reduce_max((begin_gaussian_mag, end_gaussian_mag, bounded_normal_gaussian_mag), axis=0)

            vector_mag = tf.stack((max_magnitude, max_magnitude), axis=-1)

            result = vector_mag * vector_hat  # broadcast joint direction vector to magnitude field
            return result

    @tf.function
    def open_image(self, elem):
        """Transforms a dict data element:
        converts raw jpeg stored as bytes form to array and resizes to config determined size,
         also converts blackwhite images to 3 channels"""
        image_raw = elem["image_raw"]
        image = tf.image.decode_jpeg(image_raw, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize(image, self.IMAGE_SIZE)

        new_elem = {}
        new_elem.update(elem)
        new_elem.pop("image_raw")

        new_elem["image"] = image
        return new_elem

    @tf.function
    def apply_mask(self, elem):
        """Transforms a dict data element:
        applies background persons mask to keypoints and PAF tensors as last channel,
        to be used by the special masked loss"""

        mask = elem['mask']
        pafs = elem["pafs"]
        kpts = elem["kpts"]

        kpts = tf.concat([kpts, mask], axis=-1)  # add mask as zero channel to inputs
        pafs = tf.concat([pafs, mask], axis=-1)

        new_elem = {}
        new_elem.update(elem)
        new_elem["pafs"] = pafs
        new_elem["kpts"] = kpts

        return new_elem

    @tf.function
    def make_label_tensors(self, elem):
        """Transforms a dict data element:
        Convert keypoints to correct form label tensor.
        Convert joints to correct form label tensor.
        outputs a dict element"""
        kpts = self.keypoints_spots_vmapfn(elem['kpts'])
        pafs = self.joints_PAFs(elem['joints'])

        new_elem = {}
        new_elem.update(elem)  # if need to pass something through
        new_elem.pop('joints')

        new_elem["pafs"] = pafs
        new_elem["kpts"] = kpts

        return new_elem

    @tf.function
    def image_only_augmentation(self, elem):
        """Dataset operation, working on dict element,
        randomly changes the color,contrast,brightness of the input images"""
        image = elem["image"]
        # adjust contrast
        image = tf.image.random_contrast(image, lower=1 - self.CONTRAST_RANGE, upper=1 + self.CONTRAST_RANGE)
        image = tf.image.random_brightness(image, max_delta=self.BRIGHTNESS_RANGE)
        image = tf.image.random_hue(image, self.HUE_RANGE)
        image = tf.image.random_saturation(image, 1 - self.SATURATION_RANGE, 1 + self.SATURATION_RANGE)
        image = tf.clip_by_value(image, 0, 1)  # clipping is required as some of these functions seems to go out of bounds [0..1]

        new_elem = {}
        new_elem.update(elem)
        new_elem["image"] = image
        return new_elem

    @tf.function
    def mirror_augmentation(self, elem):
        """Dataset operation, working on dict element,
        with a 0.5 chance, flips the image horizontally"""
        new_elem = {}
        new_elem.update(elem)

        num_joints = len(self.JOINTS_DEF)
        if tf.random.uniform([1]) > 0.5:
            new_elem["image"] = tf.image.flip_left_right(elem["image"])
            img_rotated_kpts = tf.image.flip_left_right(elem["kpts"])
            new_elem["mask"] = tf.image.flip_left_right(elem["mask"])
            img_rotated_pafs = tf.image.flip_left_right(elem["pafs"])

            # must flip the x paf tensor as well
            pafY = img_rotated_pafs[..., :num_joints]
            pafX = -img_rotated_pafs[..., num_joints:]

            # must flip the labels as well
            pafY_center = pafY[..., self.JOINTS_SIDES["C"][0]:self.JOINTS_SIDES["C"][1] + 1]
            pafY_right = pafY[..., self.JOINTS_SIDES["R"][0]:self.JOINTS_SIDES["R"][1] + 1]
            pafY_left = pafY[..., self.JOINTS_SIDES["L"][0]:self.JOINTS_SIDES["L"][1] + 1]

            pafX_center = pafX[..., self.JOINTS_SIDES["C"][0]:self.JOINTS_SIDES["C"][1] + 1]
            pafX_right = pafX[..., self.JOINTS_SIDES["R"][0]:self.JOINTS_SIDES["R"][1] + 1]
            pafX_left = pafX[..., self.JOINTS_SIDES["L"][0]:self.JOINTS_SIDES["L"][1] + 1]

            pafs = tf.concat([
                    pafY_center
                    , pafY_left
                    , pafY_right
                    , pafX_center
                    , pafX_left
                    , pafX_right
                    ], axis=-1)  # this reconstitutes the PAFS tensor with flipped labels, to match the image
            new_elem["pafs"] = pafs

            kpts_center = img_rotated_kpts[..., self.KEYPOINTS_SIDES["C"][0]:self.KEYPOINTS_SIDES["C"][1] + 1]
            kpts_right = img_rotated_kpts[..., self.KEYPOINTS_SIDES["R"][0]:self.KEYPOINTS_SIDES["R"][1] + 1]
            kpts_left = img_rotated_kpts[..., self.KEYPOINTS_SIDES["L"][0]:self.KEYPOINTS_SIDES["L"][1] + 1]
            kpts = tf.concat([
                    kpts_center
                    , kpts_left
                    , kpts_right
                    ], axis=-1)  # this reconstitutes the KPTS tensor with flipped labels, to match the image

            new_elem["kpts"] = kpts
        return new_elem

# @tf.function
# def place_training_labels(elem):
#     """Distributes labels into the correct configuration for the model, ie 4 PAF stage, 2 kpt stages
#     must match the model"""
#     paf_tr = elem['paf']
#     kpt_tr = elem['kpts']
#     image = elem['image']
#
#     if INCLUDE_MASK:
#         inputs = (image,elem['mask'])
#     else:
#         inputs = image
#
#     return inputs, (paf_tr, paf_tr, paf_tr, paf_tr, kpt_tr, kpt_tr)  # this should match the model outputs, and is different for each model
