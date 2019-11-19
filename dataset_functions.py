from config import *
import tensorflow as tf

class TFrecordParser():
    def __init__(self):
        self.feature_description = {
            'id': tf.io.FixedLenFeature([1], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'size': tf.io.FixedLenFeature([2], tf.int64),
            'kpts': tf.io.FixedLenFeature([], tf.string),
            'joints': tf.io.FixedLenFeature([], tf.string)
        }
    @tf.function
    def read_tfrecord(self,serialized_example, decode_jpg=True):
        parsed = tf.io.parse_single_example(serialized_example, self.feature_description)

        idd = parsed['id']
        image_raw = parsed['image_raw']

        if decode_jpg:
            image_raw = tf.image.decode_jpeg(image_raw)

        size = parsed['size']
        kpts = tf.io.parse_tensor(parsed['kpts'], tf.float32)
        joints = tf.io.parse_tensor(parsed['joints'], tf.float32)

        kpts = tf.RaggedTensor.from_tensor(kpts)
        joints = tf.RaggedTensor.from_tensor(joints)

        return {"id": idd, "image_raw": image_raw, "size": size, "kpts": kpts, "joints": joints}

class LabelTransformer():
    def __init__(self):
        x_grid=tf.linspace(0.0,1.0,IMAGE_WIDTH)
        y_grid=tf.linspace(0.0,1.0,IMAGE_HEIGHT)
        xx,yy=tf.meshgrid(x_grid,y_grid)
        self.grid=tf.stack((xx,yy),axis=-1)

    @tf.function
    def keypoints_spots_vloop(self, kpts_tensor):
        """This transforms the keypoint coords coming from the dataset into gaussian spots label tensor
        this version of the function works via a nested loop vs the other version which uses map_fn.
        *does not support batched input
        :param kpts_tensor - must be a tf.RaggedTensor of shape (num of keypoints(17 for coco),number of persons,3)
        :return tf.Tensor of shape (num of keypoints(17 for coco),IMAGE_HEIGHT,IMAGE_WIDTH) where each point """
        kpts_tensor=kpts_tensor.to_tensor()
        results = tf.TensorArray(tf.float32, size=kpts_tensor.shape[0])
        for i in tf.range(kpts_tensor.shape[0]):
            kpts_layer = kpts_tensor[i]
            total_dist=tf.ones(IMAGE_SIZE,dtype=tf.float32)

            for kpt in kpts_layer:
                if kpt[2]==tf.constant(0.0):
                    continue
                #must add condition to deal with zeros
                ortho_dist=self.grid-kpt[0:2]
                spot_dist=tf.linalg.norm(ortho_dist,axis=-1)
                total_dist=tf.math.minimum(spot_dist,total_dist)

            results=results.write(i, total_dist)
        raw=tf.exp((-(results.stack()**2)/GAUSSIAN_SPOT_SIGMA_SQ))
        return tf.where(raw < 0.001, 0.0, raw)


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

        raw = tf.exp((-(all_dists ** 2) / GAUSSIAN_SPOT_SIGMA_SQ))
        return tf.where(raw < 0.001, 0.0, raw)


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
            return tf.ones(IMAGE_SIZE, dtype=tf.float32)  # maximum distance incase of empty kpt, not ideal but meh
        else:
            ortho_dist = self.grid - kpt[0:2]
            return tf.linalg.norm(ortho_dist, axis=-1)


    @tf.function
    def joints_PAFs(self, joints_tensor):
        """This transforms the joints coords coming from the dataset into vector fields label tensor
        *does not support batched input
        :param joints_tensor - must be a tf.RaggedTensor of shape (num of joints(19 for coco),number of persons,3)
        :return tf.Tensor of shape (num of joints(19 for coco),IMAGE_HEIGHT,IMAGE_WIDTH,2)"""
        joints_tensor = joints_tensor.to_tensor()  # seems to be mandatory for map_fn
        all_pafs = tf.map_fn(self.PAF_layer,
                             joints_tensor)  # ,parallel_iterations=20) for cpu it has no difference, maybe for gpu it will
        # this must be executed in the packing order, to produce the layers in the right order

        return tf.stack(all_pafs)


    @tf.function
    def PAF_layer(self,joints):
        """ Makes a combined PAF for all joints of the same type
        and reduces them to a single array by averaging the vectors out
        *does not support batched input
        :param joints must be a tf.Tensor of shape (n,5)"""
        layer_PAFS = tf.map_fn(self.single_joint, joints)
        return tf.math.reduce_mean(layer_PAFS,
                                   axis=0)  # averages the vectors out to combine the fields in case they intersect


    @tf.function
    def single_joint(self,joint):
        """ Makes a single vector valued PAF (part affinity field) array
        *does not support batched input
        """

        jpts = tf.reshape(joint[0:4], (2, 2))  # reshape to ((x1,y1),(x2,y2))
        if joint[4] == tf.constant(0.0):
            return tf.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 2), dtype=tf.float32)  # in case of empty joint
        else:
            # this follows the OpenPose paper ofr generating the PAFs
            vector_full = jpts[1] - jpts[0]  # get the joint vector
            vector_length = tf.linalg.norm(vector_full)  # get joint length
            vector_hat = vector_full / vector_length  # get joint unit vector

            grid_vectors = self.grid - jpts[0]  # get grid of vectors from first joint point
            projections = tf.tensordot(grid_vectors, vector_hat, 1)  # get projection on the joint unit vector

            normal_vector = tf.stack((-vector_hat[1], vector_hat[0]))
            n_projections = tf.tensordot(grid_vectors, normal_vector, 1)  # get projection on the joint normal unit vector
            na_projections = tf.abs(n_projections)  # absolute value to get both sides of rhe joint

            limit = (0 <= projections) & (projections <= vector_length) & (na_projections <= JOINT_WIDTH)

            limit_brdcst = tf.stack((limit, limit), axis=-1)  # this is for broadcasting to the 2 tuple

            return tf.where(limit_brdcst, vector_hat, tf.constant((0.0, 0.0)))
