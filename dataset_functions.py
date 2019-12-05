from config import *
import tensorflow as tf

class TFrecordParser():
    def __init__(self):
        self.feature_description = {
            'id': tf.io.FixedLenFeature([1], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'size': tf.io.FixedLenFeature([2], tf.int64),
            'kpts': tf.io.FixedLenFeature([], tf.string),
            'joints': tf.io.FixedLenFeature([], tf.string),
            'mask': tf.io.FixedLenFeature([], tf.string)

        }
    @tf.function
    def read_tfrecord(self,serialized_example):
        """Transforms a single data element from the raw TFrecord storage format to a dict of tensors"""
        parsed = tf.io.parse_single_example(serialized_example, self.feature_description)

        idd = parsed['id']
        image_raw = parsed['image_raw']
        size = parsed['size']

        kpts = tf.io.parse_tensor(parsed['kpts'], tf.float32)
        joints = tf.io.parse_tensor(parsed['joints'], tf.float32)
        mask = tf.io.parse_tensor(parsed['mask'], tf.float32)

        kpts = tf.RaggedTensor.from_tensor(kpts)
        joints = tf.RaggedTensor.from_tensor(joints)

        return {"id": idd, "image_raw": image_raw, "size": size, "kpts": kpts, "joints": joints,"mask":mask}

class LabelTransformer():
    def __init__(self):
        y_grid=tf.linspace(0.0,1.0,LABEL_HEIGHT)
        x_grid=tf.linspace(0.0,1.0,LABEL_WIDTH)
        yy,xx=tf.meshgrid(y_grid,x_grid,indexing='ij') #indexing is a must, otherwise, it's just bizzare!
        self.grid=tf.stack((yy,xx),axis=-1)

    @tf.function
    def keypoints_spots_vloop(self, kpts_tensor):
        """This transforms the keypoint coords coming from the dataset into gaussian spots label tensor
        this version of the function works via a nested loop vs the other version which uses map_fn.
        *does not support batched input
        :param kpts_tensor - must be a tf.RaggedTensor of shape (num of keypoints(17 for coco),number of persons,3)
        :return tf.Tensor of shape (IMAGE_HEIGHT,IMAGE_WIDTH,num of keypoints(17 for coco)) where each point """
        kpts_tensor=kpts_tensor.to_tensor()
        results = tf.TensorArray(tf.float32, size=kpts_tensor.shape[0])
        for i in tf.range(kpts_tensor.shape[0]):
            kpts_layer = kpts_tensor[i]
            total_dist=tf.ones((LABEL_HEIGHT, LABEL_WIDTH),dtype=tf.float32)

            for kpt in kpts_layer:
                if kpt[2]==tf.constant(0.0):
                    continue
                #must add condition to deal with zeros
                ortho_dist=self.grid-kpt[0:2]
                spot_dist=tf.linalg.norm(ortho_dist,axis=-1)
                total_dist=tf.math.minimum(spot_dist,total_dist)

            results=results.write(i, total_dist)
        raw=tf.exp((-(results.stack()**2) / KPT_HEATMAP_GAUSSIAN_SIGMA_SQ))
        result=tf.where(raw < 0.001, 0.0, raw)

        result=tf.transpose(result,(1,2,0)) #must transpose to match the moel output
        result=tf.ensure_shape(result,([LABEL_HEIGHT, LABEL_WIDTH,HEATMAP_NUM_FILTERS]))
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

        raw = tf.exp((-(all_dists ** 2) / KPT_HEATMAP_GAUSSIAN_SIGMA_SQ))
        result=tf.where(raw < 0.001, 0.0, raw)

        result=tf.transpose(result,(1,2,0)) #must transpose to match the moel output
        result=tf.ensure_shape(result,([LABEL_HEIGHT, LABEL_WIDTH,HEATMAP_NUM_FILTERS]),name="kpts_enusured_shape")
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
            return tf.ones((LABEL_HEIGHT, LABEL_WIDTH), dtype=tf.float32)  # maximum distance incase of empty kpt, not ideal but meh
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
        result=tf.stack(all_pafs)

        #must transpose to fit the label (NJOINTS,LABEL_HEIGHT, LABEL_WIDTH, 2) to
        # [LABEL_HEIGHT, LABEL_WIDTH,PAF_NUM_FILTERS=NJOINTS*2]
        result=tf.transpose(result, [1, 2, 0, 3])
        result_y = result[..., 0]
        result_x = result[..., 1]
        result = tf.concat((result_y, result_x), axis=-1)

        result=tf.ensure_shape(result,([LABEL_HEIGHT, LABEL_WIDTH, PAF_NUM_FILTERS]),name="paf_enusured_shape")
        return result


    @tf.function
    def layer_PAF(self, joints):
        """ Makes a combined PAF for all joints of the same type
        and reduces them to a single array by averaging the vectors out
        *does not support batched input
        :param joints must be a tf.Tensor of shape (n,5)
        :return a tensor of shape (LABEL_HEIGHT, LABEL_WIDTH, 2)"""
        layer_PAFS = tf.map_fn(self.single_PAF, joints)
        combined=tf.math.reduce_sum(layer_PAFS,axis=0)  # averages the vectors out to combine the fields in case they intersect
        return combined

    @tf.function
    def single_PAF(self, joint):
        """ Makes a single vector valued PAF (part affinity field) array
        *does not support batched input
        :return a tensor of shape (LABEL_HEIGHT, LABEL_WIDTH, 2)
        """
        jpts = tf.reshape(joint[0:4], (2, 2))  # reshape to ((x1,y1),(x2,y2))
        if joint[4] == tf.constant(0.0):
            return tf.zeros((LABEL_HEIGHT, LABEL_WIDTH, 2), dtype=tf.float32)  # in case of empty joint
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

            dist_from_begin = tf.linalg.norm(vectors_from_begin, axis=-1)  # get distances from the begining, and end
            dist_from_end = tf.linalg.norm(vectors_from_end, axis=-1)

            begin_gaussian_mag = tf.exp((-(dist_from_begin ** 2) / PAF_GAUSSIAN_SIGMA_SQ))  # compute gaussian bells
            end_gaussian_mag = tf.exp((-(dist_from_end ** 2) / PAF_GAUSSIAN_SIGMA_SQ))
            normal_gaussian_mag = tf.exp((-(n_projections ** 2) / PAF_GAUSSIAN_SIGMA_SQ))

            limit = (0 <= projections) & (projections <= vector_length)  # cutoff the joint before beginning and after end
            limit = tf.cast(limit, tf.float32)
            bounded_normal_gaussian_mag = normal_gaussian_mag * limit  # bound the normal distance by the endpoints

            max_magnitude = tf.math.reduce_max((begin_gaussian_mag, end_gaussian_mag, bounded_normal_gaussian_mag), axis=0)

            vector_mag = tf.stack((max_magnitude, max_magnitude), axis=-1)

            result = vector_mag * vector_hat  # broadcast joint direction vector to magnitude field
            return result


_label_transformer = LabelTransformer()
@tf.function
def make_label_tensors(elem):
    """Transforms a dict data element:
    1.Read jpg to tensor
    1.1 Resize img to correct size for network
    2.Convert keypoints to correct form label tensor
    3.Convert joints to correct form label tensor
    4.expands mask dim and ensures mask's shape
    outputs a tuple data element"""

    idd = elem['id']
    kpt_tr = _label_transformer.keypoints_spots_vmapfn(elem['kpts'])
    paf_tr = _label_transformer.joints_PAFs(elem['joints'])

    image_raw = elem["image_raw"]
    image = tf.image.decode_jpeg(image_raw, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)

    new_elem={}
    #new_elem.update(elem) #if need to pass something through

    if INCLUDE_MASK:
        mask=elem['mask']
        mask = tf.ensure_shape(mask, ([LABEL_HEIGHT, LABEL_WIDTH]))
        mask=tf.expand_dims(mask,axis=-1) #required to concat

        kpt_tr=tf.concat([mask, kpt_tr], axis=-1) #add mask as zero channel to inputs
        paf_tr=tf.concat([mask, paf_tr], axis=-1)

    new_elem["id"]= idd
    new_elem["paf"] = paf_tr
    new_elem["kpts"] = kpt_tr
    new_elem["image"] = image

    return new_elem

@tf.function
def place_training_labels(elem):
    """Distributes labels into the correct configuration for the model, ie 4 PAF stage, 2 kpt stages
    must match the model"""
    paf_tr=elem['paf']
    kpt_tr=elem['kpts']
    image=elem['image']
    
    if INCLUDE_MASK:
        inputs=(elem['mask'],image)
    else: 
        inputs=image

    return inputs,(paf_tr,paf_tr,paf_tr,paf_tr,kpt_tr,kpt_tr) #this should match the model outputs, and is different for each model