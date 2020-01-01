import pickle
import tensorflow as tf
import numpy as np
import unittest

from post_processing import post
from configs import keypoints_config as kc
from configs import post_config as pc

HEIGHT = 46
WIDTH = 46
y_grid = tf.linspace(0.0, 1.0, HEIGHT)
x_grid = tf.linspace(0.0, 1.0, WIDTH)
yy, xx = tf.meshgrid(y_grid, x_grid, indexing='ij')  # indexing is a must, otherwise, it's just bizarre!
grid = tf.stack((yy, xx), axis=-1)


def keypoints_layer(kpts_layer, spot_size):
    layer_dists = tf.map_fn(keypoint_distance, kpts_layer)
    all_dists = tf.math.reduce_min(layer_dists, axis=0)
    raw = tf.exp((-(all_dists ** 2) / spot_size))
    return raw

def keypoint_distance(kpt):
    if kpt[2] == tf.constant(0.0):
        return tf.ones((HEIGHT, WIDTH), dtype=tf.float32)  # maximum distance in case of empty kpt, not ideal but meh
    else:
        ortho_dist = grid - kpt[0:2]
        return tf.linalg.norm(ortho_dist, axis=-1)


def layer_PAF(joints):
    layer_PAFS = tf.map_fn(single_PAF, joints)
    combined = tf.math.reduce_sum(layer_PAFS, axis=0)  # averages the vectors out to combine the fields in case they intersect
    return combined


PAF_GAUSSIAN_SIGMA_SQ = 0.0015


def single_PAF(joint):
    jpts = tf.reshape(joint[0:4], (2, 2))  # reshape to ((x1,y1),(x2,y2))
    if joint[4] == tf.constant(0.0) or tf.reduce_all(jpts[1] - jpts[0] == 0.0):
        return tf.zeros((HEIGHT, WIDTH, 2), dtype=tf.float32)  # in case of empty joint
    else:
        # this follows the OpenPose paper of generating the PAFs
        vector_full = jpts[1] - jpts[0]  # get the joint vector
        vector_length = tf.linalg.norm(vector_full)  # get joint length
        vector_hat = vector_full / vector_length  # get joint unit vector
        normal_vector = tf.stack((-vector_hat[1], vector_hat[0]))

        vectors_from_begin = grid - jpts[0]  # get grid of vectors from first joint point
        vectors_from_end = grid - jpts[1]  # get grid of vectors from second joint point

        projections = tf.tensordot(vectors_from_begin, vector_hat, 1)  # get projection on the joint unit vector
        n_projections = tf.tensordot(vectors_from_begin, normal_vector, 1)  # get projection on the joint normal unit vector

        dist_from_begin = tf.linalg.norm(vectors_from_begin, axis=-1)  # get distances from the beginning, and end
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


class TestPost(unittest.TestCase):

    def test_find_peaks_single(self):
        pt1 = [0.5, 0.5, 2]
        expected_peaks = [(int(0.5 * HEIGHT) - 1, int(0.5 * WIDTH) - 1)]
        sample = keypoints_layer(np.array([pt1], dtype=np.float32), 0.1)
        sample = sample.numpy()
        peaks = post.find_peaks(sample, 0.5)

        self.assertEqual(peaks, expected_peaks)

    def test_find_peaks_many(self):
        kpts_many = []
        for i in range(1, 4):
            for j in range(1, 4):
                kpts_many.append([i / 4.0, j / 4.0, 2])
        kpts_many = np.array(kpts_many, dtype=np.float32)
        sample = keypoints_layer(kpts_many, 0.001)
        sample = sample.numpy()

        expected_peaks = [(round(y * (HEIGHT - 1)), round(x * (WIDTH - 1))) for y, x, _ in kpts_many]
        peaks = post.find_peaks(sample, 0.5)

        self.assertEqual(peaks, expected_peaks)

    def test_line_integral(self):
        field = np.ones((100, 100), dtype=np.float32) * np.sqrt(2) / 2
        start = (0, 0)
        end = (99, 99)
        li = post.LineVectorIntegral(field, field)
        sums = li.integrate_line(start, end)
        expected_sums = (70.71073150634766, 70.71073150634766)
        self.assertEqual(sums, expected_sums)

    def test_kpt_paf_alignment_straight(self):
        field = np.ones((100, 100), dtype=np.float32) * np.sqrt(2) / 2
        start = (0, 0)
        end = (99, 99)
        alignment = post.kpt_paf_alignment(start, end, field, field)
        expected_alignment = 1.0
        self.assertAlmostEqual(alignment, expected_alignment, places=5)

    def test_kpt_paf_alignment_perpendicular(self):
        field = np.ones((100, 100), dtype=np.float32) * np.sqrt(2) / 2
        start = (99, 0)
        end = (0, 99)
        alignment = post.kpt_paf_alignment(start, end, field, field)
        expected_alignment = 0.0
        self.assertAlmostEqual(alignment, expected_alignment, places=5)

    def test_kpt_paf_alignment_reverse(self):
        field = np.ones((100, 100), dtype=np.float32) * np.sqrt(2) / 2
        start = (99, 99)
        end = (0, 0)
        alignment = post.kpt_paf_alignment(start, end, field, field)
        expected_alignment = -1.0
        self.assertAlmostEqual(alignment, expected_alignment, places=5)


class TestSkeletonizer(unittest.TestCase):
    def setUp(self) -> None:
        post.Skeletonizer.config(kc.KEYPOINTS_DEF, kc.JOINTS_DEF, pc)
        post.Skeleton.config(kc.KEYPOINTS_DEF, kc.JOINTS_DEF)

    def test_create_skeletons(self):
        d = pickle.load(open("test_case1", "rb"))
        self.kpts = d["kpts"]
        self.pafs = d["pafs"]
        self.sk = post.Skeletonizer(self.kpts, self.pafs)
        self.sk.create_skeletons()

    def test__create_joints1(self):
        d = pickle.load(open("test_case1", "rb"))
        self.kpts = d["kpts"]
        self.pafs = d["pafs"]
        self.sk = post.Skeletonizer(self.kpts, self.pafs)
        potential_kpts = self.sk._localize_potential_kpts()
        self.sk._create_joints(potential_kpts)

    def test__create_joints2(self):
        d = pickle.load(open("test_case2", "rb"))
        self.kpts = d["kpts"]
        self.pafs = d["pafs"]
        self.sk = post.Skeletonizer(self.kpts, self.pafs)
        potential_kpts = self.sk._localize_potential_kpts()
        self.sk._create_joints(potential_kpts)


if __name__ == '__main__':
    unittest.main()
