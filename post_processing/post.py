import sys

sys.path.append("..")
import numpy as np
import numba
import cv2


@numba.njit
def mark_islands(truth_islands) -> (np.ndarray, dict):
    """This creates an array marking separate truth islands in a truth array
    :param truth_islands:2D array bool array
    :return (islands array-of marked islands,island_hierarchy-islands connections dict)"""
    assert len(truth_islands.shape) == 2, "Must be 2D array"
    dim0 = truth_islands.shape[0]
    dim1 = truth_islands.shape[1]
    shape = (dim0 + 1, dim1 + 1)
    islands = np.zeros(shape, dtype=np.uint16)
    island_num = 1
    island_hierarchy = numba.typed.Dict.empty(key_type=numba.types.uint16, value_type=numba.types.uint16)

    for y in range(dim0):
        for x in range(dim1):
            if truth_islands[y, x]:
                iy = y + 1  # to handle array edge case, islands idxs are shifter by one
                ix = x + 1
                above = islands[iy - 1, ix]
                left = islands[iy, ix - 1]
                if not above and not left:
                    islands[iy, ix] = island_num
                    island_hierarchy[island_num] = island_num  # stays None if islands is isolated, or gets changed to master islands later
                    island_num += 1
                elif above and not left:
                    islands[iy, ix] = above
                elif not above and left:
                    islands[iy, ix] = left
                elif above and left:
                    # make above child of left
                    islands[iy, ix] = above
                    if above != left:
                        island_hierarchy[left] = above
    return islands[1:, 1:], island_hierarchy


@numba.njit
def sort_island_hierarchy(island_hierarchy):
    """converts the hierarchical island dict to a map to the top level island"""
    compact = numba.typed.Dict.empty(key_type=numba.types.uint16, value_type=numba.types.uint16)
    for child, parent in island_hierarchy.items():
        last_parent = child
        while parent != last_parent:
            last_parent = parent
            parent = island_hierarchy[parent]
        compact[child] = last_parent
    return compact


@numba.njit
def islands_max(heatmap, islands, island_hierarchy):
    """This returns the maximum value from values for each island from islands"""
    dim0 = islands.shape[0]
    dim1 = islands.shape[1]

    # islands_max=np.zeros(LIMIT_NUM_ISLANDS, dtype=np.float32)-1
    islands_max_dict = numba.typed.Dict.empty(key_type=numba.types.uint16, value_type=numba.types.float32)
    for island_num in set(island_hierarchy.values()):
        islands_max_dict[island_num] = 0  # init here, because that's the only way

    peaks = {}
    for y in range(dim0):
        for x in range(dim1):
            if islands[y, x]:
                island_num = islands[y, x]
                top_island_num = island_hierarchy[island_num]
                if heatmap[y, x] > islands_max_dict[top_island_num]:  #
                    islands_max_dict[top_island_num] = heatmap[y, x]
                    peaks[top_island_num] = (y, x)

    # translate to list view and remove island numbers
    peaks_l = []
    islands_max_l = []
    for k in peaks:
        peaks_l.append(peaks[k])
        islands_max_l.append(islands_max_dict[k])

    return peaks_l, islands_max_l


@numba.njit
def find_peaks(heatmap, threshold):
    """This takes a 2D heatmap, and returns all peaks on discontinuous regions (islands) for which the heatmap is above the threshold"""
    truth_islands = heatmap > threshold  # get which parts are above the threshold
    segmented_islands, island_hierarchy = mark_islands(truth_islands)  # segment and label the discontinuous regions, returns a island hierarchy dict.
    if not len(island_hierarchy):  # in case nothing found
        return None
    sorted_island_hierarchy = sort_island_hierarchy(island_hierarchy)  # flatten the island hierarchy to point to the top island label
    peaks, island_max = islands_max(heatmap, segmented_islands, sorted_island_hierarchy)  # get the maximum peak location (and value) for each island
    return peaks  # ,island_max


spec = [
        ('field_y', numba.float32[:, :]),
        ('field_x', numba.float32[:, :]),
        ('sum_y', numba.float32),
        ('sum_x', numba.float32),
        # ('num_fields', numba.uint16),
        ]


@numba.jitclass(spec)
class LineVectorIntegral:
    def __init__(self, field_y, field_x):
        self.field_y = field_y
        self.field_x = field_x
        self._init_sums()

    def _init_sums(self):
        self.sum_y = 0.0
        self.sum_x = 0.0

    def _integrate(self, y, x):
        self.sum_x += self.field_x[y, x]
        self.sum_y += self.field_y[y, x]

    def _integrate_line_high(self, y0, x0, y1, x1):
        dx = x1 - x0
        dy = y1 - y0
        xi = 1
        if dx < 0:
            xi = -1
            dx = -dx
        D = 2 * dx - dy
        x = x0
        for y in range(y0, y1 + 1):
            self._integrate(y, x)
            if D > 0:
                x = x + xi
                D = D - 2 * dy
            D = D + 2 * dx

    def _integrate_line_low(self, y0, x0, y1, x1):
        dx = x1 - x0
        dy = y1 - y0
        yi = 1
        if dy < 0:
            yi = -1
            dy = -dy
        D = 2 * dy - dx
        y = y0
        for x in range(x0, x1 + 1):
            self._integrate(y, x)
            if D > 0:
                y = y + yi
                D = D - 2 * dx
            D = D + 2 * dy

    def integrate_line(self, start, end):
        """Integrates the y,x fields in a straight line from the start coords to the end coords
        :param start: 2-tuple of uint, line start coords, must be within defined field limits
        :param end: 2-tuple of uint, line end coords, must be within defined field limits
        :return 2-tuple of integral sum of the line on the x and y fields"""
        y0 = start[0]
        x0 = start[1]
        y1 = end[0]
        x1 = end[1]
        self._init_sums()
        if abs(y1 - y0) < abs(x1 - x0):
            if x0 > x1:
                self._integrate_line_low(y1, x1, y0, x0)
            else:
                self._integrate_line_low(y0, x0, y1, x1)
        else:
            if y0 > y1:
                self._integrate_line_high(y1, x1, y0, x0)
            else:
                self._integrate_line_high(y0, x0, y1, x1)
        return self.sum_y, self.sum_x


@numba.njit
def kpt_paf_alignment(start_kpt, end_kpt, paf_y, paf_x):
    """This creates a score for the PAF field alignment between 2 keypoints
    by doing a vector valued line integral on the straight line between the starting point and the ending,
    on the PAF vector field.
    :param start_kpt - 2-tuple for the starting point of a potential joint
    :param end_kpt - 2-tuple for the ending
    :param paf_y - a field for the y component of the paf
    :param paf_x - a field for the x component of the paf"""
    pot_joint_vec = np.array(end_kpt, dtype=np.float32) - np.array(start_kpt, dtype=np.float32)
    pot_joint_vec_l = np.sqrt((pot_joint_vec ** 2).sum())
    if pot_joint_vec_l == 0.0:  # handle zero length
        return 0.5  # assuming that if kpts overlap they are somewhat related
    pot_joint_unit_vec = np.divide(pot_joint_vec, pot_joint_vec_l)

    li = LineVectorIntegral(paf_y, paf_x)
    paf_sum = li.integrate_line(start_kpt, end_kpt)  # get vector sum of the PAF field
    paf_sum_np = np.array(paf_sum, dtype=np.float32)

    paf_samples_count = (np.abs(pot_joint_vec)).max() + 1.0  # calc number of vectors in paf sum
    average_paf = paf_sum_np / paf_samples_count  # average across length of line

    alignment = np.sum(pot_joint_unit_vec * average_paf)  # calc the alignment
    return alignment


class Skeletonizer:
    @classmethod
    def config(cls, KEYPOINTS_DEF, JOINTS_DEF, post_config):
        cls.KEYPOINTS_DEF = KEYPOINTS_DEF
        cls.JOINTS_DEF = JOINTS_DEF
        cls.KEYPOINTS_HEATMAP_THRESHOLD = post_config.KEYPOINTS_HEATMAP_THRESHOLD
        cls.JOINT_ALIGNMENT_THRESHOLD = post_config.JOINT_ALIGNMENT_THRESHOLD

    def __init__(self, kpts, pafs):
        """
        :param pafs: numpy raw pafs output from the trained model
        :param kpts: numpy raw kpts output from the trained model
        """
        self.kpts = kpts
        self.pafs = pafs

        self.LABEL_HEIGHT_RANGE = self.kpts.shape[0] - 1
        self.LABEL_WIDTH_RANGE = self.kpts.shape[1] - 1

    def _localize_potential_kpts(self):
        """This converts the trained model output keypoints heatmaps tensor to coordinates of potential keypoint
        coordinates. find_peaks thresholds the input, segmenting the input into islands of certainty
        and for each island finds the max coords.
        for each keypoint type (from KEYPOINT_DEF) it stores all hits
        :returns a dict of kpts vs their locations"""
        potential_kpts = {}
        for kpt_name, kpt in self.KEYPOINTS_DEF.items():
            kpt_idx = kpt["idx"]
            kpt_heatmap = self.kpts[..., kpt_idx]
            peaks = find_peaks(kpt_heatmap, self.KEYPOINTS_HEATMAP_THRESHOLD)
            potential_kpts[kpt_name] = peaks
        return potential_kpts

    def _create_joints(self, potential_kpts: dict):
        """Creates the best joints from the potential kpts by using the paf vector fields
        all possible joints are scored by alignment with the paf field.
        and then only the best joint matching (bipartite matching) is created according to max alignment score
        and by limiting joints to a 1-to-1 start kpt to end kpt joints
        :param potential_kpts dict of discovered kpts and their coordinates in the processed image
        :returns dict of joint names vs their coords (start,end)"""
        joints = {}
        x_paf_offset = len(self.JOINTS_DEF)
        for joint_name, joint in self.JOINTS_DEF.items():  # work by joints definitions
            start_kpt_name = joint["kpts"][0]
            end_kpt_name = joint["kpts"][1]
            start_kpts = potential_kpts[start_kpt_name]
            end_kpts = potential_kpts[end_kpt_name]
            if not start_kpts or not end_kpts:
                continue
            paf_y = self.pafs[..., joint["idx"]]  # get individual y paf field
            paf_x = self.pafs[..., joint["idx"] + x_paf_offset]
            joint_candidates = self._joint_scoring(start_kpts, end_kpts, paf_y, paf_x)  #
            max_num_joints = min(len(start_kpts), len(end_kpts))
            matched_joints = self._joint_matching(joint_candidates, max_num_joints)
            joints[joint_name] = matched_joints
        return joints

    @staticmethod
    def _joint_scoring(start_kpts: list, end_kpts: list, paf_y: np.ndarray, paf_x: np.ndarray):
        """This scores the alignment between all joint start keypoints and joint end keypoints
        using the paf vector field (split to x,y components)
        :param start_kpts list of 2-tuple of ints, joint start coordinates
        :param end_kpts list of 2-tuple of ints, joint end coordinates
        :param paf_y - a field for the y component of the paf
        :param paf_x - a field for the x component of the paf"""
        joint_candidates = []
        for start_kpt in start_kpts:
            for end_kpt in end_kpts:
                alignment = kpt_paf_alignment(start_kpt, end_kpt, paf_y, paf_x)
                t = (alignment, start_kpt, end_kpt)
                joint_candidates.append(t)
        return joint_candidates

    def _joint_matching(self, joint_candidates: list, num_joints: int):
        """This takes a list of possible joint candidates and makes a bipartite matching of the joints according to
        alignment score, and creating only 1 to 1 connections
        :param joint_candidates list of 3-tuple (alignment,start_kpt,end_kpt), all possible bipartite connections
        :param num_joints max number of possible joints, is min of len(start_kpts),len(end_kpts)
        :returns matched joint list of (start_kpt,end_kpt)"""
        filtered_candidates = filter(lambda jc: jc[0] > self.JOINT_ALIGNMENT_THRESHOLD, joint_candidates)
        sorted_candidates = sorted(filtered_candidates, key=lambda x: x[0], reverse=True)  # sort to find highest alignment joints

        matched_start_kpts = []
        matched_end_kpts = []
        count = 0
        matched_joints = []
        for alignment, start_kpt, end_kpt in sorted_candidates:

            if start_kpt not in matched_start_kpts and end_kpt not in matched_end_kpts:  # only match those joints
                # for which no endpoint is already in another joint
                matched_joints.append((start_kpt, end_kpt))
                matched_start_kpts.append(start_kpt)
                matched_end_kpts.append(end_kpt)
                count += 1
            if count >= num_joints:  # num_joints is the maximum possible number of joints, which is the min of number of
                # start kpts or end kpts
                break
        return matched_joints

    def _build_skeletons(self, joint_lists):
        """Builds the complete skeletons from the disconnected joints, by matching endpoints and coordinates between the joints"""
        skeletons = []
        for joint_name, joints_coords in joint_lists.items():
            start_kpt_name = self.JOINTS_DEF[joint_name]["kpts"][0]
            end_kpt_name = self.JOINTS_DEF[joint_name]["kpts"][1]
            for start_coord, end_coord in joints_coords:
                found_match = False
                for skeleton in skeletons:
                    if skeleton.match_joint(joint_name, start_kpt_name, end_kpt_name, start_coord, end_coord):
                        found_match = True
                        break
                if not found_match:
                    new_sk = Skeleton(joint_name, start_kpt_name, end_kpt_name, start_coord, end_coord)
                    skeletons.append(new_sk)
        return skeletons

    def _normalize_joint_coords(self, joint_lists):
        """for all coordinates in joint_lists, scale the coord to 0..1 range"""
        normalized_joint_lists = {}
        for joint_name, joints_coords in joint_lists.items():
            normalized_joints_coords = []
            for start_coord, end_coord in joints_coords:
                normalized_start_coord = self._normalize_coord(start_coord)
                normalized_end_coord = self._normalize_coord(end_coord)
                normalized_joints_coords.append((normalized_start_coord, normalized_end_coord))
            normalized_joint_lists[joint_name] = normalized_joints_coords
        return normalized_joint_lists

    def _normalize_coord(self, coord):
        normalized_y = coord[0] / self.LABEL_HEIGHT_RANGE
        normalized_x = coord[1] / self.LABEL_WIDTH_RANGE
        return (normalized_y, normalized_x)

    def create_skeletons(self):
        """Creates skeletons from the kpts and pafs tensors
        :return list of Skeleton"""
        potential_kpts = self._localize_potential_kpts()
        joint_lists = self._create_joints(potential_kpts)
        normalized_joint_list = self._normalize_joint_coords(joint_lists)
        skeletons = self._build_skeletons(normalized_joint_list)
        return skeletons


class Skeleton:
    @classmethod
    def config(cls, KEYPOINTS_DEF, JOINTS_DEF):
        cls.KEYPOINTS_DEF = KEYPOINTS_DEF
        cls.JOINTS_DEF = JOINTS_DEF

    def __init__(self, joint_name, start_kpt_name, end_kpt_name, start_coord, end_coord):
        self.joints = {}
        self.keypoints = {}

        # add first joint
        self.joints[joint_name] = (start_coord, end_coord)
        self.keypoints[start_kpt_name] = start_coord
        self.keypoints[end_kpt_name] = end_coord

    def match_joint(self, joint_name, start_kpt_name, end_kpt_name, start_coord, end_coord):
        if start_kpt_name in self.keypoints and self.keypoints[start_kpt_name] == start_coord:
            self.joints[joint_name] = (start_coord, end_coord)
            self.keypoints[start_kpt_name] = start_coord
            self.keypoints[end_kpt_name] = end_coord
            return True
        if end_kpt_name in self.keypoints and self.keypoints[end_kpt_name] == end_coord:
            self.joints[joint_name] = (start_coord, end_coord)
            self.keypoints[start_kpt_name] = start_coord
            self.keypoints[end_kpt_name] = end_coord
            return True
        return False

    def draw_skeleton(self, joint_draw, kpt_draw):
        """Uses the joint_draw and kpt_draw callables to draw the skeleton
         :param joint_draw: callable with parameters (start_coord,end_coord,joint_name)
         start_coord and end_coord are 2-tuple
         joint_name is str of the joint
         :param kpt_draw: callable with parameters (kpt_coord,kpt_name)
         where kpt_coord is a 2-tuple of keypoint coordinates
         and kpt_name is the keypoint name"""
        for joint_name, (start_coord, end_coord) in self.joints.items():
            joint_draw(start_coord, end_coord, joint_name)
        for kpt_name, kpt_coord in self.keypoints.items():
            kpt_draw(kpt_coord, kpt_name)
