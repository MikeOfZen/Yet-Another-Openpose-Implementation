import argparse
import cv2
import visualizations as vis
from applications.model_wrapper import ModelWrapper

import configs.draw_config as draw_config

model_path = "../trained_models/model11_test-15Sun1219-2101"


class VideoApp:
    def __init__(self, input_video_file, output_filename, fourcc_str, fps):
        assert len(fourcc_str) == 4

        self.model_wrapper = ModelWrapper(model_path)

        self.video_reader = cv2.VideoCapture(input_video_file)
        if not self.video_reader.isOpened():
            raise IOError("Error opening video file")
        height, width = self.get_video_size()
        self.fps = fps
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self.video_writer = cv2.VideoWriter(output_filename, cv2.CAP_FFMPEG, fourcc, fps, (width, height))

    def get_video_size(self):
        ret, img_bgr = self.video_reader.read()
        return img_bgr.shape[0], img_bgr.shape[1]

    def process_frame(self, img):
        skeletons = self.model_wrapper.process_image(img)

        skeleton_drawer = vis.SkeletonDrawer(img, draw_config)
        for skeleton in skeletons:
            skeleton.draw_skeleton(skeleton_drawer.joint_draw, skeleton_drawer.kpt_draw)
        return img

    def run(self, skip):
        print("Processing video")
        print("Press ESC to exit\n")
        cv2.namedWindow("video-process", cv2.WINDOW_AUTOSIZE)
        i = 0
        while True:
            ret, img_bgr = self.video_reader.read()
            if skip and skip > 0:
                skip -= 1
                continue
            if not ret:
                break
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            processed_img_rgb = self.process_frame(img_rgb)
            processed_img_bgr = cv2.cvtColor(processed_img_rgb, cv2.COLOR_RGB2BGR)

            self.video_writer.write(processed_img_bgr)
            cv2.imshow("video-process", processed_img_bgr)

            key = cv2.waitKey(1)
            if key == 27:  # Esc key to stop
                break
            print(".", end="", flush=True)
            if not i % self.fps:
                print("-", i / self.fps)

        cv2.destroyWindow("video-process")
        self.video_reader.release()
        self.video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video with "Yet another Openpose implementation".')
    parser.add_argument('input', type=str, help='The video to process')
    parser.add_argument('output', type=str, help='The output filename')
    parser.add_argument('--fourcc', type=str, required=True, help='optional fourcc codec code (must be installed and usable by OpenCV)')
    parser.add_argument('--fps', type=float, required=False, default=30, help='optional input video fps')
    parser.add_argument('--skip', type=int, required=False, help='optional number of frames to skip')
    args = parser.parse_args()

    app = VideoApp(args.input, args.output, args.fourcc, args.fps)
    app.run(args.skip)
