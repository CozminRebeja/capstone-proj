"""
Object detection on panorama images.

Usage:
    python3 detection.py input.jpg output.jpg
"""
import argparse
import cv2
import numpy as np

from ..paths import ASSETS_DIR
from ..projection.stereo import pano2stereo, realign_bbox

CONFIDENCE_THRESHOLD = 0.45
OBJECTNESS_THRESHOLD = 0.55
NMS_THRESHOLD = 0.25
INPUT_RESOLUTION = (416, 416)
MIN_BOX_AREA_RATIO = 0.001


class Yolo():
    """
    Wrapped YOLO network from OpenCV DNN.
    """

    def __init__(
        self,
        conf_threshold=CONFIDENCE_THRESHOLD,
        objectness_threshold=OBJECTNESS_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        resolution=INPUT_RESOLUTION,
    ):
        model_configuration = str(ASSETS_DIR / 'yolov3.cfg')
        model_weight = str(ASSETS_DIR / 'yolov3.weights')

        self.classes = None
        class_file = str(ASSETS_DIR / 'coco.names')
        with open(class_file, 'rt') as file:
            self.classes = file.read().rstrip('\n').split('\n')

        net = cv2.dnn.readNetFromDarknet(
            model_configuration, model_weight)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.yolo = net

        self.conf_threshold = float(conf_threshold)
        self.objectness_threshold = float(objectness_threshold)
        self.nms_threshold = float(nms_threshold)
        self.resolution = tuple(resolution)
        print('Model Initialization Done!')

    def detect(self, frame):
        '''
        The yolo function which is provided by opencv

        Args:
            frames(np.array): input picture for object detection

        Returns:
            ret(np.array): all possible boxes with dim = (N, classes+5)
        '''
        blob = cv2.dnn.blobFromImage(np.float32(frame), 1/255, self.resolution,
                                     [0, 0, 0], 1, crop=False)

        self.yolo.setInput(blob)
        layers_names = self.yolo.getLayerNames()
        unconnected = self.yolo.getUnconnectedOutLayers()
        # OpenCV<=3 returns Nx1, newer versions return 1-D arrays; flatten to unify.
        unconnected = unconnected.flatten().astype(int).tolist()
        output_layer = [layers_names[i - 1] for i in unconnected]
        outputs = self.yolo.forward(output_layer)

        ret = np.zeros((1, len(self.classes)+5))
        for out in outputs:
            ret = np.concatenate((ret, out), axis=0)
        return ret

    def draw_bbox(self, frame, class_id, conf, left, top, right, bottom):
        '''
        Drew a Bounding Box

        Args:
            frame(np.array): the base image for painting box on
            class_id(int)  : id of the object
            conf(float)    : confidential score for the object
            left(int)      : the left pixel for the box
            top(int)       : the top pixel for the box
            right(int)     : the right pixel for the box
            bottom(int)    : the bottom pixel for the box

        Return:
            frame(np.array): the image with bounding box on it
        '''
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 1)

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if self.classes:
            assert(class_id < len(self.classes))
            label = '%s:%s' % (self.classes[class_id], label)

        #Display the label at the top of the bounding box
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv2.rectangle(frame,
                      (left, top - round(1.5*label_size[1])),
                      (left + round(label_size[0]), top + base_line),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return frame

    def _sanitize_box(self, center_x, center_y, width, height, frame_width, frame_height):
        """Clamp a normalized pano-space box to valid image bounds."""
        # YOLO gives center/size values; OpenCV NMS expects top-left/width/height.
        left = max(0.0, center_x - width / 2.0)
        top = max(0.0, center_y - height / 2.0)
        right = min(float(frame_width), center_x + width / 2.0)
        bottom = min(float(frame_height), center_y + height / 2.0)

        width = right - left
        height = bottom - top
        if width <= 0.0 or height <= 0.0:
            return None

        min_area = frame_width * frame_height * MIN_BOX_AREA_RATIO
        if width * height < min_area:
            return None

        left = int(np.floor(left))
        top = int(np.floor(top))
        right = int(np.ceil(right))
        bottom = int(np.ceil(bottom))

        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            return None

        return [left, top, width, height]

    def nms_selection(self, frame, output):
        '''
        Packing the openCV Non-Maximum Suppression Selection Algorthim

        Args:
            frame(np.array) : the input image for getting the size
            output(np.array): scores from yolo, and transform into confidence and class

        Returns:
            class_ids (list)  : the list of class id for the output from yolo
            confidences (list): the list of confidence for the output from yolo
            boxes (list)      : the list of box coordinate for the output from yolo
            indices (list)    : the list of box after NMS selection

        '''
        print('NMS selecting...')
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class
        # with the highest score.
        class_ids = []
        confidences = []
        boxes = []
        for detection in output:
            objectness = float(detection[4])
            if objectness < self.objectness_threshold:
                continue

            scores = detection[5:]
            class_id = np.argmax(scores)
            class_score = float(scores[class_id])
            confidence = objectness * class_score
            if confidence < self.conf_threshold:
                continue

            # Raw YOLO detections are normalized relative to the projected frame.
            center_x = float(detection[0] * frame_width)
            center_y = float(detection[1] * frame_height)
            width = float(detection[2] * frame_width)
            height = float(detection[3] * frame_height)
            box = self._sanitize_box(
                center_x, center_y, width, height, frame_width, frame_height
            )
            if box is None:
                continue

            class_ids.append(class_id)
            confidences.append(confidence)
            boxes.append(box)

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        if not boxes:
            return class_ids, confidences, boxes, []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.conf_threshold,
            self.nms_threshold,
        )

        return class_ids, confidences, boxes, indices

    def process_output(self, input_img, frames):
        '''
        Main progress in the class.
        Detecting the pics >> Calculate Re-align BBox >> NMS selection >> Draw BBox

        Args:
            input_img(np.array): the original pano image
            frames(list)       : the results from pan2stereo, the list contain four spects of view

        Returns:
            base_frame(np.array): the input pano image with BBoxes
        '''
        first_flag = True
        outputs = None

        print('Yolo Detecting...')
        for face, frame in enumerate(frames):
            output = self.detect(frame)
            for i in range(output.shape[0]):
                # Move each face-space detection back into normalized panorama space.
                output[i, 0], output[i, 1], output[i, 2], output[i, 3] =\
                realign_bbox(output[i, 0], output[i, 1], output[i, 2], output[i, 3], face)
            if not first_flag:
                outputs = np.concatenate([outputs, output], axis=0)
            else:
                outputs = output
                first_flag = False

        base_frame = input_img
        # Need to inverse project into panorama space before the final NMS pass.
        class_ids, confidences, boxes, indices = self.nms_selection(base_frame, outputs)
        print('Painting Bounding Boxes..')
        for i in indices:
            if isinstance(i, (list, tuple, np.ndarray)):
                i = i[0]
            else:
                i = int(i)
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.draw_bbox(base_frame, class_ids[i], confidences[i],
                           left, top, left + width, top + height)

        return base_frame

def main():
    """
    Run detection on a single panorama image.
    """
    parser = argparse.ArgumentParser(description="Detect objects on a panorama image.")
    parser.add_argument("input", help="Path to the input panorama image.")
    parser.add_argument("output", help="Path to the output image with boxes.")
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Minimum objectness*class confidence to keep a box (default: %(default)s).",
    )
    parser.add_argument(
        "--objectness-threshold",
        type=float,
        default=OBJECTNESS_THRESHOLD,
        help="Minimum YOLO objectness score to keep a box (default: %(default)s).",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=NMS_THRESHOLD,
        help="NMS IoU threshold in panorama space (default: %(default)s).",
    )
    args = parser.parse_args()

    my_net = Yolo(
        conf_threshold=args.conf_threshold,
        objectness_threshold=args.objectness_threshold,
        nms_threshold=args.nms_threshold,
    )

    input_pano = cv2.imread(args.input)
    if input_pano is None:
        raise RuntimeError(f"Unable to read input panorama image: {args.input}")
    projections = pano2stereo(input_pano)

    output_frame = my_net.process_output(input_pano, projections)
    cv2.imwrite(args.output, output_frame)

if __name__ == '__main__':
    main()
