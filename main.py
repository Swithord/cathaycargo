import time

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO
import numpy as np


detection_graphics = [
    ('UN marker', (0, 0, 0)),
    ('UN number', (255, 0, 0)),
    ('Cargo Only', (0, 255, 0)),
    ('Class 1', (0, 255, 0)),
    ('Class 2', (0, 0, 255)),
    ('Class 3', (255, 255, 0)),
    ('Class 4', (255, 0, 255)),
    ('Class 5', (0, 255, 255)),
    ('Class 6', (255, 255, 255)),
    ('Class 7', (128, 0, 0)),
    ('Class 8', (0, 128, 0)),
    ('Class 9', (0, 0, 128)),
    ('Goods', (0, 128, 128)),
    ('Limited Quantities', (128, 128, 128)),
    ('Lithium Batteries', (128, 0, 128)),
    ('Orientation Marker', (128, 128, 0)),
]



class DetectedObject:
    def __init__(self, cls, confidence, bounding_box, bounding_box_xywh):
        assert len(bounding_box) == 4
        self.cls = int(cls)
        self.confidence = float(confidence)
        self.bounding_box = bounding_box  # In xyxy format
        self.bounding_box_xywh = bounding_box_xywh  # In xywh format
        self.contained_objects = []

    def completely_contains(self, other):
        x1, y1, x2, y2 = self.bounding_box
        otherx1, othery1, otherx2, othery2 = other.bounding_box
        return (x1 < otherx1 and x2 > otherx2 and y1 < othery1 and y2 > othery2)

    def partially_contains(self, other):
        x1, y1, x2, y2 = self.bounding_box
        otherx1, othery1, otherx2, othery2 = other.bounding_box
        return not (otherx2 < x1 or otherx1 > x2 or othery2 < y1 or othery1 > y2)

    def contains(self, other):
        # select your strategy here
        return self.partially_contains(other)

    def validate(self):
        if self.cls != 12:  # not goods
            return True
        # current logic: if UN number exists but no DG label, not valid
        contains_un = any({True if obj.cls in [0, 1] else False for obj in self.contained_objects})
        contains_dg = any(
            {True if obj.cls in [3, 4, 5, 6, 7, 8, 9, 10, 11] else False for obj in self.contained_objects})
        return not contains_un or contains_dg


# resultObj is an image
def get_DetectedObject_from_image(resultObj):
    boxes = resultObj.boxes
    classes = boxes.cls.tolist()
    confidences = boxes.conf.tolist()
    bounding_boxes = boxes.xyxy.tolist()
    bounding_boxes_xywh = boxes.xywh.tolist()

    try:
        assert (len(classes) == len(confidences) == len(bounding_boxes))
    except AssertionError:
        print('List lengths do not match')

    non_goods_objects = []
    goods_objects = []
    for i in range(len(classes)):
        cls = classes[i]
        obj = DetectedObject(cls, confidences[i], bounding_boxes[i], bounding_boxes_xywh[i])
        if cls == 12:  # goods
            goods_objects.append(obj)
        else:
            non_goods_objects.append(obj)

    for good in goods_objects:
        for non_good in non_goods_objects:
            if good.contains(non_good):
                good.contained_objects.append(non_good)

    return non_goods_objects, goods_objects


def get_box_color(valid):
    if valid == True:
        return 'g'
    return 'r'


def get_textbox(valid):
    return 'valid' if valid else 'invalid'


def print_validation_results(goods_objects):
    for good in goods_objects:
        if good.validate():
            print(f'Good at {good.bounding_box_xywh} passed')
        else:
            print(f'Good at {good.bounding_box_xywh} did not pass')


# Load the YOLOv9 model
model = YOLO('best.pt')


def get_objects_aggregation(non_goods_objects):
    name_conversions = ['UN number', 'UN marker', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9', '', '', '', '', '', '']
    agg = {}
    for object in non_goods_objects:
        try:
            name = name_conversions[object.cls]
        except IndexError:
            continue

        if name not in agg:
            agg[name] = 1
        else:
            agg[name] += 1

    return agg

import cv2
import numpy as np

def plot_custom_boxes(frame, non_goods_objects, goods_objects):
    for good in goods_objects:
        if good.contained_objects:
            x, y, w, h = good.bounding_box_xywh
            x -= w / 2
            y -= h / 2
            color = (0, 255, 0) if good.validate() else (0, 0, 255)
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 5)
            cv2.putText(frame, get_textbox(good.validate()), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)

    for good in non_goods_objects:
        x, y, w, h = good.bounding_box_xywh
        x -= w / 2
        y -= h / 2
        text, color = detection_graphics[good.cls]
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 5)
        cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)


def main():
    # Initialize the camera
    camera_index = 0
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to access camera with index {camera_index}.")
        return

    print('Press "q" to exit the program.')

    # Give the camera a few seconds to initialize
    time.sleep(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('?')
            break

        # Run object detection
        result = model(frame)

        if isinstance(result, list):
            result = result[0]

        non_goods_objects, goods_objects = get_DetectedObject_from_image(result)

        # Plot custom boxes
        plot_custom_boxes(frame, non_goods_objects, goods_objects)

        # Display the resulting frame
        cv2.imshow('YOLOv9 Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
