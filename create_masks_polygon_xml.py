import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

def create_mask_from_polygon_xml(xml_path, save_mask_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image size
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Loop over all objects
    for obj in root.findall("object"):

        polygon = obj.find("polygon")
        if polygon is None:
            continue

        points = []

        # Collect x,y pairs
        i = 1
        while True:
            x_tag = polygon.find(f"x{i}")
            y_tag = polygon.find(f"y{i}")

            if x_tag is None or y_tag is None:
                break

            x = int(float(x_tag.text))
            y = int(float(y_tag.text))

            points.append([x, y])
            i += 1

        if len(points) > 0:
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

    # Save mask
    Image.fromarray(mask).save(save_mask_path)


def process_folder(xml_dir, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)

    for file in os.listdir(xml_dir):
        if file.endswith(".xml"):
            xml_path = os.path.join(xml_dir, file)
            mask_name = file.replace(".xml", ".jpg")
            save_path = os.path.join(mask_dir, mask_name)

            create_mask_from_polygon_xml(xml_path, save_path)

    print("All masks created successfully.")


process_folder(
    xml_dir="dataset/crack/val/annotations",
    mask_dir="dataset/crack/val/masks"
)


