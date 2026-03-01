import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

def create_mask_from_bbox_xml(xml_path, save_mask_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image size
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Loop over objects
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        # Fill rectangle
        mask[ymin:ymax, xmin:xmax] = 255

    Image.fromarray(mask).save(save_mask_path)


def process_bbox_folder(xml_dir, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)

    for file in os.listdir(xml_dir):
        if file.endswith(".xml"):
            xml_path = os.path.join(xml_dir, file)
            mask_name = file.replace(".xml", ".jpg")
            save_path = os.path.join(mask_dir, mask_name)

            create_mask_from_bbox_xml(xml_path, save_path)

    print("Bounding box masks created.")


process_bbox_folder(
    xml_dir="dataset/taping/val/annotations",
    mask_dir="dataset/taping/val/masks"
)
