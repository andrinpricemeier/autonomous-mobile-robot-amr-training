from typing import List
import xml.etree.ElementTree as ET


class VOCBoundingBox:
    """Represents a bounding box in a Pascal VOC file."""

    def __init__(self, name, xmin, ymin, xmax, ymax) -> None:
        self.name = name
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


class VOCAnnotation:
    """Represents an annotation in a Pascal VOC file."""
    def __init__(self, filename, width, height):
        self.filename = filename
        self.width = width
        self.height = height
        self.bounding_boxes: List[VOCBoundingBox] = []

    def add_bounding_box(self, bb: VOCBoundingBox) -> None:
        self.bounding_boxes.append(bb)

class VOCFile:
    """Represents a Pascal VOC file."""

    def __init__(self, filepath) -> None:
        self.filepath = filepath

    def load(self) -> VOCAnnotation:
        tree = self.__load_file()
        root = tree.getroot()
        annotation = VOCAnnotation(root.find('filename').text, int(float(root.find('size')[0].text)), int(float(root.find('size')[1].text)))
        for member in tree.getroot().findall("object"):
            bb = VOCBoundingBox(
                member[0].text,
                float(member[4][0].text),
                float(member[4][1].text),
                float(member[4][2].text),
                float(member[4][3].text),
            )
            annotation.add_bounding_box(bb)
        return annotation

    def save(self, annotation: VOCAnnotation, save_filepath: str) -> None:
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "my-project-name"
        ET.SubElement(root, "filename").text = annotation.filename
        ET.SubElement(root, "path").text = "my-project-name/" + annotation.filename
        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "Unspecified"
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(annotation.width)
        ET.SubElement(size, "height").text = str(annotation.height)
        ET.SubElement(size, "depth").text = "3"
        for bb in annotation.bounding_boxes:
            obj = self.__create_object(bb)
            root.append(obj)
        if save_filepath is None:
            save_filepath = self.filepath
        self.filepath = save_filepath
        tree = ET.ElementTree(root)
        tree.write(save_filepath, encoding="UTF-8")

    def __load_file(self) -> ET.ElementTree:
        return ET.parse(self.filepath)

    def __create_object(self, bb: VOCBoundingBox) -> ET.Element:
        obj = ET.Element("object")
        ET.SubElement(obj, "name").text = bb.name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "Unspecified"
        ET.SubElement(obj, "difficult").text = "Unspecified"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(bb.xmin)
        ET.SubElement(bndbox, "ymin").text = str(bb.ymin)
        ET.SubElement(bndbox, "xmax").text = str(bb.xmax)
        ET.SubElement(bndbox, "ymax").text = str(bb.ymax)
        return obj
