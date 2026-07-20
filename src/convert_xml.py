import os
import xml.etree.ElementTree as ET

# Path to your XML files
xml_dir = 'dataset/train/labels'
# The exact class name you used in LabelImg
classes = ['license_plate'] 

def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

print("Starting XML to YOLO TXT conversion...")
converted_count = 0

for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith('.xml'):
        continue
    
    file_path = os.path.join(xml_dir, xml_file)
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    txt_filename = os.path.join(xml_dir, xml_file.replace('.xml', '.txt'))
    
    with open(txt_filename, 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), 
                 float(xmlbox.find('xmax').text), 
                 float(xmlbox.find('ymin').text), 
                 float(xmlbox.find('ymax').text))
            
            bb = convert_coordinates((w, h), b)
            out_file.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")
    
    converted_count += 1

print(f"Success! Converted {converted_count} XML files to YOLO TXT format.")