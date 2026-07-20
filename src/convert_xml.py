import os
import xml.etree.ElementTree as ET

def convert_box(size, box):
    # Normalized YOLO format: center_x, center_y, width, height (all 0 to 1)
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def main():
    lbl_dir = 'dataset/train/labels'
    # Define the exact class name found in your XML
    classes = ['numberplate']
    
    print("Starting robust XML to YOLO TXT conversion...")
    xml_files = [f for f in os.listdir(lbl_dir) if f.lower().endswith('.xml')]
    
    converted_count = 0
    
    for xml_file in xml_files:
        xml_path = os.path.join(lbl_dir, xml_file)
        txt_path = os.path.join(lbl_dir, xml_file.replace('.xml', '.txt'))
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image dimensions
            size_tag = root.find('size')
            if size_tag is None:
                continue
            width = int(size_tag.find('width').text)
            height = int(size_tag.find('height').text)
            
            yolo_lines = []
            
            # Loop through all objects in the XML
            for obj in root.findall('object'):
                cls_name = obj.find('name').text.strip().lower()  # strip spaces and match lowercase
                
                if cls_name in classes:
                    cls_id = classes.index(cls_name)
                    xml_box = obj.find('bndbox')
                    
                    xmin = float(xml_box.find('xmin').text)
                    ymin = float(xml_box.find('ymin').text)
                    xmax = float(xml_box.find('xmax').text)
                    ymax = float(xml_box.find('ymax').text)
                    
                    # Convert coordinates to YOLO format
                    bb = convert_box((width, height), (xmin, xmax, ymin, ymax))
                    yolo_lines.append(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")
            
            # Only write the text file if we actually found bounding boxes
            if yolo_lines:
                with open(txt_path, 'w') as f:
                    f.writelines(yolo_lines)
                converted_count += 1
                
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")

    print(f"Success! Converted {converted_count} XML files with valid bounding boxes.")

if __name__ == '__main__':
    main()