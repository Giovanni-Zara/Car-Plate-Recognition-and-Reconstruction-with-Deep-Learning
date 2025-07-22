# THIS FILE IS FOR ANALYSIS ONLY, THAT IS WHY IT DOES NOT FOLLOW THE USUAL LAYOUT (GLOBALS, UTILS, ETC.)

from imports import *
from global_vars import *
from utils import *

# Ensure we're in the correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Current working directory: {os.getcwd()}")

# FOR TRAIN
input_dir = './ccpd_green/train'
output_dir = './yolo_dataset/images/train'
new_size = (640, 640)  # Replace with desired dimensions

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over all files in the input directory
#resize_images(input_dir, output_dir, new_size)



# call the class and the method to actually create the new datas with the txt annotations
img_directory = './yolo_dataset/images/train'
output_directory = './yolo_dataset/labels/train'
os.makedirs(output_directory, exist_ok=True)

# Create dataset with original and target sizes specified
dataset = LicensePlateDataset(img_directory, output_directory, 
                             original_size=(720, 1160), target_size=(640, 640))
dataset.create_annotation_files()





# FOR EVAL
input_dir = './ccpd_green/val'
output_dir = './yolo_dataset/images/val'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over all files in the input directory
#resize_images(input_dir, output_dir, new_size)


# call the class and the method to actually create the new datas with the txt annotations
img_directory = './yolo_dataset/images/val'
output_directory = './yolo_dataset/labels/val'
os.makedirs(output_directory, exist_ok=True)

# Create dataset with original and target sizes specified  
dataset = LicensePlateDataset(img_directory, output_directory,
                             original_size=(720, 1160), target_size=(640, 640))
dataset.create_annotation_files()

# OPTIONAL: Visualize a few bounding boxes to verify they're correct
print("\nTesting bounding box accuracy...")
if len(dataset.image_names) > 0:
    test_image = dataset.image_names[0]
    print(f"Testing with image: {test_image}")
    
    # Get coordinates for verification
    x_center, y_center, width, height = dataset.get_bbox_coords_YOLO_format(test_image)
    print(f"YOLO coordinates: center=({x_center:.3f}, {y_center:.3f}), size=({width:.3f}, {height:.3f})")
    
    # Verify coordinates are in valid range
    if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
        print(" Coordinates are in valid YOLO range [0, 1]")
    else:
        print(" WARNING: Coordinates are outside valid range!")
    
    # to save visualization
    dataset.visualize_bbox(test_image, f"bbox_test_{test_image}.png")

print("Dataset creation completed!")