import json
import os
import shutil

# Path to the existing COCO dataset JSON file
coco_dataset_path = 'instances_val2017.json'  #main json file_name

# List of specific image IDs to include in the mini dataset
specific_image_ids = [25393, 45550, 86755, 90108, 91654, 91921, 148719, 293625, 394206, 479099] #specifica image id

# Output directory for the mini dataset
output_dir = './coco_new/'

# Load the existing COCO dataset JSON file
with open(coco_dataset_path, 'r') as f:
    coco_dataset = json.load(f)

# Create a new dictionary for the mini dataset
mini_dataset = {
    'info': coco_dataset['info'],
    'licenses': coco_dataset['licenses'],
    'categories': coco_dataset['categories'],
    'images': [],
    'annotations': []
}

# Iterate over the images in the existing COCO dataset
for ids in specific_image_ids:
    for image in coco_dataset['images']:
    # Check if the image ID is in the list of specific image IDs
        if image['id'] == ids:
            # Add the image to the mini dataset
            print(image['id'])
            mini_dataset['images'].append(image)
            # Find all the annotations for this image and add them to the mini dataset
            for annotation in coco_dataset['annotations']:
                if annotation['image_id'] == image['id']:
                    mini_dataset['annotations'].append(annotation)

# Write the mini dataset JSON file
os.makedirs(output_dir, exist_ok=True)
mini_dataset_path = os.path.join(output_dir, 'mini_dataset.json')# name of new json file 
with open(mini_dataset_path, 'w') as f:
    json.dump(mini_dataset, f)

# Copy the images for the mini dataset to the output directory
for image in mini_dataset['images']:
    image_path = os.path.join('/media/coco_10k_img/coco2017/val2017/', image['file_name']) #maindataset from which mini dataset is created
    shutil.copy(image_path, output_dir)
