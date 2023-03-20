import json,os
import pdb, time
import shutil
import sys

####################################################################################################

val_json_file = "instances_val2017.json"
val_folder = "val2017"

val_new_json_file = "instances_val2017_small.json"
val_new_imgs_name = "val_2017_small"

json_file = ""
img_folder = ""
dest_root = ""
new_json_file = ""
new_imgs_name = ""

trunc_images_list = []

TRAIN_NUM_IMAGES = 0
VAL_NUM_IMAGES = 4
NUM_IMAGES = ""

####################################################################################################

def generate_annotations(datadir, dest_root):
    json_file = val_json_file
    new_json_file = val_new_json_file

    data_tmp = datadir + "/" +  json_file
    with open(data_tmp, 'r') as f:
        coco = json.load(f)

    non_cnts = 0
    cnts = 0
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if type(ann['segmentation']) != type({}):
            if cnts < 2:
                trunc_images_list.append(img_id)
                cnts = cnts + 1
            else:
                continue
        else:
            if non_cnts < 2:
                trunc_images_list.append(img_id)
                non_cnts = non_cnts + 1
            else:
                continue
        if (cnts+non_cnts == VAL_NUM_IMAGES):
            break

    new_images = []
    new_annotations = []

    for img in coco['images']:
        if img['id'] in trunc_images_list:
            new_images.append(img)

    for ann in coco['annotations']:
        if ann['image_id'] in trunc_images_list:
            new_annotations.append(ann)

    coco['images'] = new_images
    coco['annotations'] = new_annotations

    print("begin to save")

    dest_tmp = dest_root + "/" + new_json_file
    with open(dest_tmp, 'w') as ff:
        json.dump(coco, ff)

def generate_images(dest_root, source_root):
    src_folder = ""
    new_imgs_name = val_new_imgs_name
    src_folder = val_folder

    new_dir_path = dest_root + "/" + new_imgs_name

    if os.path.exists(new_dir_path):
        shutil.rmtree(new_dir_path)
    os.mkdir(new_dir_path)

    for img in trunc_images_list:
        img_name = str(img).zfill(12) + ".jpg"
        src_file  = source_root+ "/" + src_folder + "/" + img_name
        dest_file = new_dir_path + "/" + img_name
        shutil.copyfile(src_file, dest_file)

def main():
    n = len(sys.argv)
    if n < 2:
        print("ATMOST ONE ARGUMENT EXPECTED: 0 for TRAINING and 1 for VALIDATION \n")
        print("USAGE: \n")
        print("python create_tiny_coco.py <path to Source coco folder> <path to empty folder [TRAIN/VAL]> \n")
        exit(0)

    source_root = sys.argv[1]
    dest_root = os.path.abspath(os.getcwd()) + "/" + sys.argv[2]

    datadir = source_root +  '/annotations/'

    if os.path.exists(dest_root):
        shutil.rmtree(dest_root)
    os.mkdir(dest_root)

    generate_annotations(datadir, dest_root)
    generate_images(dest_root, source_root)

if __name__=="__main__":
    main()