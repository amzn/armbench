import os
import argparse
import json
import shutil


def do_conversion(file, induct_folder, nominal_folder, multi_pick_folder, package_defect_folder, use_symb_link):
    if file.endswith('.json'):
        # check the label
        jf = open(os.path.join(induct_folder, file))
        data = json.load(jf)

        img_file = data['id']
        if not img_file.endswith('.jpg'):
            img_file = img_file + '.jpg'

        img_location = os.path.join(induct_folder, img_file)

        if os.path.isfile(img_location):
            label = data['label']

            if label == 'nominal':
                if use_symb_link:
                    os.symlink(img_location, os.path.join(nominal_folder, img_file))
                else:
                    shutil.copy(img_location, nominal_folder)

            elif (label == 'multi_pick'):
                if use_symb_link:
                    os.symlink(img_location, os.path.join(multi_pick_folder, img_file))
                else:
                    shutil.copy(img_location, multi_pick_folder)

            elif (label == 'package_defect'):
                if use_symb_link:
                    os.symlink(img_location, os.path.join(package_defect_folder, img_file))
                else:
                    shutil.copy(img_location, package_defect_folder)
            else:
                print ('unrecognized class label')
        else:
            print ("File not found: ", img_location)

        jf.close()


def convert_images(dataset_path, csv_location, output_folder, use_symb_link=True):
    f = open (csv_location, 'r')

    nominal_folder = os.path.join(output_folder, 'nominal')
    multi_pick_folder  = os.path.join(output_folder, 'multi_pick')
    package_defect_folder  = os.path.join(output_folder, 'package_defect')

    try:
        os.makedirs(nominal_folder)
        os.makedirs(multi_pick_folder)
        os.makedirs(package_defect_folder)
    except OSError as error:
        pass
    
    for line in f.readlines():
        id = line.strip()

        # get images under this induct ID
        induct_folder = os.path.join(dataset_path, 'data', id)

        if os.path.exists(induct_folder):

            [do_conversion(file, induct_folder, nominal_folder, multi_pick_folder, package_defect_folder, use_symb_link) for file in os.listdir(induct_folder)]

    f.close()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", required=True)
    parser.add_argument("--use_symb_link", action='store_true', default=True)

    args = parser.parse_args()
    train_csv = os.path.join(args.dataset_path, "train.csv")
    test_csv = os.path.join(args.dataset_path, "test.csv")

    train_path = os.path.join(args.dataset_path, "imagenet/train")
    val_path = os.path.join(args.dataset_path, "imagenet/test")

    # training images
    convert_images(args.dataset_path, train_csv, train_path, args.use_symb_link)

    # test images
    convert_images(args.dataset_path, test_csv, val_path, args.use_symb_link)




    
