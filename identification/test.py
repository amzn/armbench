import os
import pickle
import json
import csv
import argparse


from PIL import Image
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import pdist
import torchvision.transforms as transforms
import torch


class Img2Vec():
    def __init__(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model == 'dino_v1':
            print ('Loading Dino V1 model')
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        else:
            print ('Loading Dino V2 model')
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        self.model = self.model.to(self.device)
        self.model.eval()

        self.scaler = transforms.Resize((224, 224))

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img):
        """ Get vector embedding from PIL image
        :param img: PIL Image
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)
        h_x = self.model(image)
        embedding = h_x.detach().cpu().numpy()[0, :]  
    
        return embedding
    
def get_visual_feature_distance_parallel(seg_feature, feature_embeddings_model, manifest_images):
	features = []

	features.append(seg_feature)
	for asin_image in manifest_images:
		asin_feature = feature_embeddings_model[asin_image]
		features.append(asin_feature)

	features = np.asarray(features)
	feature_distances = pdist(features, 'cosine')
	feature_distances = feature_distances[:features.shape[0] - 1]

	return feature_distances


def features_over_testcases(img2vec, testcases, observations, pick_location):
    testcase_embeddings = dict()
    
    for test_id, testcase in enumerate(testcases):
        print ('Extracting features for testcase %d/%d - %s' % (test_id, len(testcases), testcase))
        for obs in observations:
            obs_image_path = os.path.join(pick_location, testcase, obs)
            if os.path.exists(obs_image_path):
                obs_image = Image.open(obs_image_path)
                seg_feature = img2vec.get_vec(obs_image)
                testcase_embeddings[obs_image_path] = seg_feature
            else:
                testcase_embeddings[obs_image_path] = None

    return testcase_embeddings


def features_over_fnskus(img2vec, fnskus, gallery_location):
    feature_embeddings = dict()
    fnsku_map = dict()

    for fnsku_id, fnsku in enumerate(fnskus):
        print (f"Embedding extraction for FNSKU-{fnsku_id}")
        fnsku_directory = os.path.join(gallery_location, fnsku)
        fnsku_map[fnsku] = []
        if os.path.exists(fnsku_directory):
            for fnsku_image in os.listdir(fnsku_directory):
                filename = os.path.join(fnsku, fnsku_image)
                vec = img2vec.get_vec(Image.open(os.path.join(gallery_location,filename)))
                feature_embeddings[filename] = vec
                fnsku_map[fnsku].append(filename)

    return feature_embeddings, fnsku_map

def get_manifest_image_names(manifest, fnsku_map):
    manifest_images = []

    for fnsku in manifest:
        if fnsku in fnsku_map:
            manifest_images.extend(fnsku_map[fnsku])

    return manifest_images

def parse_arguments(parser):
    parser.add_argument("-d", "--dataset_path", default="", required=True)
    parser.add_argument("-o", "--observations", nargs='+', default=['PickRGB.jpg', 'FarTrayRGB.jpg', 'OnArmLowRGB.jpg', 'ToteWallRGB.jpg'], required=False)
    parser.add_argument("-m", "--model", default='dino_v1', choices=['dino_v1', 'dino_v2'], required=False)
    args = parser.parse_args()
    return args

def run_test(args):
    img2vec = Img2Vec(args.model)

    gallery_location = os.path.join(args.dataset_path, "Reference_Images")
    pick_location = os.path.join(args.dataset_path, "Picks")
    testset_path = os.path.join(args.dataset_path, "train-test-split.pickle")
    results_location = os.path.join(args.dataset_path, "output.csv")
    embedding_location = os.path.join(args.dataset_path, "embedding.pkl")
    testcase_embedding_location = os.path.join(args.dataset_path, "testcase_embedding.pkl")

    with open(testset_path, 'rb') as f:
        train_test_split = pickle.load(f)
        testcases = train_test_split['testset']
        reference_fnskus = train_test_split['testset-objects']

    feature_embeddings = {}
    if os.path.exists(embedding_location):
        feature_embeddings_file = open(embedding_location, 'rb')
        feature_embeddings_saved = pickle.load(feature_embeddings_file)
        feature_embeddings = feature_embeddings_saved['feature_embeddings']
        fnsku_map = feature_embeddings_saved['fnsku_map']
        feature_embeddings_file.close()
    else:
        print (f'Extracting features for reference images for {len(reference_fnskus)} FNSKUS')
        feature_embeddings, fnsku_map = features_over_fnskus(img2vec, reference_fnskus, gallery_location)
        feature_embeddings_file = open(embedding_location, 'wb')
        feature_embeddings_saved = dict()
        feature_embeddings_saved['feature_embeddings'] = feature_embeddings
        feature_embeddings_saved['fnsku_map'] = fnsku_map
        pickle.dump(feature_embeddings_saved, feature_embeddings_file)
        feature_embeddings_file.close()

    testcase_embeddings = {}
    if os.path.exists(testcase_embedding_location):
        testcase_embeddings_file = open(testcase_embedding_location, 'rb')
        testcase_embeddings_saved = pickle.load(testcase_embeddings_file)
        testcase_embeddings = testcase_embeddings_saved['testcase_embeddings']
        testcase_embeddings_file.close()
    else:
        print (f'Extracting features for {len(testcases)} testcases')
        testcase_embeddings = features_over_testcases(img2vec, testcases, args.observations, pick_location)
        testcase_embeddings_file = open(testcase_embedding_location, 'wb')
        testcase_embeddings_saved = dict()
        testcase_embeddings_saved['testcase_embeddings'] = testcase_embeddings
        pickle.dump(testcase_embeddings_saved, testcase_embeddings_file)
        testcase_embeddings_file.close()


    print (f'Number of testcases: {len(testcases)}')

    all_results = []
    fieldnames = ['testcase', 'predicted-fnsku', 'predicted-score', 'gt-fnsku', 'gt-ref', 'manifest-size', 'effective-manifest', 'all-predictions', 'all-scores', 'confidence']
    for test_id, testcase in enumerate(testcases):
        print ('Evaluating case %d/%d - %s' % (test_id, len(testcases), testcase))
        
        try:
            gt_file = os.path.join(pick_location, testcase, 'annotation.json')
            with open(gt_file, 'r') as infile:
                gt_json = json.load(infile)
            gt_fnsku = gt_json['GT_ID']
        except:
            print ("Cannot read GT")
            continue
        
        try:
            container_file = os.path.join(pick_location, testcase, 'container.json')
            with open(container_file, 'r') as infile:
                container_json = json.load(infile)
            fnskus = [fnsku for fnsku in container_json]
        except:
            print ("Cannot read manifest")
            continue

        # compute features
        seg_features = []
        for obs in args.observations:
            obs_image_path = os.path.join(pick_location, testcase, obs)
            feature = testcase_embeddings[obs_image_path]

            if feature is not None:
                seg_features.append(feature)

        manifest_images = get_manifest_image_names(fnskus, fnsku_map)
        if len(manifest_images) == 0:
            result = {'testcase': testcase, 
                      'predicted-fnsku': '', 
                      'predicted-score': 0,
                      'gt-fnsku': gt_fnsku, 
                      'gt-ref': False, 
                      'manifest-size': len(fnskus), 
                      'effective-manifest': 0,
                      'all-predictions': [],
                      'all-scores': [],
                      'confidence': 0.0}
            all_results.append(result)
            continue

        unique_fnskus = []
        for item in manifest_images:
            item = item.split('/')[0]
            if item not in unique_fnskus:
                unique_fnskus.append(item)

        # get nearest neighbor
        all_distances = []
        all_images = []
        for _, seg_feature in enumerate(seg_features):
            feature_distances = get_visual_feature_distance_parallel(seg_feature, feature_embeddings, manifest_images)
            all_distances.extend(feature_distances)
            all_images.extend(manifest_images)
        
        best_indices = np.argsort(all_distances)
        pred_imgname = all_images[best_indices[0]]
        pred_fnsku = pred_imgname.split('/')[0]

        unique_best_fnsku = []
        unique_best_score = []
        for i in range(0, len(best_indices)):
            curr_pred_imgname = all_images[best_indices[i]]
            curr_pred_fnsku = curr_pred_imgname.split('/')[0]
            
            if curr_pred_fnsku not in unique_best_fnsku:
                unique_best_fnsku.append(curr_pred_fnsku)
                unique_best_score.append(all_distances[best_indices[i]])

        probability = softmax(-np.array(unique_best_score))

        if len(probability) > 1:
            confidence = 1 - (probability[1]/probability[0])
        else:
            confidence = 1.0

        result = {'testcase': testcase, 
                  'predicted-fnsku': pred_fnsku, 
                  'predicted-score': all_distances[best_indices[0]],
                  'gt-fnsku': gt_fnsku, 
                  'gt-ref': gt_fnsku in unique_fnskus, 
                  'manifest-size': len(fnskus), 
                  'effective-manifest': len(unique_fnskus),
                  'all-predictions': unique_best_fnsku,
                  'all-scores': unique_best_score,
                  'confidence': confidence}
        
        all_results.append(result)

    with open(results_location, 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    run_test(args)