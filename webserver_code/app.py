import cv2
import joblib
import hdbscan
from PIL import Image
from umap import UMAP
from ultralyticsplus import YOLO
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

model = YOLO('yolov8m.pt')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

with open('bb_cluster.pkl', 'rb') as file:
    hdbscan_cluster = joblib.load(file)

with open('umap_embed.pkl', 'rb') as file:
    umap_embed = joblib.load(file)
    
clip_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")

app = Flask(__name__)

def extract_persons(image_filepath):
    """
        Code for processing the image and extracting persons from the image
    """
    # yolov8 medium model prediction
    results = model.predict(image_filepath)
    
    person_detect = dict()
    person_detect['image_path'] = image_filepath
    
    for result in results:                                          # iterate results
        boxes = result.boxes.cpu().numpy()                          # get boxes on cpu in numpy
        
        for box in boxes:                                           # iterate boxes
            r = box.xyxy[0].astype(int)                             # get corner points as int

            if model.model.names[int(box.cls.item())] == 'person':

                if 'bounding_box' not in person_detect.keys():
                    person_detect['bounding_box'] = [r]
                else:
                    person_detect['bounding_box'].append(r)
                    
    return person_detect

def cluster_persons(person_detect):
    """
        This function creates embedding for the bounding box, compress it and attributes cluster to it.
    """
    cluster_ids = list()
    image = cv2.imread(person_detect['image_filepath'])
    
    for bbox in person_detect['bounding_box']:
        
        bb_img = image[bbox[1]:bbox[1]+r[3],bbox[0]:bbox[0]+r[2]]
        embedding = clip_model.encode(Image.fromarray(bb_img))
        umap_trans = umap_embed.transform([embedding])
        cluster, _ = hdbscan.approximate_predict(cl2, [umap_trans])
        
        cluster_id.append(cluster)
        
    return cluster_ids
                    

@app.route('/get_people_from_image', methods=['POST'])
def get_people_from_image():

    image_filepath = request.json['image_filepath']

    person_detect = extract_persons(image_filepath)
    cluster_labels = cluster_persons(person_detect)

    result = {'bounding_boxes': person_detect['bounding_box'], 'cluster_labels': cluster_labels}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
