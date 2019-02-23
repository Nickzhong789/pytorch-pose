import os
import json
import numpy as np
import random


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


anno_dir = './cow_data/annotion/'
anno_total = []

anno_list = os.listdir(anno_dir)
val_idx_list = random.sample(range(len(anno_list)), k=100)
idx = 0
for anno_name in anno_list:
    anno_dict = dict()
    anno_file = anno_dir + anno_name
    with open(anno_file, 'r') as f:
        data = json.load(f)[0]
        print(data)
        # exit(0)

        anno_dict['dataset'] = 'Cow'
        if idx in val_idx_list:
            anno_dict['isValidation'] = 1.000
        else
            anno_dict['isValidation'] = 0.000
        image = data['image']
        anno_dict['img_paths'] = image['imgName']
        anno_dict['img_width'] = image['width']
        anno_dict['img_height'] = image['height']
        anno_dict['joint_self'] = []
        # print(data['bboxs'][0])
        # exit(0)
        if not data['bboxs']:
            continue
        bboxs = data['bboxs'][0]
        for kp in bboxs[0]['keypoints']:
            anno_dict['joint_self'].append([float('%.3f' % x) for x in kp[1:]])

        anno_total.append(anno_dict)
        bb = bboxs[0]['rectangle']
        center_x = (bb[2] - bb[0]) / 2
        center_y = (bb[3] - bb[1]) / 2
        anno_dict['objpos'] = [center_x, center_y]
        anno_dict['scale_provided'] = (bb[3] - bb[1])

with open('./annotation.json', 'w') as f:
    json.dump(anno_total, f, cls=MyEncoder)
