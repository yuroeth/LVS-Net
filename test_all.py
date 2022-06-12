import os
import os.path as osp
from time import sleep
import numpy as np

for i in range(343):
    os.system("CUDA_VISIBLE_DEVICES=0 python test_once.py --scene KingsCollege --experiment line0508/ --base_dir /cluster/project/infk/courses/252-0579-00L/group05/datasets/cambridge_line --image_id {}".format(i))
    sleep(2)

data_dir = "/cluster/project/infk/courses/252-0579-00L/group05/models/cambridge_logs/line0508/dataset[cambridge]scene[KingsCollege]train_batch_size[2]epochs[100]lr[0.0001]use_aug[True]seg_channel[64]/logs/cm_degree_metrics"

translation_list = []
angular_list = []
for i in range(343):
    file_path = osp.join(data_dir, "image_{}.npz".format(i)) 
    result_i = np.load(file_path)
    translation_list.append(result_i["trans_dist"][()])
    angular_list.append(result_i["ang_dist"][()])

print("line0508: translation median: {:.3f}, min translation: {:.3f}, max translation: {:.3f}, mean translation: {:.3f}".format(np.median(translation_list), np.min(translation_list), np.max(translation_list), np.mean(translation_list)))
print("line0508: angular median: {:.3f}, min angular: {:.3f}, max angular: {:.3f}, mean angular: {:.3f}".format(np.median(angular_list), np.min(angular_list), np.max(angular_list), np.mean(angular_list)))