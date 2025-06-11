from evaluator_SO import EvaluatorSO
from evaluator_MO import EvaluatorMO


project_path = '/ivi/zfs/s0/original_homes/jliu4/PCISeg/'
datasets = ['scannet40', 's3dis', 'kitti360']


val_list_single = ['../../datasets/ScanNet/single/object_ids.npy', '../../datasets/S3DIS/single/object_ids.npy', '../../datasets/KITTI360/single/object_ids.npy']
val_list_classes_single = ['../../datasets/ScanNet/single/object_classes.txt', '../../datasets/S3DIS/single/object_classes.txt', '../../datasets/KITTI360/single/object_classes.txt']
results_name_single = ['../results/ScanNet_single/val_results_single.csv', '../results/S3DIS_single/val_results_single.csv', '../results/KITTI360_single/val_results_single.csv']

val_list_multi = ['../../datasets/ScanNet/val_list.json', '../../datasets/S3DIS/val_list.json', '../../datasets/KITTI360/val_list.json']
results_name_multi = ['../results/ScanNet_multi/val_results_multi.csv', '../results/S3DIS_multi/val_results_multi.csv', '../results/KITTI360_multi/val_results_multi.csv']

if __name__ == '__main__':
    idx = 2  # 0: ScanNet, 1: S3DIS, 2: KITTI360
    single = False # True: single object, False: multi object

    if single:
        evaluator = EvaluatorSO(datasets[idx], val_list_single[idx], val_list_classes_single[idx], results_name_single[idx], [0.5,0.65,0.8,0.85,0.9])
        results_dict = evaluator.eval_results()
    else:
        evaluator = EvaluatorMO(val_list_multi[idx], results_name_multi[idx], [0.5,0.65,0.8,0.85,0.9])
        results_dict = evaluator.eval_results()