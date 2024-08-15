# Dataset Setup

## DAIR-V2X-I
Download DAIR-V2X-I dataset from official [website](https://thudair.baai.ac.cn/index).
```
ln -s [single-infrastructure-side root] ./data/dair-v2x
python scripts/data_converter/dair2kitti.py --source-root data/dair-v2x-i --target-root data/dair-v2x-i-kitti
```

## Rope3D
Download Rope3D dataset from official [website](https://thudair.baai.ac.cn/index).
```
ln -s [rope3d root] ./data/rope3d
python scripts/data_converter/rope2kitti.py --source-root data/rope3d --target-root data/rope3d-kitti
```

## KITTI
Download KITTI dataset from official [website](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

## KITTI-360
Download KITTI-360 dataset from official [website](https://www.cvlibs.net/datasets/kitti-360/).

## Waymo
Download Waymo dataset from official [website](https://waymo.com/open/download/).

Set up environment
```
# Decompress the Waymo zip files into their corresponding directories
ls *.tar | xargs -i tar xvf {} -C your_target_dir
# Set up environment
conda create -n py36_waymo_tf python=3.7
conda activate py36_waymo_tf
conda install cudatoolkit=11.3 -c pytorch
# Newer versions of tf are not in conda. tf>=2.4.0 is compatible with conda.
pip install tensorflow-gpu==2.4
conda install pandas
pip3 install waymo-open-dataset-tf-2-4-0 --user
```

Parse Waymo dataset 
```
ln -s [waymo root] ./data/waymo/raw_data
conda activate py36_waymo_tf
python scripts/data_converter/converter.py --load_dir data/waymo --save_dir data/waymo/parse_data --split training   --num_proc 10
python scripts/data_converter/converter.py --load_dir data/waymo --save_dir data/waymo/parse_data --split validation   --num_proc 10
```

Convert Waymo dataset parsed to KITTI format
```
python scripts/data_converter/waymo2kitti.py --source-root data/waymo/parse_data --target-root data/waymo-kitti
```

BEVHeight
├── data
│   ├── waymo
│   │   ├── ImageSets
│   │   │   ├── train_tfrecord.txt
│   │   │   ├── val_tfrecord.txt
│   │   ├── raw_data
│   │   │   ├── training
│   │   │   │   ├── segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
│   │   │   ├── validation
│   │   │   │   ├── segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord
│   │   ├── parse_data
│   │   │   ├── training_org
│   │   │   │   ├── segment-id
│   │   │   ├── validation_org
│   │   │   │   ├── segment-id
│   │   ├── waymo_train_org.txt
│   │   ├── waymo_val_org.txt
│   ├── waymo-kitti
│   │   ├── ImageSets
│   │   ├── training
│   │   ├── validation


## nuScenes
Download Waymo dataset from official [website](https://www.nuscenes.org/nuscenes#download/).
```
git clone https://github.com/abhi1kumar/nuscenes-devkit

```

The directory will be as follows.
```
BEVHeight
├── data
│   ├── dair-v2x-i
│   │   ├── velodyne
│   │   ├── image
│   │   ├── calib
│   │   ├── label
|   |   └── data_info.json
|   └── dair-v2x-i-kitti
|   |   ├── training
|   |   |   ├── calib
|   |   |   ├── label_2
|   |   |   └── images_2
|   |   └── ImageSets
|   |        ├── train.txt
|   |        └── val.txt
|   ├── rope3d
|   |   ├── training
|   |   ├── validation
|   |   ├── training-image_2a
|   |   ├── training-image_2b
|   |   ├── training-image_2c
|   |   ├── training-image_2d
|   |   └── validation-image_2
|   ├── rope3d-kitti
|   |   ├── training
|   |   |   ├── calib
|   |   |   ├── denorm
|   |   |   ├── label_2
|   |   |   └── images_2
|   |   └── map_token2id.json
|   └── kitti
|   |   ├── training
|   |   |   ├── calib
|   |   |   ├── label_2
|   |   |   └── images_2
|   |   ├── testing
|   |   |   ├── calib
|   |   |   ├── label_2
|   |   |   └── images_2
|   |   └── ImageSets
|   |        ├── train.txt
|   |        └── val.txt
|   |        └── val.txt
|   └── kitti-360
|   |   ├── training
|   |   |   ├── calib
|   |   |   ├── label_2
|   |   |   └── images_2
|   |   ├── testing
|   |   |   ├── calib
|   |   |   ├── label_2
|   |   |   └── images_2
|   |   └── ImageSets
|   |        ├── train.txt
|   |        └── val.txt
|   |        └── val.txt
|   └── waymo-kitti
|   |   ├── training
|   |   |   ├── calib
|   |   |   ├── label_2
|   |   |   └── images_2
|   |   ├── testing
|   |   |   ├── calib
|   |   |   ├── label_2
|   |   |   └── images_2
|   |   └── ImageSets
|   |        ├── train.txt
|   |        └── val.txt
|   |        └── val.txt
|   |       
└── ...
```


## Visualize the dataset in KITTI format
```
python scripts/data_converter/visual_tools.py --data_root data/waymo-kitti --demo_dir ./demo
```

## Prepare infos for **DAIR-V2X-I**/**Rope3D**/**KITTI**/**KITTI-360**/**Waymo** datasets.
```
# DAIR-V2X-I Dataset
python scripts/gen_info_dair.py
# Rope3D Dataset
python scripts/gen_info_rope3d.py
# KITTI Dataset
python scripts/gen_info_kitti.py --data_root data/kitti
# KITTI-360 Dataset
python scripts/gen_info_kitti.py --data_root data/kitti-360
# Waymo Dataset
python scripts/gen_info_kitti.py --data_root data/waymo-kitti
```
