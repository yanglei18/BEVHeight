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
