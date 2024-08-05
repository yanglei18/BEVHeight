import os
import pathlib
import re
import csv
import cv2
import math

import numpy as np

from tqdm import tqdm

def get_image_index_str(img_idx):
    return "{:06d}".format(img_idx)

def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    if len(lines) == 0 or len(lines[0]) < 15:
        content = []
    else:
        content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    '''
    annotations['bbox'] = np.array(
        [[0, 0, 200, 200] for x in content]).reshape(-1, 4) 
    '''
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]).reshape(
            -1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations

def clip2pi(ry):
    if ry > 2 * np.pi:
        ry -= 2 * np.pi
    if ry < - 2* np.pi:
        ry += 2 * np.pi
    return ry

def alpha2roty(alpha, pos):
    ry = alpha + np.arctan2(pos[0], pos[2])
    if ry > np.pi:
        ry -= 2 * np.pi
    if ry < -np.pi:
        ry += 2 * np.pi
    return ry

def load_calib_kitti(calib_filename):
    with open(calib_filename, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        P2, Tr_velo2cam = None, None
        for line, row in enumerate(reader):
            if row[0] == 'P2:':
                P2 = row[1:]
                P2 = [float(i) for i in P2]
                P2 = np.array(P2, dtype=np.float32).reshape(3, 4)
                continue
            elif row[0] == 'Tr_velo_to_cam:':
                Tr_velo2cam = row[1:]
                Tr_velo2cam = [float(i) for i in Tr_velo2cam]
                Tr_velo2cam = np.array(Tr_velo2cam, dtype=np.float32).reshape(3, 4)
                continue
    return Tr_velo2cam, P2

def cam2velo(r_velo2cam, t_velo2cam):
    Tr_velo2cam = np.eye(4)
    Tr_velo2cam[:3, :3] = r_velo2cam
    Tr_velo2cam[:3 ,3] = t_velo2cam.flatten()
    Tr_cam2velo = np.linalg.inv(Tr_velo2cam)
    r_cam2velo = Tr_cam2velo[:3, :3]
    t_cam2velo = Tr_cam2velo[:3, 3]
    return r_cam2velo, t_cam2velo

def normalize_angle(angle):
    alpha_tan = np.tan(angle)
    alpha_arctan = np.arctan(alpha_tan)
    if np.cos(angle) < 0:
        alpha_arctan = alpha_arctan + math.pi
    return alpha_arctan


def get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar):
    center_lidar = [float(center_lidar[0]), float(center_lidar[1]), float(center_lidar[2])]
    lidar_r = np.matrix([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]])
    l, w, h = obj_size
    center_lidar[2] = center_lidar[2] - h / 2
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = lidar_r * corners_3d_lidar + np.matrix(center_lidar).T
    return corners_3d_lidar.T

def get_camera_3d_8points(obj_size, yaw_lidar, center_lidar, center_in_cam, r_velo2cam, t_velo2cam):
    liadr_r = np.matrix([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]])
    l, w, h = obj_size
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = liadr_r * corners_3d_lidar + np.matrix(center_lidar).T
    if len(t_velo2cam.shape) == 1:
        t_velo2cam = t_velo2cam[:, np.newaxis]
    corners_3d_cam = r_velo2cam * corners_3d_lidar + t_velo2cam
    
    x0, z0 = corners_3d_cam[0, 0], corners_3d_cam[2, 0]
    x3, z3 = corners_3d_cam[0, 3], corners_3d_cam[2, 3]
    dx, dz = x0 - x3, z0 - z3
    yaw = math.atan2(-dz, dx)
    alpha = yaw - math.atan2(center_in_cam[0], center_in_cam[2])
    if alpha > math.pi:
        alpha = alpha - 2.0 * math.pi
    if alpha <= (-1 * math.pi):
        alpha = alpha + 2.0 * math.pi
    alpha_arctan = normalize_angle(alpha)
    return alpha_arctan, yaw

def bbbox2bbox(box3d, Tr_velo_to_cam, camera_intrinsic, img_size=[1920, 1080]):
    corners_3d = np.array(box3d)
    corners_3d_extend = np.concatenate(
        [corners_3d, np.ones((corners_3d.shape[0], 1), dtype=np.float32)], axis=1) 
    corners_3d_extend = np.matmul(Tr_velo_to_cam, corners_3d_extend.transpose(1, 0))
    
    corners_2d = np.matmul(camera_intrinsic, corners_3d_extend)
    corners_2d = corners_2d[:2] / corners_2d[2]
    box2d = np.array([min(corners_2d[0]), min(corners_2d[1]),
                      max(corners_2d[0]), max(corners_2d[1])])
    
    # [xmin, ymin, xmax, ymax]
    box2d[0] = max(box2d[0], 0.0)
    box2d[1] = max(box2d[1], 0.0)
    box2d[2] = min(box2d[2], img_size[0])
    box2d[3] = min(box2d[3], img_size[1])
    return box2d

def get_annos(idx):
    image_idx_str = get_image_index_str(idx)
    label_filename = os.path.join(label_folder, (image_idx_str + '.txt'))
    calib_filename = os.path.join(calib_folder, (image_idx_str + '.txt'))
    Tr_velo2cam, _ = load_calib_kitti(calib_filename)
    
    r_velo2cam, t_velo2cam = Tr_velo2cam[:3,:3], Tr_velo2cam[:3,3]
    r_cam2velo, t_cam2velo = cam2velo(r_velo2cam, t_velo2cam)
    Tr_cam2velo = np.eye(4)
    Tr_cam2velo[:3, :3], Tr_cam2velo[:3, 3] = r_cam2velo, t_cam2velo
    
    fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                    'dl', 'lx', 'ly', 'lz', 'ry']
    annos = list()
    with open(label_filename, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
        for line, row in enumerate(reader):
            class_name = row["type"]
            alpha = float(row["alpha"])
            pos = np.array((float(row['lx']), float(row['ly']), float(row['lz'])), dtype=np.float32)
            ry = float(row["ry"])
            if alpha > np.pi:
                alpha -= 2 * np.pi
                ry = alpha2roty(alpha, pos)
            alpha = clip2pi(alpha)
            ry = clip2pi(ry)
            yaw_lidar = 0.5 * np.pi - ry
            dim = [float(row['dl']), float(row['dw']), float(row['dh'])]
            truncated = int(row["truncated"])
            occluded = int(row["occluded"])
            loc_cam = np.array([float(row['lx']), float(row['ly']), float(row['lz']), 1.0]).reshape(4, 1)
            loc_lidar = np.matmul(Tr_cam2velo, loc_cam).squeeze(-1)[:3]
            loc_lidar[2] += 0.5 * float(row['dh'])
            annos.append(
                {
                    "class_name": class_name,
                    "alpha": alpha,
                    "loc_lidar": loc_lidar,
                    "loc_cam": loc_cam,
                    "truncated": truncated,
                    "occluded": occluded,
                    "yaw_lidar": yaw_lidar,
                    "ry": ry,
                    "dim": dim
                }
            )
    return annos

def write_kitti_in_txt(pred_lines, path_txt):
    wf = open(path_txt, "w")
    for line in pred_lines:
        line_string = " ".join(line) + "\n"
        wf.write(line_string)
    wf.close()

if __name__ == "__main__":
    print("hello world...")
    
    label_folder = "data/thutraf-i/training/label_2"
    calib_folder = "data/thutraf-i/training/calib"
    image_folder = "data/thutraf-i/training/image_2"
    rectified_label_folder = "data/thutraf-i/training/label_rectified"
    os.makedirs(rectified_label_folder, exist_ok=True)
    
    filepaths = pathlib.Path(label_folder).glob('*.txt')
    prog = re.compile(r'^\d{6}.txt$')
    filepaths = filter(lambda f: prog.match(f.name), filepaths)
    image_ids = [int(p.stem) for p in filepaths]
    image_ids = sorted(image_ids)

    for idx in tqdm(image_ids):
        image_idx_str = get_image_index_str(idx)
        label_filename = os.path.join(label_folder, (image_idx_str + '.txt'))
        rectified_label_filename = os.path.join(rectified_label_folder, (image_idx_str + '.txt'))
        calib_filename = os.path.join(calib_folder, (image_idx_str + '.txt'))
        image_filename = os.path.join(image_folder, (image_idx_str + '.jpg'))
        Tr_velo2cam, P2 = load_calib_kitti(calib_filename)
        r_velo2cam, t_velo2cam = Tr_velo2cam[:3, :3], Tr_velo2cam[:3, 3]
        annos = get_annos(idx)
        # image = cv2.imread(image_filename)
        img_size = [3500, 864]
        P2[0, 2] = 1725
        Tr_velo2cam[:3, 3] = np.array([0.032651, 3.229831, 40.107632])
        gt_lines = []
        for anno in annos:
            obj_size, yaw_lidar, loc_lidar, loc_cam, ry = anno["dim"], anno["yaw_lidar"], anno["loc_lidar"], anno["loc_cam"], anno["ry"]
            l, w, h = obj_size
            class_name = anno["class_name"]
            box = get_lidar_3d_8points([l, w, h], yaw_lidar, loc_lidar)
            alpha, yaw = get_camera_3d_8points(
                [l, w, h], yaw_lidar, loc_lidar, loc_cam, r_velo2cam, t_velo2cam
            )
            box2d = bbbox2bbox(box, Tr_velo2cam, P2[:3,:3], img_size)
            loc_cam = loc_cam[:,0].tolist()
            i1 = class_name
            i2 = str(0)
            i3 = str(0)
            i4 = str(round(alpha, 4))
            i5, i6, i7, i8 = (
                str(round(box2d[0], 4)),
                str(round(box2d[1], 4)),
                str(round(box2d[2], 4)),
                str(round(box2d[3], 4)),
            )
            i9, i10, i11 = str(round(h, 4)), str(round(w, 4)), str(round(l, 4))
            i12, i13, i14 = str(round(loc_cam[0], 4)), str(round(loc_cam[1], 4)), str(round(loc_cam[2], 4))
            i15 = str(round(ry, 4))
            line = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15]
            gt_lines.append(line)
            # cv2.rectangle(image, (int(box2d[0]), int(box2d[1])), (int(box2d[2]), int(box2d[3])), (0, 255, 0), 2) 
        # cv2.imwrite("image_bbox.jpg", image)
        write_kitti_in_txt(gt_lines, rectified_label_filename)   
