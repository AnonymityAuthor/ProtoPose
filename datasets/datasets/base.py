import os
import copy
import numpy as np
import json_tricks as json
from xtcocotools.coco import COCO
from torch.utils.data import Dataset
from datasets import Compose
from datasets.datasets.metric import keypoint_pck_accuracy, keypoint_mpck_accuracy


class BaseCOCODataset(Dataset):
    """Base class for top-down datasets.

    All top-down datasets should subclass it.
    All subclasses should overwrite:
        Methods:`_get_db`

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline : A sequence of data transforms.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 load_ann=True,
                 test_mode=False,
                 bbox_file=None):

        self.image_info = {}
        self.ann_info = {}

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.bbox_file = bbox_file
        self.use_gt_bbox = True if bbox_file is None else False

        self.det_bbox_thr = data_cfg['det_bbox_thr']
        self.use_nms = data_cfg['use_nms']
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']

        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])

        self.db = []
        self.eval_ids = []
        self.sigmas = None
        self.dataset_name = None
        self.pipeline = Compose(self.pipeline)

        if load_ann:
            self.coco = COCO(self.ann_file)

            cats = [
                cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())
            ]
            self.classes = ['__background__'] + cats
            self.num_classes = len(self.classes)
            self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
            self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
            self._coco_ind_to_class_ind = dict(
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:])
            self.img_ids = self.coco.getImgIds()
            self.num_images = len(self.img_ids)
            self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)

    def __len__(self, ):
        """Get the size of the dataset."""
        return len(self.db)

    def __getitem__(self, idx):
        """Get the sample given index."""
        results = copy.deepcopy(self.db[idx])
        results['ann_info'] = self.ann_info

        return self.pipeline(results)

    def _xywh2cs(self, x, y, w, h, padding_ratio=1.25):
        """This encodes bbox(x,y,w,w) into (center, scale)

        Args:
            x, y, w, h

        Returns:
            tuple: A tuple containing center and scale.

            - center (np.ndarray[float32](2,)): center of the bbox (x, y).
            - scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        aspect_ratio = self.ann_info['image_size'][0] / self.ann_info[
            'image_size'][1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        # if (not self.test_mode) and np.random.rand() < 0.3:
        #     center += 0.4 * (np.random.rand(2) - 0.5) * [w, h]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

        scale = scale * padding_ratio

        return center, scale

    def _get_mapping_id_name(self, imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    def _get_db(self):
        """Load dataset."""
        gt_db, id2Cat = self._load_coco_keypoint_annotations()

        return gt_db, id2Cat

    def _load_coco_keypoint_annotations(self):
        """Ground truth bbox and keypoints."""
        gt_db, id2Cat = [], dict()
        for img_id in self.img_ids:
            db_tmp, id2Cat_tmp = self._load_coco_keypoint_annotation_kernel(img_id)
            gt_db.extend(db_tmp)
            id2Cat.update(id2Cat_tmp)
        return gt_db, id2Cat

    def _load_coco_keypoint_annotation_kernel(self, image_id):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]
        Args:
            image_id: coco image id
        Returns:
            dict: db entry
        """
        img_ann = self.coco.loadImgs(image_id)[0]
        width = img_ann['width']
        height = img_ann['height']
        num_joints = self.ann_info['num_joints']

        ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w))
            y2 = min(height - 1, y1 + max(0, h))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        bbox_id = 0
        kpt_db = []
        id2Cat = dict()
        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            center, scale = self._xywh2cs(*obj['clean_bbox'][:4])
            category_id = obj['category_id']
            image_file = os.path.join(self.img_prefix, self.id2name[image_id])
            kpt_db.append({
                'image_file': image_file,
                'center': center,
                'scale': scale,
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'bbox': obj['clean_bbox'],
                'bbox_score': 1,
                'bbox_id': bbox_id,
                'dataset': self.dataset_name
            })

            id2Cat.update({(image_id, bbox_id): category_id})
            bbox_id = bbox_id + 1

        return kpt_db, id2Cat

    def _prepare_keypoint_results(self, results, **kwargs):
        kpts = []
        for result in results:
            preds = result['preds']
            boxes = result['boxes']
            image_paths = result['image_paths']
            bbox_ids = result['bbox_ids']
            ep_ids = result['ep_ids']
            eval_masks = result['eval_masks']

            batch_size = len(image_paths)
            for i in range(batch_size):
                kpts.append({
                    'keypoints': preds[i].tolist(),
                    'center': boxes[i][0:2].tolist(),
                    'scale': boxes[i][2:4].tolist(),
                    'area': boxes[i][4],
                    'score': boxes[i][5],
                    'image_id': self.name2id[image_paths[i][len(self.img_prefix):]],
                    'bbox_id': bbox_ids[i],
                    'ep_id': ep_ids[i],
                    'eval_mask': eval_masks[i]
                })

        return kpts

    def _write_keypoint_results(self, keypoints, res_file):
        """Write results into a json file."""

        # align results and gts
        keypoints_dict = dict()
        for img_kpts in keypoints:
            image_id = img_kpts['image_id']
            bbox_id = img_kpts['bbox_id']
            ep_id = img_kpts['ep_id']
            keypoints_dict.update({(image_id, bbox_id, ep_id): img_kpts})

        db_inds = list()
        db_eval = [self.db[item['id']] for item in self.eval_ids]
        ep_ids = [item['ep'] for item in self.eval_ids]
        for i, gt in enumerate(db_eval):
            bbox_id = gt['bbox_id']
            image_file = gt['image_file']
            image_id = self.name2id[image_file[len(self.img_prefix):]]
            ep_id = ep_ids[i]
            db_inds.append((image_id, bbox_id, ep_id))

        results = list()
        for id in db_inds:
            results.append(keypoints_dict[id])

        # write json file
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _get_normalize_factor(self, gts, *args, **kwargs):
        """Get the normalize factor. generally inter-ocular distance measured
        as the Euclidean distance between the outer corners of the eyes is
        used. This function should be overrode, to measure NME.

        Args:
            gts (np.ndarray[N, K, 2]): Groundtruth keypoint location.

        Returns:
            np.ndarray[N, 2]: normalized factor
        """
        return np.ones([gts.shape[0], 2], dtype=np.float32)

    def pad_array_list(self, arr_list):
        max_length = max(len(arr) for arr in arr_list)
        padded_arrays = list()
        for arr in arr_list:
            pad_width = [(0, max_length - len(arr))] + [(0, 0)] * (len(arr.shape) - 1)
            padded_arrays.append(np.pad(arr, pad_width, 'constant', constant_values=0))

        return np.array(padded_arrays)

    def _do_keypoint_eval(self, res_file, metric):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metric (dict): Metric to be performed.
                Options: 'PCK', 'mPCK'.

        Returns:
            List: Evaluation results for evaluation metric.
        """
        db_eval = [self.db[item['id']] for item in self.eval_ids]
        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(db_eval)

        outputs = []
        gts = []
        masks = []
        threshold_bbox = []

        for pred, item in zip(preds, db_eval):
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(item['joints_3d'])[:, :-1])
            masks.append(np.bitwise_and((np.array(item['joints_3d_visible'])[:, 0]) > 0, pred['eval_mask']))
            bbox = np.array(item['bbox'])
            bbox_thr = np.max(bbox[2:])
            threshold_bbox.append(np.array([bbox_thr, bbox_thr]))

        outputs = self.pad_array_list(outputs)
        gts = self.pad_array_list(gts)
        masks = self.pad_array_list(masks)
        threshold_bbox = np.array(threshold_bbox)

        if metric['item'] == 'PCK':
            pck = keypoint_pck_accuracy(outputs, gts, masks, metric['pck_thr'], threshold_bbox)
            ref_score = pck
            info_str = 'PCK:{:.4f}'.format(pck)
        elif metric['item'] == 'mPCK':
            mpck, pcks = keypoint_mpck_accuracy(outputs, gts, masks, metric['pck_thr'], threshold_bbox)
            ref_score = mpck
            info_str = 'mPCK:{:.4f};  '.format(mpck)
            info_str += '||'.join(['PCK@{:.2f}:{:.4f}'.format(th, pck) for th, pck in zip(metric['pck_thr'], pcks)])
        else:
            raise NotImplementedError

        return ref_score, info_str

    def evaluate(self, results, res_folder, metric, **kwargs):
        assert res_folder is not None
        res_file = os.path.join(res_folder, 'result_keypoints.json')

        kpts = self._prepare_keypoint_results(results, **kwargs)
        self._write_keypoint_results(kpts, res_file)
        score, info_str = self._do_keypoint_eval(res_file, metric)

        return score, info_str