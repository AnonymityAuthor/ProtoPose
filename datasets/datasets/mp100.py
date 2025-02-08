# Copyright (c) OpenMMLab. All rights reserved.
import os
import copy
import numpy as np
from datasets import Compose
from datasets.datasets.base import BaseCOCODataset
from datasets.datasets.kpt_dict_lib import mp100_kpt_set_dict, global_kpt_set_dict


class MP100(BaseCOCODataset):
    """

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline: A sequence of data transforms.
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
                 fsl_pipeline=None,
                 num_samples_per_task=6,
                 num_shots=1,
                 num_qrys=15,
                 num_episodes=100):
        super().__init__(
            ann_file, img_prefix, data_cfg, pipeline, load_ann, test_mode)

        self.ann_info['use_different_joint_weights'] = False
        self.ann_info['joint_weights'] = None

        self.task_table = {
            'human_hand': ['hand', ],
            'human_face': ['face', 'amur_tiger_body'],
            'human_body': ['person', ],
            'mammal_body': ['antelope_body', 'beaver_body', 'bison_body', 'bobcat_body', 'cat_body',
                            'cheetah_body', 'cow_body', 'deer_body', 'dog_body', 'elephant_body', 'fox_body',
                            'giraffe_body', 'gorilla_body', 'hamster_body', 'hippo_body', 'horse_body', 'leopard_body',
                            'lion_body', 'otter_body', 'panda_body', 'panther_body', 'pig_body', 'polar_bear_body',
                            'rabbit_body', 'raccoon_body', 'rat_body', 'rhino_body', 'sheep_body', 'skunk_body',
                            'spider_monkey_body', 'squirrel_body', 'weasel_body', 'wolf_body', 'zebra_body', 'macaque'],
            'bird_body': ['Grebe', 'Gull', 'Kingfisher', 'Sparrow', 'Tern', 'Warbler', 'Woodpecker', 'Wren'],
            'insect_body': ['fly', 'locust'],
            'animal_face': ['alpaca_face', 'arcticwolf_face', 'bighornsheep_face', 'blackbuck_face', 'bonobo_face',
                            'californiansealion_face', 'camel_face', 'capebuffalo_face', 'capybara_face',
                            'chipmunk_face', 'commonwarthog_face', 'dassie_face', 'fallowdeer_face', 'fennecfox_face',
                            'ferret_face', 'gentoopenguin_face', 'gerbil_face', 'germanshepherddog_face',
                            'gibbons_face', 'goldenretriever_face', 'greyseal_face', 'grizzlybear_face', 'guanaco_face',
                            'klipspringer_face', 'olivebaboon_face',  'onager_face', 'pademelon_face',
                            'proboscismonkey_face', 'przewalskihorse_face', 'quokka_face'],
            'vehicle': ['bus', 'car', 'suv'],
            'furniture': ['bed', 'chair', 'sofa', 'swivelchair', 'table'],
            'clothes': ['long_sleeved_dress', 'long_sleeved_outwear', 'long_sleeved_shirt',
                        'shorts', 'short_sleeved_dress', 'short_sleeved_outwear', 'short_sleeved_shirt',
                        'skirt', 'sling', 'sling_dress', 'trousers', 'vest', 'vest_dress']
        }
        self.flip_index_dict = {
            'person': [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
            'hand': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'face': [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 26, 25, 24, 23, 22, 21, 20,
                     19, 18, 17, 27, 28, 29, 30, 35, 34, 33, 32, 31, 45, 44, 43, 42, 47, 46, 39, 38, 37, 36,
                     41, 40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63, 62, 61, 60, 67, 66, 65],
            'amur_tiger_body': [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 14, 13, 12, 17, 16, 15, 18],
            'mammal_body': [1, 0, 2, 3, 4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13],
            'macaque': [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
            'bird_body': [0, 1, 2, 3, 4, 5, 10, 11, 12, 9, 6, 7, 8, 13, 14],
            'fly': [0, 2, 1, 3, 4, 5, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 6, 7, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 31, 30],
            'locust': [0, 1, 2, 3, 4, 20, 21, 22, 23, 24, 25, 16, 27, 28, 29, 30, 31, 32, 33, 34, 5, 6, 7, 8, 9, 10,
                       11, 12, 13, 14, 15, 16, 17, 18, 19],
            'animal_face': [3, 2, 1, 0, 4, 6, 5, 7, 8],
            'vehicle': [1, 0, 3, 2, 5, 4, 7, 6, 8, 10, 9, 12, 11],
            'bed': [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
            'chair': [1, 0, 3, 2, 5, 4, 7, 6, 9, 8],
            'sofa': [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6],
            'swivelchair': [4, 3, 2, 1, 0, 5, 6, 8, 7, 10, 9, 12, 11],
            'table': [2, 3, 0, 1, 6, 7, 4, 5],
            'long_sleeved_dress': [0, 5, 4, 3, 2, 1, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,
                                   19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6],
            'long_sleeved_outwear': [0, 33, 4, 5, 2, 3, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 36, 18, 17,
                                     16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 1, 37, 38, 19, 34, 35],
            'long_sleeved_shirt': [0, 5, 4, 3, 2, 1, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
                                   15, 14, 13, 12, 11, 10, 9, 8, 7, 6],
            'shorts': [2, 1, 0, 9, 8, 7, 6, 5, 4, 3],
            'short_sleeved_dress': [0, 5, 4, 3, 2, 1, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
                                    12, 11, 10, 9, 8, 7, 6],
            'short_sleeved_outwear': [0, 25, 4, 5, 2, 3, 24, 23, 22, 21, 20, 19, 18, 17, 16, 28, 14, 13, 12, 11, 10, 9,
                                      8, 7, 6, 1, 29, 30, 15, 26, 27],
            'short_sleeved_shirt': [0, 5, 4, 3, 2, 1, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8,
                                    7, 6],
            'skirt': [2, 1, 0, 7, 6, 5, 4, 3],
            'sling': [0, 5, 4, 3, 2, 1, 14, 13, 12, 11, 10, 9, 8, 7, 6],
            'sling_dress': [0, 5, 4, 3, 2, 1, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6],
            'trousers': [2, 1, 0, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3],
            'vest': [0, 5, 4, 3, 2, 1, 14, 13, 12, 11, 10, 9, 8, 7, 6],
            'vest_dress': [0, 5, 4, 3, 2, 1, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6],
        }

        if load_ann:
            if 18 in self.coco.cats: # 19-kpt 'human face' named 'amur_tiger_body'
                self.coco.cats[18]['name'] = 'amur_tiger_body'

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
            self.db, self.id2Cat = self._get_db()

            self.db_cat_table = [self.classes[self._coco_ind_to_class_ind[cat_id]]
                                 for cat_id in self.id2Cat.values()]
            self.task_mapper, self.flip_index_mapper, self.meta_info_mapper = self._map_cat_info()

            self.num_samples_per_task = num_samples_per_task
            self.num_shots = num_shots
            self.num_qrys = num_qrys
            self.num_episodes = num_episodes

            if not test_mode:
                self.db_tasks = self._divide_data_by_task()
                self.all_groups = self._sample_pairs()
            else:
                self.fsl_pipeline = Compose(fsl_pipeline)
                self.all_groups = self._sample_fewshot_pairs()

        self.dataset_name = 'mp100'

    def _map_cat_info(self):
        task_mapper = dict()
        flip_index_mapper = dict()
        meta_info_mapper = dict()

        for key in self.task_table:
            cat_list = self.task_table[key]

            for cat in cat_list:
                task_mapper[cat] = key

        for key in task_mapper:
            if key in self.flip_index_dict:
                flip_index = self.flip_index_dict[key]
            else:
                flip_index = self.flip_index_dict[task_mapper[key]]

            flip_index_mapper[key] = flip_index

            meta_type = mp100_kpt_set_dict[key]['type']
            meta_keys = mp100_kpt_set_dict[key]['keys']
            global_keys = global_kpt_set_dict[task_mapper[key]]
            index_map = {value: index for index, value in enumerate(global_keys)}
            ids = [index_map[x] for x in meta_keys]
            meta_info_mapper[key] = dict(type=meta_type, ids=ids)

        return task_mapper, flip_index_mapper, meta_info_mapper

    def _divide_data_by_task(self):
        db_tasks = {item: list() for item in self.task_table}
        for i, cate_name in enumerate(self.db_cat_table):
            task_name = self.task_mapper[cate_name]
            db_tasks[task_name].append(i)

        dict_copy = db_tasks.copy()
        for key in dict_copy:
            if len(db_tasks[key]) > 0:
                db_tasks[key] = np.array(db_tasks[key])
            else:
                del db_tasks[key]

        return db_tasks

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

            # the number of keypoint for this specific category
            num_joints = int(len(obj['keypoints']) / 3)

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

    def pad_and_shuffle_ids(self):
        tasks, ids_list = self.db_tasks.keys(), self.db_tasks.values()
        max_len = max([len(self.db_tasks[key]) for key in self.db_tasks])
        sampled_ids_list = []
        for ids in ids_list:
            np.random.shuffle(ids)
            sampled_ids = np.random.choice(ids, max_len - len(ids), replace=True)
            sampled_ids_list.append(np.concatenate((ids, sampled_ids), axis=0))

        return dict(zip(tasks, sampled_ids_list))

    def _sample_pairs(self):
        db_tasks = self.pad_and_shuffle_ids()
        tasks = list(db_tasks.keys())
        max_len = len(db_tasks[tasks[0]])
        sampled_len = max_len // self.num_samples_per_task

        all_groups = list()
        for i in range(sampled_len):
            samples_per_group = list()
            for t in np.random.choice(len(db_tasks), len(db_tasks), replace=False):
                task_key = tasks[t]
                num_samples = self.num_samples_per_task
                sampled_ids = db_tasks[task_key][i*num_samples: (i+1)*num_samples]
                samples_per_group.append(dict(task=task_key, ids=sampled_ids))
            all_groups.append(samples_per_group)

        return all_groups

    def _sample_fewshot_pairs(self):
        assert self.test_mode
        db_cats = {cat: list() for cat in self.classes[1:]}
        for id, cat in enumerate(self.db_cat_table):
            db_cats[cat].append(id)

        all_groups = list()
        for ep in range(self.num_episodes):
            sup_ids = list()
            qry_ids = list()
            for cat in db_cats:
                ids = db_cats[cat]
                sampled_ids = np.random.choice(ids, size=self.num_shots + self.num_qrys, replace=False).tolist()
                sup_ids.append(sampled_ids[:self.num_shots])
                qry_ids.append(sampled_ids[self.num_shots:])
                self.eval_ids += [dict(id=id, ep=ep) for id in sampled_ids[self.num_shots:]]

            all_groups.append(dict(sup_ids=np.array(sup_ids), qry_ids=np.array(qry_ids)))

        return all_groups

    def _get_single_data(self, db_id):
        cat_name = self.db_cat_table[db_id]
        results = copy.deepcopy(self.db[db_id])
        results['ann_info'] = copy.deepcopy(self.ann_info)
        results['ann_info']['flip_index'] = self.flip_index_mapper[cat_name]
        results['ann_info']['num_output_channels'] = results['joints_3d'].shape[0]
        results['ann_info']['num_joints'] = results['joints_3d'].shape[0]
        results['ann_info']['prompt_embedding_info'] = self.meta_info_mapper[cat_name]
        results['ann_info']['category'] = cat_name

        return results

    def __getitem__(self, idx):
        """Get the sample given index."""
        if self.test_mode:
            results_dict = dict(
                sup_results=list(),
                qry_results=list(),
            )

            ep_group = self.all_groups[idx]

            # load support samples
            sup_ids = ep_group['sup_ids']
            sup_masks = list()
            for single_cat_ids in sup_ids:
                single_cat_mask = 0
                for db_id in single_cat_ids:
                    results = self._get_single_data(db_id)
                    results['task'] = self.task_mapper[self.db_cat_table[db_id]]
                    # results_list.append(self.fsl_pipeline(results))
                    results_dict['sup_results'].append(results)
                    single_cat_mask += results['joints_3d_visible'][:, 0]
                sup_masks.append(single_cat_mask == len(single_cat_ids))

            # load query samples
            qry_ids = ep_group['qry_ids']
            for i, single_cat_ids in enumerate(qry_ids):
                for db_id in single_cat_ids:
                    results = self._get_single_data(db_id)
                    results['ann_info']['eval_mask'] = sup_masks[i]
                    results['ann_info']['ep_id'] = idx
                    # results_list.append(self.pipeline(results))
                    results_dict['qry_results'].append(results)

            return results_dict
        else:
            results_list = list()
            single_group = self.all_groups[idx]
            for item in single_group:
                for db_id in item['ids']:
                    results = self._get_single_data(db_id)
                    results_list.append(self.pipeline(results))

            return results_list

    def __len__(self, ):
        """Get the size of the dataset."""
        return len(self.all_groups)
