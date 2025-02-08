global_kpt_set_dict = {
    "human_hand": [
        "wrist",
        "thumb root", "thumb proximal joint", "thumb distal joint", "thumb tip",
        "forefinger root", "forefinger proximal joint", "forefinger distal joint", "forefinger tip",
        "middle finger root", "middle finger proximal joint", "middle finger distal joint", "middle finger tip",
        "ring finger root", "ring finger proximal joint", "ring finger distal joint", "ring finger tip",
        "pinky finger root", "pinky finger proximal joint", "pinky finger distal joint", "pinky finger tip"
    ],
    "human_face": [
        "right back cheek upper contour point", "right back cheek mid contour point",
        "right back cheek lower contour point",
        "right front cheek upper contour point", "right front cheek mid contour point",
        "right front cheek lower contour point",
        "right chin upper contour point", "right chin lower contour point",
        "mid chin contour point",
        "left chin lower contour point", "left chin upper contour point",
        "left front cheek lower contour point",
        "left front cheek mid contour point", "left front cheek upper contour point",
        "left back cheek lower contour point",
        "left back cheek mid contour point", "left back cheek upper contour point",
        "right brow tail", "right outer brow rise", "right brow arch",
        "right inner brow rise", "right brow beginning",
        "left brow beginning", "left inner brow rise",
        "left brow arch", "left outer brow rise", "left brow tail",
        "nose root", "upper nose bridge ", "lower nose bridge", "nose tip",
        "right nose wing", "right nose opening", "nose column", "left nose opening", "left nose wing",
        "right outer canthus", "right outer upper eyelid", "right inner upper eyelid",
        "right inner canthus", "right inner lower eyelid", "right outer lower eyelid",
        "left inner canthus", "left inner upper eyelid", "left outer upper eyelid",
        "left outer canthus", "left outer lower eyelid", "left inner lower eyelid",
        "right lip corner", "right upper lip margin", "right lip arch", "lip top",
        "left lip arch", "left upper lip margin",
        "left lip corner", "left lower lip margin", "left bottom lip bulge", "lip bottom",
        "right bottom lip bulge", "right lower lip margin",
        "right lip commissure", "right lip furrow", "upper lip indentation", "left lip furrow",
        "left lip commissure",
        "left top lip bulge", "lower lip indentation", "right top lip bulge",
        "lip center", "right eye", "left eye",
    ],
    'human_body': [
        "mid head top", "nose", "neck",
        "left eye", "right eye", "left ear", "right ear",
        "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist",
        "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"
    ],
    "mammal_body": [
        "mid head top", "nose", "neck", 'left ear', 'right ear',
        "left eye", "right eye", "upper lip", "lower lip", "left lip corner", "right lip corner",
        "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist",
        "mid back torso", "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle",
        "tail root", "tail middle", "tail end"
    ],
    "bird_body": [
        "back", "beak", "belly", "breast", "crown", "forehead", "left eye", "left leg", "left wing",
        "nape", "right eye", "right leg", "right wing", "tail", "throat"
    ],
    "insect_body": [
        "head", "neck", "thorax", "fore abdomen", "hind abdomen",
        "left antenna tip", "left antenna base", "left eye",
        "left foreleg base", "left foreleg knee", "left foreleg ankle", "left foreleg tip",
        "left midleg base", "left midleg knee", "left midleg ankle", "left midleg tip",
        "left hindleg base", "left hindleg knee", "left hindleg ankle", "left hindleg tip",
        "right antenna tip", "right antenna base", "right eye",
        "right foreleg base", "right foreleg knee", "right foreleg ankle", "right foreleg tip",
        "right midleg base", "right midleg knee", "right midleg ankle", "right midleg tip",
        "right hindleg base", "right hindleg knee", "right hindleg ankle", "right hindleg tip",
        "left wing", "right wing"
    ],
    "animal_face": [
        "right outer canthus", "right inner canthus", "left inner canthus", "left outer canthus",
        "nose tip", "right lip corner", "left lip corner", "upper lip", "lower lip"
    ],
    "vehicle": [
        "right front wheel", "left front wheel", "right back wheel", "left back wheel",
        "right head light", "left head light", "right rear light", "left rear light",
        "exhaust pipe",
        "right front roof corner", "left front roof corner", "right front back corner", "left front back corner"
    ],
    "furniture": [
        'left rear foot', 'right front armrest corner',
        'right rear foot', 'left front foot', 'right rear base corner',
        'left rear base corner', 'left front base corner', 'left top backrest corner',
        'middle column top', 'right front foot', 'left front armrest corner',
        'middle column bottom', 'right rear armrest corner', 'left rear armrest corner',
        'right top backrest corner', 'right front base corner',
        'wheel 1', 'wheel 2', 'wheel 3', 'wheel 4', 'wheel 5'
    ],
    "clothes": [
        'left mid inner sleeve side', 'right sleeve crown', 'left sleeve crown', 'right outer knee seam',
        'right lower opening', 'right bottom opening', 'left lower opening', 'left lower outer sleeve side',
        'right armhole', 'mid lower collar', 'right top opening', 'right chest seam', 'mid hem',
        'left bottom hem', 'right outer sleeve cuff', 'left upper opening', 'left waistline', 'left inner knee seam',
        'right lower collar', 'left upper outer sleeve side', 'left leg side seam', 'left chest seam',
        'left outer knee seam', 'right inner sleeve cuff', 'right lower outer sleeve side',
        'left inner sleeve cuff', 'right collar', 'left lower collar', 'left upper inner sleeve side',
        'right upper opening', 'right bottom hem', 'mid bottom hem', 'left collar', 'left armhole',
        'right mid inner sleeve side', 'left outer sleeve cuff', 'right upper outer sleeve side',
        'right inner knee seam', 'mid waistline', 'right leg side seam', 'right waistline', 'mid collar',
        'right upper inner sleeve side', 'right lower inner sleeve side', 'right leg cuff', 'left bottom opening',
        'left side seam', 'right side seam', 'left mid outer sleeve side', 'right hem', 'left top opening',
        'left leg cuff', 'left lower inner sleeve side', 'right mid outer sleeve side', 'left hem'
    ]
}

coco_kpts = [
    "nose", "left eye", "right eye", "left ear", "right ear",
    "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist", "left hip", "right hip",
    "left knee", "right knee", "left ankle", "right ankle"
]

onehand10k_kpts = [
    "wrist", "thumb root", "thumb proximal joint", "thumb distal joint", "thumb tip",
    "forefinger root", "forefinger proximal joint", "forefinger distal joint", "forefinger tip",
    "middle finger root", "middle finger proximal joint", "middle finger distal joint",
    "middle finger tip", "ring finger root", "ring finger proximal joint",
    "ring finger distal joint", "ring finger tip",
    "pinky finger root", "pinky finger proximal joint", "pinky finger distal joint", "pinky finger tip"
]

aflw_19kpt = [
    "right brow tail", "right brow arch", "right brow beginning", "left brow beginning", "left brow arch", "left brow tail",
    "right outer canthus", "right eye", "right inner canthus", "left inner canthus", "left eye", "left outer canthus",
    "right nose wing", "nose tip", "left nose wing",
    "right lip corner", "lip center", "left lip corner",
    "mid chin contour point"
]

face_300w = [
    "right back cheek upper contour point", "right back cheek mid contour point",
    "right back cheek lower contour point", "right front cheek upper contour point",
    "right front cheek mid contour point", "right front cheek lower contour point",
    "right chin upper contour point", "right chin lower contour point",
    "mid chin contour point", "left chin lower contour point",
    "left chin upper contour point", "left front cheek lower contour point", "left front cheek mid contour point",
    "left front cheek upper contour point", "left back cheek lower contour point",
    "left back cheek mid contour point", "left back cheek upper contour point",
    "right brow tail", "right outer brow rise", "right brow arch",
    "right inner brow rise", "right brow beginning",
    "left brow beginning", "left inner brow rise",
    "left brow arch", "left outer brow rise", "left brow tail",
    "nose root", "upper nose bridge ", "lower nose bridge", "nose tip",
    "right nose wing", "right nose opening", "nose column", "left nose opening",
    "left nose wing", "right outer canthus", "right outer upper eyelid", "right inner upper eyelid",
    "right inner canthus", "right inner lower eyelid", "right outer lower eyelid",
    "left inner canthus", "left inner upper eyelid", "left outer upper eyelid",
    "left outer canthus", "left outer lower eyelid", "left inner lower eyelid",
    "right lip corner", "right upper lip margin", "right lip arch", "lip top",
    "left lip arch", "left upper lip margin",
    "left lip corner", "left lower lip margin", "left bottom lip bulge", "lip bottom",
    "right bottom lip bulge", "right lower lip margin",
    "right lip commissure", "right lip furrow", "upper lip indentation", "left lip furrow",
    "left lip commissure", "left top lip bulge", "lower lip indentation", "right top lip bulge"
]

ap10k = [
    "left eye", "right eye", "nose", "neck", "tail root", "left shoulder", "left elbow", "left wrist",
    "right shoulder", "right elbow", "right wrist", "left hip", "left knee", "left ankle", "right hip",
    "right knee", "right ankle"
]

macaque_pose = [
    'nose',
    'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder',
    'left elbow', 'right elbow', 'left wrist', 'right wrist',
    'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle'
]

cub_200 = [
    "back", "beak", "belly", "breast", "crown", "forehead", "left eye", "left leg", "left wing",
    "nape", "right eye", "right leg", "right wing", "tail", "throat"
]

vinegar_fly = [
    "head", "left eye", "right eye", "neck", "thorax", "hind abdomen",
    "right foreleg base", "right foreleg knee", "right foreleg ankle", "right foreleg tip",
    "right midleg base", "right midleg knee", "right midleg ankle", "right midleg tip",
    "right hindleg base", "right hindleg knee", "right hindleg ankle", "right hindleg tip",
    "left foreleg base", "left foreleg knee", "left foreleg ankle", "left foreleg tip",
    "left midleg base", "left midleg knee", "left midleg ankle", "left midleg tip",
    "left hindleg base", "left hindleg knee", "left hindleg ankle", "left hindleg tip",
    "left wing", "right wing"
]

locust = [
    "head", "neck", "thorax", "fore abdomen", "hind abdomen",
    "left antenna tip", "left antenna base", "left eye",
    "left foreleg base", "left foreleg knee", "left foreleg ankle", "left foreleg tip",
    "left midleg base", "left midleg knee", "left midleg ankle", "left midleg tip",
    "left hindleg base", "left hindleg knee", "left hindleg ankle", "left hindleg tip",
    "right antenna tip", "right antenna base", "right eye",
    "right foreleg base", "right foreleg knee", "right foreleg ankle", "right foreleg tip",
    "right midleg base", "right midleg knee", "right midleg ankle", "right midleg tip",
    "right hindleg base", "right hindleg knee", "right hindleg ankle", "right hindleg tip",
]

animal_web = [
    "right outer canthus", "right inner canthus", "left inner canthus", "left outer canthus",
    "nose tip", "right lip corner", "left lip corner", "upper lip", "lower lip"
]

carfusion = [
    "right front wheel", "left front wheel", "right back wheel", "left back wheel",
    "right head light", "left head light", "right rear light", "left rear light",
    "exhaust pipe",
    "right front roof corner", "left front roof corner", "right front back corner", "left front back corner"
]

keypoint5_bed = [
    "right rear foot", "right front foot", "right rear base corner", "right front base corner",
    "right top backrest corner",
    "left rear foot", "left front foot", "left rear base corner", "left front base corner",
    "left top backrest corner"
]

keypoint5_chair = [
    "right front foot", "left front foot", "left rear foot", "right rear foot",
    "right front base corner", "left front base corner", "left rear base corner",
    "right rear base corner",
    "right top backrest corner", "left top backrest corner"
]

keypoint5_swivelchair = [
    "wheel 1", "wheel 2", "wheel 3", "wheel 4", "wheel 5",
    "middle column bottom", "middle column top",
    "right front base corner", "left front base corner",
    "left rear base corner", "right rear base corner",
    "right top backrest corner", "left top backrest corner"
]

keypoint5_sofa = [
    "right rear foot", "right front foot", "right rear base corner", "right front base corner",
    "right rear armrest corner", "right front armrest corner", "right top backrest corner",
    "left rear foot", "left front foot", "left rear base corner", "left front base corner",
    "left rear armrest corner", "left front armrest corner", "left top backrest corner"
]

keypoint5_table = [
    "right front base corner", "right rear base corner", "left front base corner",
    "left rear base corner",
    "right front foot", "right rear foot", "left front foot", "left rear foot"
]

df2_long_sleeved_dress = [
    "mid collar", "right collar", "right lower collar",
    "mid lower collar", "left lower collar", "left collar",
    "right sleeve crown", "right upper outer sleeve side",
    "right mid outer sleeve side", "right lower outer sleeve side",
    "right outer sleeve cuff", "right inner sleeve cuff",
    "right lower inner sleeve side", "right mid inner sleeve side",
    "right upper inner sleeve side", "right armhole",
    "right chest seam", "right side seam", "right waistline", "right leg side seam",
    "right bottom hem", "mid bottom hem", "left bottom hem",
    "left leg side seam", "left waistline", "left side seam", "left chest seam",
    "left armhole", "left upper inner sleeve side",
    "left mid inner sleeve side", "left lower inner sleeve side",
    "left inner sleeve cuff", "left outer sleeve cuff",
    "left lower outer sleeve side", "left mid outer sleeve side",
    "left upper outer sleeve side", "left sleeve crown"
]

df2_long_sleeved_outwear = [
    "mid collar", "right top opening",
    "right lower collar", "right collar", "left lower collar", "left collar",
    "right sleeve crown", "right upper outer sleeve side",
    "right mid outer sleeve side", "right lower outer sleeve side",
    "right outer sleeve cuff", "right inner sleeve cuff",
    "right lower inner sleeve side", "right mid inner sleeve side",
    "right upper inner sleeve side", "right armhole",
    "right chest seam", "right side seam",
    "right hem", "right bottom opening", "left hem",
    "left side seam", "left chest seam",
    "left armhole", "left upper inner sleeve side",
    "left mid inner sleeve side", "left lower inner sleeve side",
    "left inner sleeve cuff", "left outer sleeve cuff",
    "left lower outer sleeve side", "left mid outer sleeve side",
    "left upper outer sleeve side", "left sleeve crown",
    "left top opening", "left upper opening", "left lower opening", "left bottom opening",
    "right upper opening", "right lower opening"
]

df2_long_sleeved_shirt = [
    "mid collar", "right collar", "right lower collar",
    "mid lower collar", "left lower collar", "left collar",
    "right sleeve crown", "right upper outer sleeve side",
    "right mid outer sleeve side", "right lower outer sleeve side",
    "right outer sleeve cuff", "right inner sleeve cuff",
    "right lower inner sleeve side", "right mid inner sleeve side",
    "right upper inner sleeve side", "right armhole",
    "right chest seam", "right side seam",
    "right hem", "mid hem", "left hem",
    "left side seam", "left chest seam",
    "left armhole", "left upper inner sleeve side",
    "left mid inner sleeve side", "left lower inner sleeve side",
    "left inner sleeve cuff", "left outer sleeve cuff",
    "left lower outer sleeve side", "left mid outer sleeve side",
    "left upper outer sleeve side", "left sleeve crown"
]

df2_shorts = [
    "right waistline", "mid waistline", "left waistline",
    "right leg side seam", "right bottom hem", "right leg cuff",
    "mid bottom hem",
    "left leg cuff", "left bottom hem", "left leg side seam"
]

df2_short_sleeved_dress = [
    "mid collar", "right collar", "right lower collar",
    "mid lower collar", "left lower collar", "left collar",
    "right sleeve crown", "right upper outer sleeve side",
    "right outer sleeve cuff", "right inner sleeve cuff",
    "right upper inner sleeve side", "right armhole",
    "right chest seam", "right side seam", "right waistline", "right leg side seam",
    "right bottom hem", "mid bottom hem", "left bottom hem",
    "left leg side seam", "left waistline", "left side seam", "left chest seam",
    "left armhole", "left upper inner sleeve side",
    "left inner sleeve cuff", "left outer sleeve cuff",
    "left upper outer sleeve side", "left sleeve crown"
]

df2_short_sleeved_outwear = [
    "mid collar", "right top opening",
    "right lower collar", "right collar", "left lower collar", "left collar",
    "right sleeve crown", "right upper outer sleeve side",
    "right outer sleeve cuff", "right inner sleeve cuff",
    "right upper inner sleeve side", "right armhole",
    "right chest seam", "right side seam",
    "right hem", "right bottom opening", "left hem",
    "left side seam", "left chest seam",
    "left armhole", "left upper inner sleeve side",
    "left inner sleeve cuff", "left outer sleeve cuff",
    "left upper outer sleeve side", "left sleeve crown",
    "left top opening", "left upper opening", "left lower opening", "left bottom opening",
    "right upper opening", "right lower opening"
]

df2_short_sleeved_shirt = [
    "mid collar", "right collar", "right lower collar",
    "mid lower collar", "left lower collar", "left collar",
    "right sleeve crown", "right upper outer sleeve side",
    "right outer sleeve cuff", "right inner sleeve cuff",
    "right upper inner sleeve side", "right armhole",
    "right chest seam", "right side seam",
    "right hem", "mid hem", "left hem",
    "left side seam", "left chest seam",
    "left armhole", "left upper inner sleeve side",
    "left inner sleeve cuff", "left outer sleeve cuff",
    "left upper outer sleeve side", "left sleeve crown"
]

df2_skirt = [
    "right waistline", "mid waistline", "left waistline",
    "right leg side seam", "right bottom hem", "mid bottom hem", "left bottom hem", "left leg side seam"
]

df2_sling = [
    "mid collar", "right collar", "right lower collar",
    "mid lower collar", "left lower collar", "left collar",
    "right sleeve crown", "right armhole", "right side seam",
    "right hem", "mid hem", "left hem",
    "left side seam", "left armhole", "left sleeve crown"
]

df2_sling_dress = [
    "mid collar", "right collar", "right lower collar",
    "mid lower collar", "left lower collar", "left collar",
    "right sleeve crown", "right armhole",
    "right side seam", "right waistline", "right leg side seam",
    "right bottom hem", "mid bottom hem", "left bottom hem",
    "left leg side seam", "left waistline", "left side seam",
    "left armhole", "left sleeve crown"
]

df2_trousers = [
    "right waistline", "mid waistline", "left waistline",
    "right leg side seam", "right outer knee seam", "right bottom hem", "right leg cuff", "right inner knee seam",
    "mid bottom hem",
    "left inner knee seam", "left leg cuff", "left bottom hem", "left outer knee seam", "left leg side seam"
]

df2_vest = [
    "mid collar", "right collar", "right lower collar",
    "mid lower collar", "left lower collar", "left collar",
    "right sleeve crown", "right armhole", "right side seam",
    "right hem", "mid hem", "left hem",
    "left side seam", "left armhole", "left sleeve crown"
]

df2_vest_dress = [
    "mid collar", "right collar", "right lower collar",
    "mid lower collar", "left lower collar", "left collar",
    "right sleeve crown", "right armhole",
    "right side seam", "right waistline", "right leg side seam",
    "right bottom hem", "mid bottom hem", "left bottom hem",
    "left leg side seam", "left waistline", "left side seam",
    "left armhole", "left sleeve crown"
]

mp100_kpt_set_dict = {
    'hand': {'type': 'human_hand', 'keys': onehand10k_kpts},
    'face': {'type': 'human_face', 'keys': face_300w},
    'amur_tiger_body': {'type': 'human_face', 'keys': aflw_19kpt},
    'person': {'type': 'human_body', 'keys': coco_kpts},
    'antelope_body': {'type': 'mammal_body', 'keys': ap10k},
    'beaver_body': {'type': 'mammal_body', 'keys': ap10k},
    'bison_body': {'type': 'mammal_body', 'keys': ap10k},
    'bobcat_body': {'type': 'mammal_body', 'keys': ap10k},
    'cat_body': {'type': 'mammal_body', 'keys': ap10k},
    'cheetah_body': {'type': 'mammal_body', 'keys': ap10k},
    'cow_body': {'type': 'mammal_body', 'keys': ap10k},
    'deer_body': {'type': 'mammal_body', 'keys': ap10k},
    'dog_body': {'type': 'mammal_body', 'keys': ap10k},
    'elephant_body': {'type': 'mammal_body', 'keys': ap10k},
    'fox_body': {'type': 'mammal_body', 'keys': ap10k},
    'giraffe_body': {'type': 'mammal_body', 'keys': ap10k},
    'gorilla_body': {'type': 'mammal_body', 'keys': ap10k},
    'hamster_body': {'type': 'mammal_body', 'keys': ap10k},
    'hippo_body': {'type': 'mammal_body', 'keys': ap10k},
    'horse_body': {'type': 'mammal_body', 'keys': ap10k},
    'leopard_body': {'type': 'mammal_body', 'keys': ap10k},
    'lion_body': {'type': 'mammal_body', 'keys': ap10k},
    'otter_body': {'type': 'mammal_body', 'keys': ap10k},
    'panda_body': {'type': 'mammal_body', 'keys': ap10k},
    'panther_body': {'type': 'mammal_body', 'keys': ap10k},
    'pig_body': {'type': 'mammal_body', 'keys': ap10k},
    'polar_bear_body': {'type': 'mammal_body', 'keys': ap10k},
    'rabbit_body': {'type': 'mammal_body', 'keys': ap10k},
    'raccoon_body': {'type': 'mammal_body', 'keys': ap10k},
    'rat_body': {'type': 'mammal_body', 'keys': ap10k},
    'rhino_body': {'type': 'mammal_body', 'keys': ap10k},
    'sheep_body': {'type': 'mammal_body', 'keys': ap10k},
    'skunk_body': {'type': 'mammal_body', 'keys': ap10k},
    'spider_monkey_body': {'type': 'mammal_body', 'keys': ap10k},
    'squirrel_body': {'type': 'mammal_body', 'keys': ap10k},
    'weasel_body': {'type': 'mammal_body', 'keys': ap10k},
    'wolf_body': {'type': 'mammal_body', 'keys': ap10k},
    'zebra_body': {'type': 'mammal_body', 'keys': ap10k},
    'macaque': {'type': 'mammal_body', 'keys': macaque_pose},
    'Grebe': {'type': 'bird_body', 'keys': cub_200},
    'Gull': {'type': 'bird_body', 'keys': cub_200},
    'Kingfisher': {'type': 'bird_body', 'keys': cub_200},
    'Sparrow': {'type': 'bird_body', 'keys': cub_200},
    'Tern': {'type': 'bird_body', 'keys': cub_200},
    'Warbler': {'type': 'bird_body', 'keys': cub_200},
    'Woodpecker': {'type': 'bird_body', 'keys': cub_200},
    'Wren': {'type': 'bird_body', 'keys': cub_200},
    'fly': {'type': 'insect_body', 'keys': vinegar_fly},
    'locust': {'type': 'insect_body', 'keys': locust},
    'alpaca_face': {'type': 'animal_face', 'keys': animal_web},
    'arcticwolf_face': {'type': 'animal_face', 'keys': animal_web},
    'bighornsheep_face': {'type': 'animal_face', 'keys': animal_web},
    'blackbuck_face': {'type': 'animal_face', 'keys': animal_web},
    'bonobo_face': {'type': 'animal_face', 'keys': animal_web},
    'californiansealion_face': {'type': 'animal_face', 'keys': animal_web},
    'camel_face': {'type': 'animal_face', 'keys': animal_web},
    'capebuffalo_face': {'type': 'animal_face', 'keys': animal_web},
    'capybara_face': {'type': 'animal_face', 'keys': animal_web},
    'chipmunk_face': {'type': 'animal_face', 'keys': animal_web},
    'commonwarthog_face': {'type': 'animal_face', 'keys': animal_web},
    'dassie_face': {'type': 'animal_face', 'keys': animal_web},
    'fallowdeer_face': {'type': 'animal_face', 'keys': animal_web},
    'fennecfox_face': {'type': 'animal_face', 'keys': animal_web},
    'ferret_face': {'type': 'animal_face', 'keys': animal_web},
    'gentoopenguin_face': {'type': 'animal_face', 'keys': animal_web},
    'gerbil_face': {'type': 'animal_face', 'keys': animal_web},
    'germanshepherddog_face': {'type': 'animal_face', 'keys': animal_web},
    'gibbons_face': {'type': 'animal_face', 'keys': animal_web},
    'goldenretriever_face': {'type': 'animal_face', 'keys': animal_web},
    'greyseal_face': {'type': 'animal_face', 'keys': animal_web},
    'grizzlybear_face': {'type': 'animal_face', 'keys': animal_web},
    'guanaco_face': {'type': 'animal_face', 'keys': animal_web},
    'klipspringer_face': {'type': 'animal_face', 'keys': animal_web},
    'olivebaboon_face': {'type': 'animal_face', 'keys': animal_web},
    'onager_face': {'type': 'animal_face', 'keys': animal_web},
    'pademelon_face': {'type': 'animal_face', 'keys': animal_web},
    'proboscismonkey_face': {'type': 'animal_face', 'keys': animal_web},
    'przewalskihorse_face': {'type': 'animal_face', 'keys': animal_web},
    'quokka_face': {'type': 'animal_face', 'keys': animal_web},
    'bus': {'type': 'vehicle', 'keys': carfusion},
    'car': {'type': 'vehicle', 'keys': carfusion},
    'suv': {'type': 'vehicle', 'keys': carfusion},
    'bed': {'type': 'furniture', 'keys': keypoint5_bed},
    'chair': {'type': 'furniture', 'keys': keypoint5_chair},
    'sofa': {'type': 'furniture', 'keys': keypoint5_sofa},
    'swivelchair': {'type': 'furniture', 'keys': keypoint5_swivelchair},
    'table': {'type': 'furniture', 'keys': keypoint5_table},
    'long_sleeved_dress': {'type': 'clothes', 'keys': df2_long_sleeved_dress},
    'long_sleeved_outwear': {'type': 'clothes', 'keys': df2_long_sleeved_outwear},
    'long_sleeved_shirt': {'type': 'clothes', 'keys': df2_long_sleeved_shirt},
    'shorts': {'type': 'clothes', 'keys': df2_shorts},
    'short_sleeved_dress': {'type': 'clothes', 'keys': df2_short_sleeved_dress},
    'short_sleeved_outwear': {'type': 'clothes', 'keys': df2_short_sleeved_outwear},
    'short_sleeved_shirt': {'type': 'clothes', 'keys': df2_short_sleeved_shirt},
    'skirt': {'type': 'clothes', 'keys': df2_skirt},
    'sling': {'type': 'clothes', 'keys': df2_sling},
    'sling_dress': {'type': 'clothes', 'keys': df2_sling_dress},
    'trousers': {'type': 'clothes', 'keys': df2_trousers},
    'vest': {'type': 'clothes', 'keys': df2_vest},
    'vest_dress': {'type': 'clothes', 'keys': df2_vest_dress}
}


if __name__ == '__main__':
    for key in global_kpt_set_dict:
        print('{}=({}, 768),'.format(key, len(global_kpt_set_dict[key])))