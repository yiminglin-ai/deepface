import logging
import os

import cv2
import mxnet as mx
import numpy as np
from deepface.DeepFace import build_model
from deepface.extendedmodels import Age
from keras.preprocessing import image as KPI
from torch.utils.data import Dataset
from tqdm import tqdm


class MXFaceDataset(Dataset):
    """
    Mxnet RecordIO face dataset.
    """

    def __init__(self, root_dir: str, transforms=None, split="training") -> None:
        super(MXFaceDataset, self).__init__()
        self.transform = transforms
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, f"{split}.rec")
        path_imgidx = os.path.join(root_dir, f"{split}.idx")
        path_imglst = os.path.join(root_dir, f"{split}.lst")
        items = [
            line.strip().split("\t") for line in open(path_imglst, "r")
        ]  # img_idx, 0, img_path

        self.img_idx_to_path = {int(item[0]): item[-1] for item in items}
        # path_landmarks = os.path.join(root_dir, "landmarks.csv")
        # self.path_to_landmarks = _read_landmarks(path_landmarks)

        logging.info("loading recordio %s...", path_imgrec)
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
        logging.info("loading recordio %s done", path_imgrec)
        self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        img_idx = self.imgidx[index]
        s = self.imgrec.read_idx(img_idx)
        header, sample = mx.recordio.unpack_img(s, cv2.IMREAD_UNCHANGED)
        if self.transform is not None:
            sample = self.transform(sample)

        return (
            sample,
            header.label[0],
            img_idx,
            self.img_idx_to_path[img_idx],
        )

    def __len__(self):
        return len(self.imgidx)


if __name__ == "__main__":
    actions = ("age", "gender")
    root_dir = "/mnt/trainingdb0/data/face-recognition/internal.face-verification/v5.1/multilabel_5_1_cache_2/test/"
    ds = MXFaceDataset(root_dir)

    models = {}
    if "emotion" in actions:
        models["emotion"] = build_model("Emotion")

    if "age" in actions:
        models["age"] = build_model("Age")

    if "gender" in actions:
        models["gender"] = build_model("Gender")

    if "race" in actions:
        models["race"] = build_model("Race")
    res = []

    for i in tqdm(range(len(ds))):
        image, id_label, img_idx, path = ds[i]
        image = cv2.resize(image, (224, 224))
        image = KPI.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image /= 255.0
        age_predictions = models["age"].predict(image, verbose=0)[0, :]
        age = Age.findApparentAge(age_predictions)
        gender = models["gender"].predict(image, verbose=0)[0, :].argmax()
        res.append([img_idx, id_label, age, gender, path])

    # save to lst
    with open(
        "../results/multilabel_5_1_cache_2_test_deepface_age_gender_results.lst", "w"
    ) as f:
        for item in res:
            f.write("\t".join(map(str, item)) + "\n")
