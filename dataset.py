import cv2

import os
from torch.utils.data import Dataset
import config


class DatasetLoader(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.root_trainA = os.path.join(root_dir, config.TRAIN_A_FOLDER)
        self.root_trainB = os.path.join(root_dir, config.TRAIN_B_FOLDER)
        self.transform = transform

        self.trainA_images = os.listdir(self.root_trainA)
        self.trainB_images = os.listdir(self.root_trainB)
        self.trainA_len = len(self.trainA_images)
        self.trainB_len = len(self.trainB_images)
        self.length_dataset = max(self.trainA_len, self.trainB_len)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        trainA_img = self.trainA_images[index % self.trainA_len]
        trainB_img = self.trainB_images[index % self.trainB_len]

        trainA_img_path = os.path.join(self.root_trainA, trainA_img)
        trainB_img_path = os.path.join(self.root_trainB, trainB_img)

        # trainA_img = np.array(Image.open(trainA_img_path).convert("RGB"))
        # trainB_img = np.array(Image.open(trainB_img_path).convert("RGB"))

        trainA_img = cv2.cvtColor(cv2.imread(trainA_img_path), cv2.COLOR_BGR2RGB)
        trainB_img = cv2.cvtColor(cv2.imread(trainB_img_path), cv2.COLOR_BGR2RGB)

        if self.transform:
            augmentations = self.transform(image=trainA_img, image0=trainB_img)
            trainA_img = augmentations["image"]
            trainB_img = augmentations["image0"]

        return trainA_img, trainB_img


def test():
    loader = DatasetLoader("./data/horse2zebra")
    print(len(loader))
    img1, img2 = loader[5]
    cv2.imshow("test", img1)
    cv2.imshow("test2", img2)
    cv2.waitKey()


if __name__ == '__main__':
    test()



