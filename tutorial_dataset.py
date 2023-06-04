import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/record.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        filenames = [item['source'], item['source1'], item['target']]
        prompt = item['prompt']

        sources = []
        for i in range(2):
            source = cv2.imread('./training/' + filenames[i])
            # OpenCV reads images in BGR order.
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            # Normalize source images to [0, 1].
            source = source.astype(np.float32) / 255.0
            sources.append(source)

        target = cv2.imread('./training/' + filenames[2])
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint1=sources[0], hint2=sources[1])
