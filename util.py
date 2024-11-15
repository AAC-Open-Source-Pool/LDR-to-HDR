import os
import cv2
import numpy as np
import torch
from numpy.random import uniform


def process_path(directory, create=False):
    directory = os.path.expanduser(directory)
    directory = os.path.normpath(directory)
    directory = os.path.abspath(directory)
    if create:
        if not os.path.exists(directory):
            os.makedirs(directory)
    return directory


def split_path(directory):
    directory = process_path(directory)
    name, ext = os.path.splitext(os.path.basename(directory))
    return os.path.dirname(directory), name, ext


def compose(transforms):
    if not isinstance(transforms, list):
        raise ValueError("Expected a list of transforms")

    def composition(obj):
        for transform in transforms:
            obj = transform(obj)
        return obj

    return composition


def map_range(x, low=0, high=1):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)


def str2bool(x):
    if x is None or x.lower() in ['no', 'false', 'f', '0']:
        return False
    else:
        return True


def cv2torch(np_img):
    rgb = np_img[:, :, (2, 1, 0)]
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def torch2cv(t_img):
    return t_img.numpy().transpose(1, 2, 0)[:, :, (2, 1, 0)]


def resize(x, size):
    return cv2.resize(x, size)


class Exposure:
    def __init__(self, stops=0.0, gamma=1.0):
        self.stops = stops
        self.gamma = gamma

    def process(self, img):
        return np.clip(img * (2 ** self.stops), 0, 1) ** self.gamma


def tone_map(img, tmo_name, **kwargs):
    if tmo_name == 'exposure':
        tmo = Exposure(**kwargs)
        return tmo.process(img)
    else:
        raise ValueError(f"Unknown TMO name: {tmo_name}")


def random_tone_map(x):
    tmo = Exposure(randomize=True)
    return map_range(tmo(x))


def clamped_gaussian(mean, std, min_value, max_value):
    while True:
        ret = np.random.normal(mean, std)
        if ret > min_value and ret < max_value:
            break
        std *= 0.99
    return ret


def exponential_size(val):
    return val * (np.exp(-np.random.uniform())) / (np.exp(0) + 1)


def index_gauss(img, precision=None, crop_size=None, random_size=True, ratio=None, seed=None):
    np.random.seed(seed)
    dims = {'w': img.shape[1], 'h': img.shape[0]}
    if precision is None:
        precision = {'w': 1, 'h': 4}

    if crop_size is None:
        crop_size = {key: int(dims[key] / 4) for key in dims}

    if ratio is not None:
        ratio = max(ratio, 1e-4)
        if ratio > 1:
            if random_size:
                crop_size['h'] = int(max(crop_size['h'], exponential_size(dims['h'])))
            crop_size['w'] = int(np.round(crop_size['h'] * ratio))
        else:
            if random_size:
                crop_size['w'] = int(max(crop_size['w'], exponential_size(dims['w'])))
            crop_size['h'] = int(np.round(crop_size['w'] / ratio))
    else:
        if random_size:
            crop_size = {key: int(max(val, exponential_size(dims[key]))) for key, val in crop_size.items()}

    centers = {
        key: int(
            clamped_gaussian(
                dim / 2, crop_size[key] / precision[key], min(int(crop_size[key] / 2), dim), 
                max(int(dim - crop_size[key] / 2), 0)
            )
        )
        for key, dim in dims.items()
    }
    starts = {key: max(center - int(crop_size[key] / 2), 0) for key, center in centers.items()}
    ends = {key: start + crop_size[key] for key, start in starts.items()}
    return np.s_[starts['h']:ends['h'], starts['w']:ends['w'], :]


def slice_gauss(img, precision=None, crop_size=None, random_size=True, ratio=None, seed=None):
    return img[index_gauss(img, precision, crop_size, random_size, ratio)]


class DirectoryDataset:
    def __init__(self, data_root_path='hdr_data', data_extensions=['.hdr', '.exr'], load_fn=None, preprocess=None):
        data_root_path = process_path(data_root_path)
        self.file_list = []
        for root, _, fnames in sorted(os.walk(data_root_path)):
            for fname in fnames:
                if any(fname.lower().endswith(extension) for extension in data_extensions):
                    self.file_list.append(os.path.join(root, fname))
        if not self.file_list:
            raise RuntimeError(f"Could not find any files with extensions: {', '.join(data_extensions)} in {data_root_path}")

        self.preprocess = preprocess

    def __getitem__(self, index):
        dpoint = cv2.imread(self.file_list[index], flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
        if self.preprocess:
            dpoint = self.preprocess(dpoint)
        return dpoint

    def __len__(self):
        return len(self.file_list)
