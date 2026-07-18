from __future__ import division

import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import numpy as np
import scipy.io as sio
import scipy.misc
from PIL import Image


def _positive_int_env(name, default):
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = int(default)
    return max(1, value)


def _bool_env(name, default):
    value = os.environ.get(name, default)
    return str(value).strip().lower() not in ("0", "false", "no", "off")


IO_WORKERS = _positive_int_env("WATERGAN_IO_WORKERS", 16)
LOG_EVERY = _positive_int_env("WATERGAN_LOG_EVERY", 10)
THROTTLE_DIAGNOSTICS = _bool_env("WATERGAN_THROTTLE_DIAGNOSTICS", "1")
_IO_EXECUTOR = None


def _as_uint8(array):
    array = np.asarray(array)
    array = np.squeeze(array)
    if array.dtype.kind == "f":
        values = np.nan_to_num(array)
        if values.size:
            minimum = float(values.min())
            maximum = float(values.max())
            if minimum >= -1.0 and maximum <= 1.0:
                if minimum < 0.0:
                    values = (values + 1.0) * 127.5
                else:
                    values = values * 255.0
        array = np.clip(values, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    return array


def imread(filename, flatten=False, mode=None):
    with Image.open(filename) as image:
        if flatten:
            image = image.convert("L")
        elif mode is not None:
            image = image.convert(mode)
        else:
            image = image.convert("RGB")
        return np.asarray(image).copy()


def imresize(array, size, interp="bilinear", mode=None):
    array = np.asarray(array)
    array = np.squeeze(array)
    if mode == "F":
        image = Image.fromarray(array.astype(np.float32), mode="F")
    else:
        image = Image.fromarray(_as_uint8(array))

    if isinstance(size, (int, float)):
        scale = size / 100.0 if isinstance(size, int) else float(size)
        new_size = (
            max(1, int(round(image.size[0] * scale))),
            max(1, int(round(image.size[1] * scale))),
        )
    else:
        # scipy.misc.imresize accepts (height, width[, channels]).
        new_size = (int(size[1]), int(size[0]))

    if interp == "nearest":
        resample = Image.NEAREST
    elif interp == "bicubic":
        resample = Image.BICUBIC
    else:
        resample = Image.BILINEAR
    return np.asarray(image.resize(new_size, resample))


def imsave(filename, array):
    Image.fromarray(_as_uint8(array)).save(filename)


def install_scipy_misc_compat():
    if not hasattr(scipy.misc, "imread"):
        scipy.misc.imread = imread
    if not hasattr(scipy.misc, "imresize"):
        scipy.misc.imresize = imresize
    if not hasattr(scipy.misc, "imsave"):
        scipy.misc.imsave = imsave


def depth_sort_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def list_depth_files(depth_dataset):
    root = os.path.join("./data", depth_dataset)
    files = []
    for pattern in ("*.mat", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        files.extend(glob(os.path.join(root, pattern)))
    return sorted(files, key=depth_sort_key)


def load_depth_array(filename):
    suffix = os.path.splitext(filename)[1].lower()
    if suffix == ".mat":
        data = sio.loadmat(filename)
        for key in ("depth", "dph", "D", "data"):
            if key in data:
                array = data[key]
                break
        else:
            keys = [key for key in data.keys() if not key.startswith("__")]
            if not keys:
                raise ValueError("No depth array found in MAT file: %s" % filename)
            array = data[keys[0]]
        array = np.asarray(array, dtype=np.float32)
    else:
        with Image.open(filename) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32)

    array = np.squeeze(array)
    if array.ndim == 3:
        array = array[:, :, 0]
    if array.size:
        maximum = float(np.nanmax(array))
        if maximum > 1.0:
            array = array / 255.0
    return np.nan_to_num(array).astype(np.float32)


def effective_train_batches(air_data, depth_data, config):
    train_limit = min(len(air_data), len(depth_data))
    if config.train_size != np.inf:
        train_limit = min(train_limit, int(config.train_size))
    return int(train_limit // config.batch_size)


def _get_executor():
    global _IO_EXECUTOR
    if IO_WORKERS <= 1:
        return None
    if _IO_EXECUTOR is None:
        _IO_EXECUTOR = ThreadPoolExecutor(max_workers=IO_WORKERS)
    return _IO_EXECUTOR


def _call_loader(task):
    loader, filename = task
    return loader(filename)


def parallel_map(loader, filenames):
    filenames = list(filenames)
    executor = _get_executor()
    if executor is None:
        return [loader(filename) for filename in filenames]
    return list(executor.map(loader, filenames))


def parallel_load_many(specs):
    lengths = []
    tasks = []
    for loader, filenames in specs:
        filenames = list(filenames)
        lengths.append(len(filenames))
        tasks.extend((loader, filename) for filename in filenames)

    executor = _get_executor()
    if executor is None:
        loaded = [_call_loader(task) for task in tasks]
    else:
        loaded = list(executor.map(_call_loader, tasks))

    result = []
    offset = 0
    for length in lengths:
        result.append(loaded[offset:offset + length])
        offset += length
    return result


def should_log(counter):
    if not THROTTLE_DIAGNOSTICS:
        return True
    return counter == 1 or counter % LOG_EVERY == 0


install_scipy_misc_compat()
