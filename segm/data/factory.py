# PATH : /home/AD.UNLV.EDU/bhattb3/segmenter_supervised/segm/data/factory.py

from segm.data.flame import FlameDataset
from segm.data.meta import MetaDataset
from segm.data.ade20k import ADE20KSegmentation
from segm.data.pascal_context import PascalContextDataset
from segm.data.cityscapes import CityscapesDataset

def create_dataset(kwargs):
    dataset_name = kwargs.get("dataset", "").lower()

    if dataset_name == "flame":
        return FlameDataset(
            image_size=kwargs.get("image_size", 512),
            crop_size=kwargs.get("crop_size", 512),
            split=kwargs.get("split", "train"),
            normalization=kwargs.get("normalization", "vit"),
            root=kwargs.get("root", "/home/AD.UNLV.EDU/bhattb3/Datasets/Flame"),
        )
        
    elif dataset_name == "meta":
        return MetaDataset(
            image_size=kwargs.get("image_size", 512),
            crop_size=kwargs.get("crop_size", 512),
            split=kwargs.get("split", "train"),
            normalization=kwargs.get("normalization", "vit"),
            root=kwargs.get("root", "/home/AD.UNLV.EDU/bhattb3/Datasets/Meta"),
        )    
        
        

    elif dataset_name == "ade20k":
        return ADE20KSegmentation(**kwargs)

    elif dataset_name == "pascal":
        return PascalContextDataset(**kwargs)

    elif dataset_name == "cityscapes":
        return CityscapesDataset(**kwargs)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

