"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import logging
import os
from math import prod
from pathlib import Path
from functools import partial
import random
from typing import Dict, Tuple, Callable
from PIL import Image, UnidentifiedImageError
from typing import List, Optional

import torch
import pypdf
import orjson
from torch.utils.data import Dataset
from transformers.modeling_utils import PreTrainedModel


class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset for processing a list of images using a preparation function.

    This dataset takes a list of image paths and applies a preparation function to each image.

    Args:
        img_list (list): List of image paths.
        prepare (Callable): A preparation function to process the images.

    Attributes:
        img_list (list): List of image paths.
        prepare (Callable): The preparation function.
    """

    def __init__(self, img_list, prepare: Callable):
        super().__init__()
        self.img_list = img_list
        self.prepare = prepare

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def ignore_none_collate(batch):
        if batch is None:
            return
        try:
            batch = [x for x in batch if x is not None and x[0] is not None]
            if len(batch) == 0:
                return
            return torch.utils.data.dataloader.default_collate(batch)
        except AttributeError:
            pass

    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_list[idx])
            return self.prepare(img)
        except Exception as e:
            logging.error(e)

class tikzDataset(Dataset):
    """
    Args:
        dataset_path: the path to the jsonl file
    """

    def __init__(
        self,
        dataset_path: str,
        tikz_model: PreTrainedModel,
        max_length: int = 64,
        split: str = "train",
    ):
        super().__init__()
        self.tikz_model = tikz_model
        self.max_length = max_length
        self.split = split
        self.dataset_path = dataset_path
        
        self.image_paths = list(sorted(os.listdir(dataset_path)))
        self.image_paths.remove('ground_truth.txt')


        self.ground_truths = []
        with open(os.path.join(dataset_path, 'ground_truth.txt')) as f:
            for line in f:
                self.ground_truths.append(line.strip())
        self.dataset_length = len(self.ground_truths)
        print(self.dataset_length)
        print(len(self.image_paths))

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels.
        Convert gt data into input_ids (tokenized string)

        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
        """
        img_path = os.path.join(self.dataset_path, self.image_paths[idx])
        img = Image.open(img_path)
        gt = self.ground_truths[idx]
        sample = {"image": img, "ground_truth": gt}

        if sample is None:
            # if sample is broken choose another randomly
            return self[random.randint(0, self.dataset_length - 1)]
        
        if sample is None or sample["image"] is None or prod(sample["image"].size) == 0:
            input_tensor = None
        else:
            input_tensor = self.tikz_model.encoder.prepare_input(
                sample["image"], random_padding=self.split == "train"
            )

        tokenizer_out = self.tikz_model.decoder.tokenizer(
            [sample["ground_truth"]], # tokenizer expects batch
        )
        input_ids = tokenizer_out["input_ids"].squeeze(0)
        attention_mask = tokenizer_out["attention_mask"].squeeze(0)

        # randomly perturb ground truth tokens
        # if self.split == "train" and self.perturb:
        #     # check if we perturb tokens
        #     unpadded_length = attention_mask.sum()
        #     while random.random() < 0.1:
        #         try:
        #             pos = random.randint(1, unpadded_length - 2)
        #             token = random.randint(
        #                 23, len(self.nougat_model.decoder.tokenizer) - 1
        #             )
        #             input_ids[pos] = token
        #         except ValueError:
        #             break
        return input_tensor, input_ids, attention_mask
    
    def __iter__(self):
        for i in range(self.dataset_length):
            yield self[i]