from datasets import load_dataset
from transformers import ViTImageProcessor
import torch


def get_dataset():
    return load_dataset("pcuenq/oxford-pets")


def get_processor(processor_name):
    return ViTImageProcessor.from_pretrained(processor_name)


def label2id_and_id2label(train_dataset):
    sorted_labels = sorted(list(set(train_dataset["label"])))
    id2label = {i: str_label for i, str_label in enumerate(sorted_labels)}
    label2id = {str_label: i for i, str_label in enumerate(sorted_labels)}
    return id2label, label2id


def collate_fn(batch):
    return {
        "pixel_values": torch.cat(
            [x["pixel_values"].unsqueeze(dim=0) for x in batch], dim=0
        ),
        "labels": torch.tensor([x["label"] for x in batch]),
    }
