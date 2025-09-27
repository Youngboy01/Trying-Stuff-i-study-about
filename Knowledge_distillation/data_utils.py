from datasets import load_dataset
from transformers import ViTImageProcessor
import torch


def get_dataset():
    return load_dataset("pcuenq/oxford-pets")


def get_processor(processor_name):
    return ViTImageProcessor.from_pretrained(processor_name)

def preprocess_data(example, processor, label2id):
    # Convert to RGB 
    example['image'] = example['image'].convert('RGB')

    # pass image into processor
    inputs = processor(example['image'], return_tensors='pt')

    # Add to example (pixel_values) and encode the label
    example['pixel_values'] = inputs['pixel_values'].squeeze()
    example['label'] = label2id[example['label']]
    return example

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
