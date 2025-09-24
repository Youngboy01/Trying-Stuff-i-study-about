import torch
import evaluate
from transformers import (
    AutoModelForImageClassification,
    ViTForImageClassification,
    ViTConfig,
)


def get_teacher_model(model_name, id2label, label2id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
    model.eval()
    return model


def get_student_model(model_name, id2label, label2id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ViTForImageClassification.from_pretrained(model_name).config
    config.id2label = id2label
    config.label2id = label2id
    config.num_labels = len(id2label)
    model = ViTForImageClassification(config).to(device)
    return model


def load_metrics(metric_name):
    return evaluate.load(metric_name)
