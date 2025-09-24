import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


class ImageDistilTrainer(Trainer):
    def __init__(
        self,
        teacher_model,
        student_model,
        temperature=5,
        lambda_param=0.9,
        *args,
        **kwargs,
    ):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model

        # Move the teacher model to the correct device and set to eval mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher.to(device)
        self.teacher.eval()

        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        self.temperature = temperature
        self.lambda_param = lambda_param

    def compute_loss(self, model, inputs, return_outputs=False):
        # We pass the inputs to both student and teacher models
        student_output = model(**inputs)

        with torch.no_grad():
            teacher_output = self.teacher(**inputs)

        # Compute soft targets for teacher and student
        # The teacher's distribution is normalized using softmax
        soft_teacher = F.softmax(teacher_output.logits / self.temperature, dim=-1)
        # The student's distribution is normalized using log_softmax
        soft_student = F.log_softmax(student_output.logits / self.temperature, dim=-1)

        # Compute the distillation loss (KL Divergence)
        # We scale the loss by the square of the temperature as recommended by Hinton et. al
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (
            self.temperature**2
        )

        # Compute the student's own loss on the true labels
        student_target_loss = student_output.loss

        # Calculate the final loss as a weighted sum of the two loss components
        loss = (
            1.0 - self.lambda_param
        ) * student_target_loss + self.lambda_param * distillation_loss
        return (loss, student_output) if return_outputs else loss


class BasicTrainer(Trainer):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
