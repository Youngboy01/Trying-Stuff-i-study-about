import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


class ImageDistillTrainer(Trainer):
    """The main goal of this class is to train student model to mimic the soft targets of larger pretrained teacher model"""

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

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # moving teacher model to same device as student model
        self.teacher.to(device)
        self.teacher.eval()  # since we dont want to train the teacher model we freeze its weights, disable dropout and normalization

        self.loss_function = nn.KLDivLoss(reduction="batchmean")  # KLD loss
        self.temperature = temperature
        self.lambda_param = lambda_param  # overall loss is weighted sum of KLD loss and CEloss so lmbda controls the weights

    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        student_output = model(**inputs)

        with torch.no_grad():
            teacher_output = self.teacher(**inputs)

        # Computing soft targets for teacher and student
        soft_teacher = F.softmax(teacher_output.logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(
            student_output.logits / self.temperature, dim=-1
        )  # log softmax cause nn.KLDloss expects input in log probability space

        distillation_loss = (
            self.loss_function(soft_student, soft_teacher) * (self.temperature**2)
        )  # The T**2 scaling is essential because when the temperature T is high, the gradients produced by the soft targets are smaller. Scaling by T**2 ensures the distillation loss does not become insignificant compared to the standard loss.

        student_target_loss = student_output.loss  # this computes the CEloss

        # Calculating the final loss as a weighted sum of the two loss components
        loss = (
            1.0 - self.lambda_param
        ) * student_target_loss + self.lambda_param * distillation_loss
        return (loss, student_output) if return_outputs else loss


class BasicTrainer(Trainer):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
