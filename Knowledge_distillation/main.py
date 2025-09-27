import argparse
from train import ImageDistillTrainer, BasicTrainer
from data_utils import get_dataset, get_processor, label2id_and_id2label, collate_fn,preprocess_data
from model_utils import get_teacher_model, get_student_model, load_metrics
from transformers import TrainingArguments
import numpy as np


def main(args):
    # Load dataset
    dataset = get_dataset()
    # Get ID and Label mappings
    id2label, label2id = label2id_and_id2label(dataset["train"])

    # Load processor
    processor = get_processor(args.processor_name)

    # Preprocess the dataset
    transformed_dataset = dataset.map(lambda x: preprocess_data(x, processor, label2id))
    transformed_dataset.set_format(
        "pt", columns=["pixel_values"], output_all_columns=True
    )

    # Split the dataset
    train_test_dataset = transformed_dataset["train"].train_test_split(test_size=0.2)
    train_val_dataset = train_test_dataset["train"].train_test_split(
        test_size=(0.1 / 0.8)
    )

    train_test_valid_dataset = {
        "train": train_val_dataset["train"],
        "valid": train_val_dataset["test"],
        "test": train_test_dataset["test"],
    }

    # Initialize models
    teacher_model = get_teacher_model(args.teacher_model_name, id2label, label2id)
    student_model = get_student_model(args.student_model_name, id2label, label2id)

    # Load metrics
    accuracy_metric = load_metrics("accuracy")

    # Define compute_metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        acc = accuracy_metric.compute(
            references=labels, predictions=np.argmax(predictions, axis=1)#The argmax function finds the index (Class ID) of the highest score for each sample
        )
        return {"accuracy": acc["accuracy"]}

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        push_to_hub=True,
        load_best_model_at_end=True,
    )

    # Initialize and train the trainer
    if args.distillation:
        trainer = ImageDistillTrainer(
            student_model=student_model,
            teacher_model=teacher_model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=train_test_valid_dataset["train"],
            eval_dataset=train_test_valid_dataset["test"],
            tokenizer=processor,
            temperature=args.temperature,
            lambda_param=args.lambda_param,
        )
    else:
        trainer = BasicTrainer(
            model=student_model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=train_test_valid_dataset["train"],
            eval_dataset=train_test_valid_dataset["test"],
            tokenizer=processor,
        )

    trainer.train()

    # Evaluate the model
    trainer.evaluate(train_test_valid_dataset["test"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation with Vision Transformers"
    )
    parser.add_argument(
        "--teacher_model_name",
        type=str,
        default="asusevski/vit-base-patch16-224-oxford-pets",
    )
    parser.add_argument(
        "--student_model_name", type=str, default="WinKawaks/vit-tiny-patch16-224"
    )
    parser.add_argument(
        "--processor_name", type=str, default="google/vit-base-patch16-224"
    )
    parser.add_argument("--output_dir", type=str, default="oxford-pets-vit-with-kd")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--distillation", type=bool, default=True)
    parser.add_argument("--temperature", type=int, default=5)
    parser.add_argument("--lambda_param", type=float, default=0.9)

    args = parser.parse_args()
    main(args)
