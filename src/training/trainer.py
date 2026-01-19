"""
Training Pipeline for Grid Compliance LLM Fine-tuning

Uses Unsloth for efficient LoRA/QLoRA fine-tuning of Qwen 2.5 models.
Supports both local training and Hugging Face Hub integration.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "pipeline.db"
OUTPUT_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Training configuration defaults
DEFAULT_CONFIG = {
    # Model settings
    "model_name": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",  # 4-bit quantized for efficiency
    "max_seq_length": 2048,
    "load_in_4bit": True,
    
    # LoRA settings
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    
    # Training settings
    "num_epochs": 3,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 10,
    "weight_decay": 0.01,
    "fp16": True,
    "logging_steps": 10,
    "save_steps": 100,
    
    # Output settings
    "output_dir": str(OUTPUT_DIR / "qwen-grid-compliance"),
    "push_to_hub": False,
    "hub_model_id": None,
}


def load_training_data_from_db():
    """Load training data directly from the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT question, answer, tier, source_file, page_number
        FROM training_dataset
        WHERE question != 'N/A' AND answer != 'N/A'
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        raise ValueError("No training data found in database. Run qa_generator first.")
    
    print(f"Loaded {len(rows)} training examples from database")
    return rows


def load_training_data_from_jsonl(path: Optional[str] = None):
    """Load training data from JSONL file."""
    if path is None:
        path = DATA_DIR / "training_data.jsonl"
    else:
        path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Training data not found at {path}. Run qa_generator --export first.")
    
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"Loaded {len(data)} training examples from {path}")
    return data


def format_prompt(question: str, answer: str, system_prompt: str = None) -> str:
    """Format a single example into the chat template format."""
    if system_prompt is None:
        system_prompt = """You are a Senior Grid Compliance Engineer specializing in UK power grid regulations.
You provide precise, technically accurate answers about:
- ENA Engineering Recommendation G99 (generator connections)
- UKPN EDS 08-5050 (EV charging connections)
- SPEN EV Fleet guidance
Always cite specific sections, values, and requirements from these standards."""

    # Qwen chat template format
    formatted = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""
    
    return formatted


def prepare_dataset(data_source: str = "db") -> Dataset:
    """
    Prepare the dataset for training.
    
    Args:
        data_source: Either "db" to load from database or path to JSONL file
    """
    if data_source == "db":
        rows = load_training_data_from_db()
        formatted_data = []
        for question, answer, tier, source, page in rows:
            formatted_data.append({
                "text": format_prompt(question, answer),
                "question": question,
                "answer": answer,
                "tier": tier,
                "source": source,
                "page": page
            })
    else:
        raw_data = load_training_data_from_jsonl(data_source)
        formatted_data = []
        for item in raw_data:
            formatted_data.append({
                "text": format_prompt(item["instruction"], item["output"]),
                "question": item["instruction"],
                "answer": item["output"],
                "tier": item.get("metadata", {}).get("tier", "unknown"),
                "source": item.get("metadata", {}).get("source", "unknown"),
                "page": item.get("metadata", {}).get("page", 0)
            })
    
    dataset = Dataset.from_list(formatted_data)
    
    # Split into train/val (90/10)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"Training examples: {len(split['train'])}")
    print(f"Validation examples: {len(split['test'])}")
    
    return split


def train(config: dict = None, data_source: str = "db", use_wandb: bool = False):
    """
    Main training function using Unsloth for efficient fine-tuning.
    
    Args:
        config: Training configuration dictionary
        data_source: "db" or path to JSONL file
        use_wandb: Whether to use Weights & Biases for logging
    """
    # Merge with defaults
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        merged = DEFAULT_CONFIG.copy()
        merged.update(config)
        config = merged
    
    print("="*60)
    print("GRID COMPLIANCE LLM TRAINING")
    print("="*60)
    print(f"\nModel: {config['model_name']}")
    print(f"LoRA rank: {config['lora_r']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']} x {config['gradient_accumulation_steps']} (gradient accumulation)")
    print(f"Learning rate: {config['learning_rate']}")
    print("="*60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\nWARNING: No GPU detected. Training will be very slow!")
    
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError as e:
        print(f"\nError: Required packages not installed: {e}")
        print("Install with: pip install unsloth trl transformers")
        return None
    
    # Load model with Unsloth optimizations
    print("\nLoading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        dtype=None,  # Auto-detect
    )
    
    # Apply LoRA
    print("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        use_gradient_checkpointing="unsloth",  # Unsloth optimization
        random_state=42,
    )
    
    # Prepare dataset
    print("\nPreparing dataset...")
    dataset_split = prepare_dataset(data_source)
    
    # Setup training arguments
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        fp16=config["fp16"] and torch.cuda.is_available(),
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=config["save_steps"],
        load_best_model_at_end=True,
        report_to="wandb" if use_wandb else "none",
        run_name=f"grid-compliance-{datetime.now().strftime('%Y%m%d-%H%M')}",
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        args=training_args,
    )
    
    # Train
    print("\nStarting training...")
    print("-"*60)
    
    trainer.train()
    
    # Save the final model
    print("\nSaving model...")
    final_output = output_dir / "final"
    model.save_pretrained(str(final_output))
    tokenizer.save_pretrained(str(final_output))
    
    # Save training config
    config_path = output_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nModel saved to: {final_output}")
    print(f"Config saved to: {config_path}")
    
    # Push to hub if configured
    if config.get("push_to_hub") and config.get("hub_model_id"):
        print(f"\nPushing to Hugging Face Hub: {config['hub_model_id']}")
        model.push_to_hub(config["hub_model_id"])
        tokenizer.push_to_hub(config["hub_model_id"])
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return model, tokenizer


def inference_test(model_path: str = None, questions: list = None):
    """
    Test the fine-tuned model with sample questions.
    
    Args:
        model_path: Path to the saved model (uses default if None)
        questions: List of test questions (uses defaults if None)
    """
    if model_path is None:
        model_path = str(OUTPUT_DIR / "qwen-grid-compliance" / "final")
    
    if questions is None:
        questions = [
            "What is the maximum power rating for a Type A generator under G99?",
            "What are the PME earthing requirements for outdoor EV chargers?",
            "How can a fleet operator reduce their connection upgrade costs?",
        ]
    
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Error: unsloth not installed")
        return
    
    print(f"Loading model from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    print("\n" + "="*60)
    print("INFERENCE TEST")
    print("="*60)
    
    system_prompt = """You are a Senior Grid Compliance Engineer specializing in UK power grid regulations.
You provide precise, technically accurate answers about ENA G99, UKPN EDS, and SPEN guidelines."""
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Question {i}]: {question}")
        print("-"*40)
        
        prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant's response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        
        print(f"[Answer]: {response}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Grid Compliance LLM")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--test", action="store_true", help="Run inference test")
    parser.add_argument("--data-source", type=str, default="db", 
                        help="Data source: 'db' or path to JSONL")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Base model name (default: Qwen2.5-3B-Instruct)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HF Hub")
    parser.add_argument("--hub-model-id", type=str, default=None, help="HF Hub model ID")
    
    args = parser.parse_args()
    
    if args.train:
        config = DEFAULT_CONFIG.copy()
        if args.model_name:
            config["model_name"] = args.model_name
        config["num_epochs"] = args.epochs
        config["batch_size"] = args.batch_size
        config["learning_rate"] = args.lr
        config["lora_r"] = args.lora_r
        config["push_to_hub"] = args.push_to_hub
        config["hub_model_id"] = args.hub_model_id
        
        train(config=config, data_source=args.data_source, use_wandb=args.wandb)
    
    elif args.test:
        inference_test()
    
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python trainer.py --train")
        print("  python trainer.py --train --epochs 5 --wandb")
        print("  python trainer.py --test")
