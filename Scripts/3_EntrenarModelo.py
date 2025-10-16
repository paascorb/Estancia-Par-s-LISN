from huggingface_hub import login
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, 
    BitsAndBytesConfig,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTConfig, SFTTrainer
import warnings

def main():
    warnings.filterwarnings("ignore")
    
    modelo = "LiquidAI/LFM2-1.2B-Extract"
    # # PRUEBA 1:
    # # Cargamos el csv y creamos el dataset
    dataset = crear_dataset()
    # # Cargamos el modelo y la configuracion Lora
    # model, tokenizer, peft_config = cargar_modelo(modelo)
    # # Cargamos los parametros de entrenamiento
    # params = parametros_entrenamiento("Modelos/")
    # # Entrenamos el modelo
    # trainer = entrenar_modelo(model, dataset, peft_config, tokenizer, params)
    # # Guardamos el modelo
    # model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    # model_to_save.save_pretrained("ProposicionadorES-LFM2-1.2B")

    # PRUEBA 2:
    trainer = SFTTrainer(
        model=modelo,
        args=SFTConfig(
            output_dir="Modelos/ProposicionadorES-LFM2-1.2B",
            chat_template_path="LiquidAI/LFM2-1.2B-Extract",
        ),
        train_dataset=load_from_disk("Datasets/ProposicionesES.hf"),
    )
    trainer.train()

def entrenar_modelo(model, dataset, peft_config, tokenizer, training_arguments):
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatear_dataset,
        args=training_arguments,
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)
    trainer.train()
    return trainer

def parametros_entrenamiento(outputDir):
    output_dir = outputDir
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 10
    optim = "paged_adamw_32bit"
    save_steps = 200
    logging_steps = 25
    learning_rate = 2e-5
    max_grad_norm = 0.3
    max_steps = 300
    warmup_ratio = 0.03
    lr_scheduler_type = "cosine"

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=True,
        tf32=True,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        disable_tqdm=False
    )

def cargar_modelo(Nombre):
    
    model_name = Nombre

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        use_cache=False,
        device_map="auto",
        trust_remote_code=True
    )

    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return model, tokenizer, peft_config


def crear_dataset():
    # Aquí creamos el dataset desde el csv que hemos generado en los pasos anteriores
    df = pd.read_csv("Datasets/ResultadoProposiciones.csv")
    custom_ds = pd.DataFrame()
    aux = []
    numero_turn = []
    for index, row in df.iterrows():
        aux.append([{"role": "user", "content": row.Respuesta},
                    {"role": "assistant", "content": row.Instruccion}])
        numero_turn.append(1)
    se = pd.Series(aux)
    df['messages'] = se.values
    se = pd.Series(numero_turn)
    df['num_turns'] = se.values
    custom_ds["messages"] = df["messages"]
    custom_ds["num_turns"] = df["num_turns"]

    Dataset.from_pandas(custom_ds).save_to_disk("Datasets/ProposicionesES.hf")

def formatear_dataset(texto):
    return f'{texto["Instruction"]}{texto["Output"]}'


if __name__ == "__main__":
    main()
