import json
import pandas as pd
import os
import torch
import string
from datasets import load_dataset, Dataset, load_from_disk
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import warnings

def main():
    warnings.filterwarnings("ignore")
    
    modelo = "LiquidAI/LFM2-2.6B"
    # # # PRUEBA 1:
    # # # Cargamos el csv y creamos el dataset
    # dataset = crear_dataset()
    # # # Cargamos el modelo y la configuracion Lora
    # # model, tokenizer, peft_config = cargar_modelo(modelo)
    # # # Cargamos los parametros de entrenamiento
    # # params = parametros_entrenamiento("Modelos/")
    # # # Entrenamos el modelo
    # # trainer = entrenar_modelo(model, dataset, peft_config, tokenizer, params)
    # # # Guardamos el modelo
    # # model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    # # model_to_save.save_pretrained("ProposicionadorES-LFM2-1.2B")

    # # PRUEBA 2:
    trainer = SFTTrainer(
        model=modelo,
        args=SFTConfig(
            output_dir="Modelos/ProposicionadorES-LFM2-2.6B_2",
            chat_template_path="LiquidAI/LFM2-2.6B",
        ),
        train_dataset=load_from_disk("Datasets/ProposicionesES_2.hf"),
    )
    trainer.train()
    # modelo = "LiquidAI/LFM2-2.6B"

    # dataset = crear_dataset()
    # dataset = load_from_disk("Datasets/ProposicionesES_2_1.hf")
    # model, tokenizer, config = cargar_modelo(modelo)

    # params = parametros_entrenamiento("Modelos/ProposicionadorES-LFM2-2.6B")
    # trainer = entrenar_modelo(model, config, tokenizer, params, dataset)
    # trainer.train()

    # trainer.save_model('Modelos/ProposicionadorES-LFM2-2.6B')



def entrenar_modelo(model, peft_config, tokenizer, training_arguments, dataset):
    trainer = SFTTrainer(
        model=model.base_model.model, # the underlying Phi-3 model
        peft_config=peft_config,  # added to fix issue in TRL>=0.20
        processing_class=tokenizer,
        args=training_arguments,
        train_dataset=dataset,
    )

    return trainer

def parametros_entrenamiento(outputDir):
    sft_config = SFTConfig(
        ## GROUP 1: Memory usage
        # These arguments will squeeze the most out of your GPU's RAM
        # Checkpointing
        gradient_checkpointing=True,    # this saves a LOT of memory
        # Set this to avoid exceptions in newer versions of PyTorch
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        # Gradient Accumulation / Batch size
        # Actual batch (for updating) is same (1x) as micro-batch size
        gradient_accumulation_steps=1,  
        # The initial (micro) batch size to start off with
        per_device_train_batch_size=16, 
        # If batch size would cause OOM, halves its size until it works
        auto_find_batch_size=True,

        ## GROUP 2: Dataset-related
        max_length=64, # renamed in v0.20
        # Dataset
        # packing a dataset means no padding is needed
        packing=True,
        packing_strategy='wrapped', # added to approximate original packing behavior

        ## GROUP 3: These are typical training parameters
        num_train_epochs=10,
        learning_rate=3e-4,
        # Optimizer
        # 8-bit Adam optimizer - doesn't help much if you're using LoRA!
        optim='paged_adamw_8bit',       
        
        ## GROUP 4: Logging parameters
        logging_steps=10,
        output_dir=outputDir,
        report_to='none',

        # ensures bf16 (the new default) is only used when it is actually available
        bf16=torch.cuda.is_bf16_supported(including_emulation=False)
    )

    return sft_config

def cargar_modelo(Nombre):
    
    model_name = Nombre

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization_config, device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        # the rank of the adapter, the lower the fewer parameters you'll need to train
        r=8,                   
        lora_alpha=16, # multiplier, usually 2*r
        bias="none",           
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        # Newer models, such as Phi-3 at time of writing, may require 
        # manually setting target modules
        target_modules = [
            "q_proj", "v_proj", "fc1", "fc2", "linear",
            "gate_proj", "up_proj", "down_proj",
        ]
        )
    model = get_peft_model(model, config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer, config


def crear_dataset():
    # Aquí creamos el dataset desde el csv que hemos generado en los pasos anteriores
    f = open('Datasets/Proposiciones_2.json')
    data = json.load(f)
    resultado = []
    custom_ds = pd.DataFrame()
    for elem in data:
        resultado.append([{'role': 'user', 'content': elem["Instrucción"]},
                        {'role': 'assistant', 'content': "\n".join(elem["Proposiciones"])}])
    se = pd.Series(resultado)
    custom_ds['messages'] = se.values
    Dataset.from_pandas(custom_ds).save_to_disk("Datasets/ProposicionesES_2_1.hf")

# def formatear_dataset(texto):
#     return f'{texto["Instruction"]}{texto["Output"]}'


if __name__ == "__main__":
    main()
