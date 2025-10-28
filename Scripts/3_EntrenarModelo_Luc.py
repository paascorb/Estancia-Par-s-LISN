"""
Finetuning FLAN-T5 base pour extraire des propositions atomiques à
partir de passages.
Basé sur l'approche décrite dans l'article Chen 2024.
Version améliorée avec sauvegarde et évaluation périodiques.
Sortie étendue à 1024 tokens.
MODIFIÉ POUR UTILISER LE GPU 1
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
     T5ForConditionalGeneration,
     TFMT5ForConditionalGeneration,
     AutoTokenizer,
     get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from datasets import Dataset as HFDataset
import evaluate
from tqdm.auto import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# FORCER L'UTILISATION DU GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class PropositionDataset(Dataset):

    def __init__(self, data: List[Dict], tokenizer: AutoTokenizer, max_input_length: int = 512, max_target_length: int = 1024):
         self.data = data
         self.tokenizer = tokenizer
         self.max_input_length = max_input_length
         self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

         # Format d'entrée: "Décomposez le passage suivant en propositions atomiques: {passage}"
        input_text = f"Décomposez le passage suivant en propositions atomiques: {item['Texto']}"
        # Format de sortie: propositions séparées par des retours à la ligne
        target_text = "\n".join(item['Proposiciones'])

         # Tokenisation
        input_encoding = self.tokenizer(
             input_text,
             max_length=self.max_input_length,
             padding="max_length",
             truncation=True,
             return_tensors="pt"
         )

        target_encoding = self.tokenizer(
             target_text,
             max_length=self.max_target_length,
             padding="max_length",
             truncation=True,
             return_tensors="pt"
         )

         # Pour T5, remplacer les pad tokens par -100 dans les labels
        labels = target_encoding["input_ids"].flatten()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
             "input_ids": input_encoding["input_ids"].flatten(),
             "attention_mask": input_encoding["attention_mask"].flatten(),
             "labels": labels
         }

def load_dataset(file_path: str) -> List[Dict]:
     """Charge le dataset depuis le fichier JSON"""
     try:
         with open(file_path, 'r', encoding='utf-8') as f:
             data = json.load(f)

         # Validation basique des données
         if not isinstance(data, list):
             raise ValueError("Le dataset doit être une liste")

         valid_data = []
         for i, item in enumerate(data):
             if not isinstance(item, dict):
                 print(f"Attention: Item {i} n'est pas un dictionnaire, ignoré")
                 continue

             required_fields = ['Texto', 'Proposiciones']
             missing_fields = [field for field in required_fields if field not in item]

             if missing_fields:
                 print(f"Attention: Item {i} manque les champs {missing_fields}, ignoré")
                 continue

             if not item['Texto'].strip():
                 print(f"Attention: Item {i} a un texte vide, ignoré")
                 continue

             if not item['Proposiciones'] or not isinstance(item['Proposiciones'], list):
                 print(f"Attention: Item {i} n'a pas de propositions valides, ignoré")
                 continue

             valid_data.append(item)

         print(f"Dataset chargé: {len(valid_data)} exemples valides sur {len(data)} total")
         return valid_data

     except FileNotFoundError:
         raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")
     except json.JSONDecodeError as e:
         raise ValueError(f"Erreur de parsing JSON: {e}")
     except Exception as e:
         raise Exception(f"Erreur lors du chargement du dataset: {e}")

def compute_f1_score(pred_propositions: List[str], true_propositions: List[str]) -> float:
     """
     Calcule le score F1 entre deux ensembles de propositions.
     Basé sur l'approche BertScore mais simplifié pour les propositions.
     """
     if not pred_propositions or not true_propositions:
         return 0.0

     # Normalisation simple des propositions (minuscules, suppression espaces)
     pred_set = set(prop.lower().strip() for prop in pred_propositions)
     true_set = set(prop.lower().strip() for prop in true_propositions)

     if not pred_set or not true_set:
         return 0.0

     # Calcul de l'intersection
     intersection = pred_set.intersection(true_set)

     precision = len(intersection) / len(pred_set)
     recall = len(intersection) / len(true_set)

     if precision + recall == 0:
         return 0.0

     f1 = 2 * (precision * recall) / (precision + recall)
     return f1

def evaluate_model(model, tokenizer, eval_dataloader, device, max_length=1024):
     """Évalue le modèle sur l'ensemble de validation avec longueur de sortie étendue"""
     model.eval()
     total_f1 = 0
     num_examples = 0

     with torch.no_grad():
         for batch in tqdm(eval_dataloader, desc="Évaluation", leave=False):
             input_ids = batch["input_ids"].to(device)
             attention_mask = batch["attention_mask"].to(device)

             # Génération des propositions avec longueur étendue
             outputs = model.generate(
                 input_ids=input_ids,
                 attention_mask=attention_mask,
                 max_length=max_length,
                 max_new_tokens=max_length,
                 num_beams=4,
                 early_stopping=True,
                 pad_token_id=tokenizer.pad_token_id,
                 do_sample=False  # Pour des résultats reproductibles
             )

             # Décodage des prédictions et des vraies valeurs
             predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
             # Filtrer les tokens -100 des labels avant le décodage
             labels_filtered = batch["labels"].clone()
             labels_filtered[labels_filtered == -100] = tokenizer.pad_token_id
             targets = tokenizer.batch_decode(labels_filtered, skip_special_tokens=True)

             # Calcul du F1 pour chaque exemple
             for pred, target in zip(predictions, targets):
                 pred_props = [p.strip() for p in pred.split('\n') if p.strip()]
                 true_props = [p.strip() for p in target.split('\n') if p.strip()]

                 f1 = compute_f1_score(pred_props, true_props)
                 total_f1 += f1
                 num_examples += 1

     avg_f1 = total_f1 / num_examples if num_examples > 0 else 0
     return avg_f1

def save_checkpoint(model, tokenizer, optimizer, scheduler, step, loss, output_dir):
     """Sauvegarde un checkpoint du modèle"""
     if step in [6000, 12000]:
        checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarde du modèle et tokenizer
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        # Sauvegarde des états de l'optimiseur et scheduler
        checkpoint_data = {
            'step': step,
            'loss': loss,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint_data, checkpoint_dir / 'training_state.pt')

        print(f"Checkpoint sauvegardé: {checkpoint_dir}")

def save_metrics_and_plots(train_losses, eval_f1s, eval_steps, output_dir):
     """Sauvegarde les métriques et génère les courbes"""
     metrics_dir = Path(output_dir) / "metrics"
     metrics_dir.mkdir(parents=True, exist_ok=True)

     # Sauvegarde des métriques en JSON
     metrics_data = {
         'train_losses': train_losses,
         'eval_f1s': eval_f1s,
         'eval_steps': eval_steps
     }
     with open(metrics_dir / 'training_metrics.json', 'w') as f:
         json.dump(metrics_data, f, indent=2)

     # Configuration du style des graphiques
     plt.style.use('seaborn-v0_8')
     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

     # Graphique 1: Loss d'entraînement
     if train_losses:
         steps = list(range(1, len(train_losses) + 1))
         ax1.plot(steps, train_losses, 'b-', linewidth=2, alpha=0.8)
         ax1.set_xlabel('Itérations')
         ax1.set_ylabel('Loss d\'entraînement')
         ax1.set_title('Évolution de la Loss d\'entraînement', fontsize=14, fontweight='bold')
         ax1.grid(True, alpha=0.3)
         ax1.set_ylim(bottom=0)

     # Graphique 2: Score F1 de validation
     if eval_f1s and eval_steps:
         ax2.plot(eval_steps, eval_f1s, 'r-o', linewidth=2, markersize=6, alpha=0.8)
         ax2.set_xlabel('Itérations')
         ax2.set_ylabel('Score F1 de validation')
         ax2.set_title('Évolution du Score F1 de validation', fontsize=14, fontweight='bold')
         ax2.grid(True, alpha=0.3)
         ax2.set_ylim(0, 1)

         # Ajout des valeurs sur les points
         for x, y in zip(eval_steps, eval_f1s):
             ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                         xytext=(0,10), ha='center', fontsize=8)

     plt.tight_layout()
     plt.savefig(metrics_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
     plt.savefig(metrics_dir / 'training_curves.pdf', bbox_inches='tight')
     plt.close()

     # Graphique séparé pour la loss avec plus de détails
     if train_losses:
         plt.figure(figsize=(12, 6))
         steps = list(range(1, len(train_losses) + 1))
         plt.plot(steps, train_losses, 'b-', linewidth=1.5, alpha=0.8)
         plt.xlabel('Itérations')
         plt.ylabel('Loss d\'entraînement')
         plt.title('Courbe de Loss détaillée', fontsize=14, fontweight='bold')
         plt.grid(True, alpha=0.3)

         # Moyenne mobile sur 100 points si suffisamment de données
         if len(train_losses) > 100:
             window_size = min(100, len(train_losses) // 10)
             moving_avg = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
             moving_steps = steps[window_size-1:]
             plt.plot(moving_steps, moving_avg, 'r-', linewidth=2, alpha=0.9,
                     label=f'Moyenne mobile ({window_size} points)')
             plt.legend()

         plt.tight_layout()
         plt.savefig(metrics_dir / 'loss_curve_detailed.png', dpi=300, bbox_inches='tight')
         plt.close()

     print(f"Métriques et graphiques sauvegardés dans: {metrics_dir}")

def main():
     parser = argparse.ArgumentParser(description="Finetuner FLAN-T5 pour l'extraction de propositions")
     parser.add_argument("--data_path", type=str, default="Datasets/Proposiciones_2.json",
                        help="Chemin vers le fichier de données")
     parser.add_argument("--model_name", type=str, default="Modelos/ProposicionadorES-T5-large",
                        help="Nom du modèle pré-entraîné")
     parser.add_argument("--output_dir", type=str, default="Modelos/ProposicionadorES-T5-large",
                        help="Répertoire de sortie pour sauvegarder le modèle")
     parser.add_argument("--batch_size", type=int, default=2,
                        help="Taille de batch (ajustez selon votre GPU)")
     parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Taux d'apprentissage")
     parser.add_argument("--num_epochs", type=int, default=3,
                        help="Nombre d'époques")
     parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Decay des poids")
     parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Nombre d'étapes de warmup")
     parser.add_argument("--max_input_length", type=int, default=512,
                        help="Longueur maximale des séquences d'entrée")
     parser.add_argument("--max_target_length", type=int, default=1024,
                        help="Longueur maximale des séquences de sortie (étendue à 1024)")
     parser.add_argument("--test_size", type=float, default=0.1,
                        help="Proportion des données pour la validation")
     parser.add_argument("--save_steps", type=int, default=2000,
                        help="Nombre d'étapes entre chaque sauvegarde")
     parser.add_argument("--eval_steps", type=int, default=15000,
                        help="Nombre d'étapes entre chaque évaluation")

     args = parser.parse_args()

     # Configuration du device - MODIFIÉ POUR FORCER LE GPU 1
     print("Configuration des devices...")
     print(f"CUDA_VISIBLE_DEVICES défini sur: {os.environ.get('CUDA_VISIBLE_DEVICES', 'non défini')}")

     if torch.cuda.is_available():
         device = torch.device("cuda:0")  # Sera mappé au GPU 1 physique grâce à CUDA_VISIBLE_DEVICES
         print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
         print(f"Mémoire GPU totale: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
         print(f"Utilisation du device: {device} (GPU 1 physique)")
     elif torch.backends.mps.is_available():
         device = torch.device("mps")
         print(f"Utilisation du device MPS (Apple Silicon): {device}")
     else:
         device = torch.device("cpu")
         print(f"Utilisation du device CPU: {device}")

     # Chargement des données
     print("Chargement des données...")
     data = load_dataset(args.data_path)
     print(f"Nombre total d'exemples: {len(data)}")

     # Division train/validation
     train_data, val_data = train_test_split(data, test_size=args.test_size, random_state=42)
     print(f"Données d'entraînement: {len(train_data)}")
     print(f"Données de validation: {len(val_data)}")

     # Chargement du modèle et tokenizer
     print(f"Chargement du modèle {args.model_name}...")
     tokenizer = AutoTokenizer.from_pretrained(args.model_name)
     model = T5ForConditionalGeneration.from_pretrained(args.model_name) # T5
    #  model = TFMT5ForConditionalGeneration.from_pretrained(args.model_name) # mT5

     # Extension de la longueur de contexte pour le tokenizer
     tokenizer.model_max_length = 2048
     print(f"Longueur de contexte du tokenizer étendue à: {tokenizer.model_max_length}")
     print(f"Longueur maximale de sortie configurée à: {args.max_target_length}")

     # Pour MPS, certaines opérations peuvent nécessiter float32
     if device.type == "mps":
         model = model.to(torch.float32)
     model.to(device)

     # Vérification que le modèle est bien sur le bon GPU
     if torch.cuda.is_available():
         print(f"Modèle chargé sur le device: {next(model.parameters()).device}")
         print(f"Mémoire GPU utilisée après chargement: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

     # Création des datasets
     train_dataset = PropositionDataset(train_data, tokenizer, args.max_input_length, args.max_target_length)
     val_dataset = PropositionDataset(val_data, tokenizer, args.max_input_length, args.max_target_length)

     # Création des dataloaders
     train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
     val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

     # Configuration de l'optimiseur
     optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

     # Calcul du nombre total d'étapes
     total_steps = len(train_dataloader) * args.num_epochs
     scheduler = get_linear_schedule_with_warmup(
         optimizer,
         num_warmup_steps=args.warmup_steps,
         num_training_steps=total_steps
     )

     print(f"Nombre total d'étapes: {total_steps}")
     print(f"Sauvegarde toutes les {args.save_steps} étapes")
     print(f"Évaluation toutes les {args.eval_steps} étapes")
     print(f"Taille de batch maintenue à: {args.batch_size}")

     # Listes pour stocker les métriques
     train_losses = []
     eval_f1s = []
     eval_steps = []

     # Entraînement
     print("Début de l'entraînement...")
     model.train()

     global_step = 0

     for epoch in range(args.num_epochs):
         print(f"\nÉpoque {epoch + 1}/{args.num_epochs}")

         epoch_losses = []
         progress_bar = tqdm(train_dataloader, desc=f"Époque {epoch + 1}")

         for batch in progress_bar:
             global_step += 1
             optimizer.zero_grad()

             input_ids = batch["input_ids"].to(device)
             attention_mask = batch["attention_mask"].to(device)
             labels = batch["labels"].to(device)

             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
             loss = outputs.loss

             loss.backward()
             optimizer.step()
             scheduler.step()

             # Stockage de la loss
             current_loss = loss.item()
             train_losses.append(current_loss)
             epoch_losses.append(current_loss)

             progress_bar.set_postfix({
                 "loss": f"{current_loss:.4f}",
                 "step": global_step,
                 "gpu_mem": f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A"
             })

             # Évaluation périodique
             if global_step % args.eval_steps == 0:
                 print(f"\nÉvaluation à l'étape {global_step}...")
                 val_f1 = evaluate_model(model, tokenizer, val_dataloader, device, args.max_target_length)
                 eval_f1s.append(val_f1)
                 eval_steps.append(global_step)
                 print(f"Score F1 de validation: {val_f1:.4f}")
                 model.train()  # Retour en mode entraînement

             # Sauvegarde périodique
             if global_step % args.save_steps == 0:
                 save_checkpoint(model, tokenizer, optimizer, scheduler,
                               global_step, current_loss, args.output_dir)

         avg_epoch_loss = np.mean(epoch_losses)
         print(f"Perte moyenne de l'époque {epoch + 1}: {avg_epoch_loss:.4f}")

     # Évaluation finale
     print("\nÉvaluation finale...")
     final_f1 = evaluate_model(model, tokenizer, val_dataloader, device, args.max_target_length)
     eval_f1s.append(final_f1)
     eval_steps.append(global_step)
     print(f"Score F1 final: {final_f1:.4f}")

     # Sauvegarde finale du modèle
     print(f"\nSauvegarde finale du modèle dans {args.output_dir}...")
     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
     model.save_pretrained(args.output_dir)
     tokenizer.save_pretrained(args.output_dir)

     # Sauvegarde de la configuration étendue
     config_data = {
         'model_name': args.model_name,
         'max_input_length': args.max_input_length,
         'max_target_length': args.max_target_length,
         'tokenizer_max_length': tokenizer.model_max_length,
         'gpu_used': str(device),
         'training_parameters': {
             'batch_size': args.batch_size,
             'learning_rate': args.learning_rate,
             'num_epochs': args.num_epochs,
             'weight_decay': args.weight_decay,
             'warmup_steps': args.warmup_steps
         }
     }
     with open(Path(args.output_dir) / 'training_config.json', 'w') as f:
         json.dump(config_data, f, indent=2)

     # Sauvegarde des métriques et génération des graphiques
     save_metrics_and_plots(train_losses, eval_f1s, eval_steps, args.output_dir)

     # Résumé final
     print("\n" + "="*60)
     print("RÉSUMÉ DE L'ENTRAÎNEMENT - GPU 1 - SORTIE 1024 TOKENS")
     print("="*60)
     print(f"GPU utilisé: {device}")
     if torch.cuda.is_available():
         print(f"Nom du GPU: {torch.cuda.get_device_name(0)}")
         print(f"Mémoire finale utilisée: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
     print(f"Nombre total d'étapes: {global_step}")
     print(f"Loss initiale: {train_losses[0]:.4f}")
     print(f"Loss finale: {train_losses[-1]:.4f}")
     print(f"Score F1 final: {final_f1:.4f}")
     if len(eval_f1s) > 1:
         print(f"Meilleur score F1: {max(eval_f1s):.4f}")
         best_step = eval_steps[eval_f1s.index(max(eval_f1s))]
         print(f"Atteint à l'étape: {best_step}")
     print(f"Longueur maximale d'entrée: {args.max_input_length}")
     print(f"Longueur maximale de sortie: {args.max_target_length}")
     print(f"Taille de batch utilisée: {args.batch_size}")
     print(f"Modèle sauvegardé dans: {args.output_dir}")
     print("="*60)

     print("Entraînement terminé sur GPU 1 avec sortie étendue à 1024 tokens!")

if __name__ == "__main__":
     main()
