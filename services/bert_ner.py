import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)

ENTITY_LABELS = [
    "O",
    "B-disease",
    "I-disease",
    "B-symptom",
    "I-symptom",
    "B-finding",
    "I-finding",
    "B-organ",
    "I-organ",
    "B-imaging_procedure",
    "I-imaging_procedure",
    "B-examination_procedure",
    "I-examination_procedure",
    "B-therapeutic_procedure",
    "I-therapeutic_procedure",
    "B-imaging_result",
    "I-imaging_result",
    "B-examination_measure",
    "I-examination_measure",
    "B-parameter",
    "I-parameter",
    "B-score",
    "I-score",
    "B-therapy",
    "I-therapy",
    "B-substance",
    "I-substance",
    "B-adverse_event",
    "I-adverse_event",
]

LABEL2ID = {label: idx for idx, label in enumerate(ENTITY_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(ENTITY_LABELS)}


class BERTNERService:
    """Service for BERT-based Named Entity Recognition."""
    
    def __init__(self, models_dir: str = "BERT_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.entity_models_dir = self.models_dir / "Entities"
        self.relation_models_dir = self.models_dir / "Relations"
        self.entity_models_dir.mkdir(parents=True, exist_ok=True)
        self.relation_models_dir.mkdir(parents=True, exist_ok=True)
        
        self.loaded_models: Dict[str, Any] = {}
        self.loaded_tokenizers: Dict[str, Any] = {}
        self.default_model: Optional[str] = None
        
        self._transformers = None
        self._torch = None
        self._device = None
        
    def _load_dependencies(self):
        if self._transformers is None:
            try:
                import transformers
                import torch
                self._transformers = transformers
                self._torch = torch
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"BERT NER service initialized. Device: {self._device}")
            except ImportError as e:
                raise ImportError(
                    "transformers and torch are required for BERT NER. "
                    "Install with: pip install transformers torch"
                ) from e
    
    def get_available_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        models = []
        
        dirs_to_scan = []
        if model_type is None or model_type == "entity":
            dirs_to_scan.append((self.entity_models_dir, "entity"))
        if model_type is None or model_type == "relation":
            dirs_to_scan.append((self.relation_models_dir, "relation"))
        
        for parent_dir, mtype in dirs_to_scan:
            if not parent_dir.exists():
                continue
            for model_dir in parent_dir.iterdir():
                if model_dir.is_dir():
                    config_path = model_dir / "config.json"
                    metadata_path = model_dir / "model_metadata.json"
                    
                    if config_path.exists():
                        model_labels = ENTITY_LABELS
                        try:
                            with open(config_path, "r", encoding="utf-8") as f:
                                config = json.load(f)
                                if "id2label" in config:
                                    labels = list(config["id2label"].values())
                                    model_labels = labels
                        except Exception:
                            pass
                        
                        model_info = {
                            "name": model_dir.name,
                            "path": str(model_dir),
                            "base_model": None,
                            "labels": model_labels,
                            "model_type": mtype,
                            "created_at": None
                        }
                        
                        if metadata_path.exists():
                            try:
                                with open(metadata_path, "r", encoding="utf-8") as f:
                                    metadata = json.load(f)
                                    model_info.update(metadata)
                                    model_info["model_type"] = mtype
                            except Exception:
                                pass
                        
                        models.append(model_info)
        
        return models
    
    def load_model(self, model_name: str) -> bool:
        self._load_dependencies()
        
        if model_name in self.loaded_models:
            return True
        
        model_path = self.entity_models_dir / model_name
        if not model_path.exists():
            model_path = self.relation_models_dir / model_name
        if not model_path.exists():
            model_path = self.models_dir / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model '{model_name}' not found in {self.models_dir}")
        
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            
            logger.info(f"Loading model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForTokenClassification.from_pretrained(str(model_path))
            model.to(self._device)
            model.eval()
            
            self.loaded_tokenizers[model_name] = tokenizer
            self.loaded_models[model_name] = model
            
            if self.default_model is None:
                self.default_model = model_name
            
            logger.info(f"Model '{model_name}' loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise
    
    def unload_model(self, model_name: str):
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            del self.loaded_tokenizers[model_name]
            
            if self.default_model == model_name:
                self.default_model = next(iter(self.loaded_models.keys()), None)
            
            if self._torch and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
    
    def classify(
        self,
        text: str,
        model_name: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], float]:
        self._load_dependencies()
        
        model_name = model_name or self.default_model
        if model_name is None:
            raise ValueError("No model available. Please load or train a model first.")
        
        if model_name not in self.loaded_models:
            self.load_model(model_name)
        
        model = self.loaded_models[model_name]
        tokenizer = self.loaded_tokenizers[model_name]
        
        model_id2label = getattr(model.config, 'id2label', None)
        if model_id2label:
            id2label = {int(k): v for k, v in model_id2label.items()}
            logger.debug(f"Using model's id2label with {len(id2label)} labels")
        else:
            id2label = ID2LABEL
            logger.debug("Using default ID2LABEL")
        
        start_time = time.perf_counter()
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with self._torch.no_grad():
            outputs = model(**inputs)
            predictions = self._torch.argmax(outputs.logits, dim=-1)[0].tolist()
        
        entities = self._extract_entities(text, predictions, offset_mapping, tokenizer, id2label)
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        return entities, inference_time
    
    def _extract_entities(
        self,
        text: str,
        predictions: List[int],
        offset_mapping: List[Tuple[int, int]],
        tokenizer,
        id2label: Dict[int, str] = None
    ) -> List[Dict[str, Any]]:
        if id2label is None:
            id2label = ID2LABEL
            
        entities = []
        current_entity = None
        prev_end = -1
        
        for idx, (pred_id, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            if start == end:
                prev_end = end
                continue
            
            label = id2label.get(pred_id, "O")
            
            is_continuation = (start == prev_end) or (start <= prev_end + 1 and text[prev_end:start].strip() == '')
            
            token_text = text[start:end]
            is_subword = False
            if hasattr(tokenizer, 'convert_ids_to_tokens'):
                try:
                    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)["input_ids"][0]
                    if idx < len(input_ids):
                        token_str = tokenizer.convert_ids_to_tokens([input_ids[idx].item()])[0]
                        is_subword = token_str.startswith('##') or token_str.startswith('Ġ')
                except:
                    pass
            
            if not is_subword and prev_end > 0 and start > prev_end:
                between_text = text[prev_end:start]
                is_subword = between_text == '' or not any(c.isspace() for c in between_text)
            
            if label == "O":
                if current_entity:
                    current_entity["text"] = text[current_entity["start_pos"]:current_entity["end_pos"]]
                    entities.append(current_entity)
                    current_entity = None
            elif label.startswith("B-"):
                entity_type = label[2:]
                
                if current_entity and current_entity["type"] == entity_type:
                    if is_continuation or is_subword:
                        current_entity["end_pos"] = end
                        current_entity["text"] = text[current_entity["start_pos"]:end]
                    else:
                        current_entity["text"] = text[current_entity["start_pos"]:current_entity["end_pos"]]
                        entities.append(current_entity)
                        current_entity = {
                            "text": text[start:end],
                            "type": entity_type,
                            "start_pos": start,
                            "end_pos": end,
                            "confidence": 0.9
                        }
                else:
                    if current_entity:
                        current_entity["text"] = text[current_entity["start_pos"]:current_entity["end_pos"]]
                        entities.append(current_entity)
                    
                    current_entity = {
                        "text": text[start:end],
                        "type": entity_type,
                        "start_pos": start,
                        "end_pos": end,
                        "confidence": 0.9
                    }
            elif label.startswith("I-"):
                entity_type = label[2:]
                if current_entity:
                    if current_entity["type"] == entity_type:
                        current_entity["end_pos"] = end
                        current_entity["text"] = text[current_entity["start_pos"]:end]
                    else:
                        current_entity["text"] = text[current_entity["start_pos"]:current_entity["end_pos"]]
                        entities.append(current_entity)
                        current_entity = {
                            "text": text[start:end],
                            "type": entity_type,
                            "start_pos": start,
                            "end_pos": end,
                            "confidence": 0.9
                        }
                else:
                    current_entity = {
                        "text": text[start:end],
                        "type": entity_type,
                        "start_pos": start,
                        "end_pos": end,
                        "confidence": 0.9
                    }
            
            prev_end = end
        
        if current_entity:
            current_entity["text"] = text[current_entity["start_pos"]:current_entity["end_pos"]]
            entities.append(current_entity)
        
        merged_entities = []
        for entity in entities:
            if merged_entities:
                prev = merged_entities[-1]
                if (prev["type"] == entity["type"] and 
                    entity["start_pos"] <= prev["end_pos"] + 2):
                    between = text[prev["end_pos"]:entity["start_pos"]]
                    if not between or between in ['-', '_', ''] or not any(c.isspace() for c in between):
                        prev["end_pos"] = entity["end_pos"]
                        prev["text"] = text[prev["start_pos"]:prev["end_pos"]]
                        continue
            merged_entities.append(entity)
        
        return merged_entities
    
    def train(
        self,
        model_name: str,
        base_model: str = "dmis-lab/biobert-base-cased-v1.1",
        training_data: Optional[str] = None,
        train_data: Optional[str] = None,
        dev_data: Optional[str] = None,
        test_data: Optional[str] = None,
        training_file: Optional[str] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        model_type: str = "entity"
    ) -> Dict[str, Any]:
        self._load_dependencies()
        
        if train_data:
            train_sentences, train_labels = self._parse_iob_data(train_data)
            
            if len(train_sentences) == 0:
                raise ValueError("No valid training sentences found in train data")
            
            logger.info(f"Train data: {len(train_sentences)} sentences")
            
            dev_sentences, dev_labels = None, None
            if dev_data:
                dev_sentences, dev_labels = self._parse_iob_data(dev_data)
                logger.info(f"Dev data: {len(dev_sentences)} sentences")
            
            test_sentences, test_labels = None, None
            if test_data:
                test_sentences, test_labels = self._parse_iob_data(test_data)
                logger.info(f"Test data: {len(test_sentences)} sentences")
            
        elif training_data or training_file:
            if training_file:
                with open(training_file, "r", encoding="utf-8") as f:
                    training_data = f.read()
            
            train_sentences, train_labels = self._parse_iob_data(training_data)
            dev_sentences, dev_labels = None, None
            test_sentences, test_labels = None, None
            
            if len(train_sentences) == 0:
                raise ValueError("No valid training sentences found in the data")
            
            logger.info(f"Training data: {len(train_sentences)} sentences")
        else:
            raise ValueError("Either train_data or training_data/training_file must be provided")
        
        from transformers import (
            AutoTokenizer,
            AutoModelForTokenClassification,
            TrainingArguments,
            Trainer,
            DataCollatorForTokenClassification
        )
        from datasets import Dataset
        import numpy as np
        
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForTokenClassification.from_pretrained(
            base_model,
            num_labels=len(ENTITY_LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
        model.to(self._device)
        
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples["tokens"],
                truncation=True,
                is_split_into_words=True,
                max_length=512
            )
            
            all_labels = []
            for i, label in enumerate(examples["ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                label_ids = []
                previous_word_idx = None
                
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                
                all_labels.append(label_ids)
            
            tokenized_inputs["labels"] = all_labels
            return tokenized_inputs
        
        train_dataset_dict = {
            "tokens": train_sentences,
            "ner_tags": train_labels
        }
        train_dataset = Dataset.from_dict(train_dataset_dict)
        tokenized_train_dataset = train_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        tokenized_eval_dataset = None
        if dev_sentences and dev_labels:
            eval_dataset_dict = {
                "tokens": dev_sentences,
                "ner_tags": dev_labels
            }
            eval_dataset = Dataset.from_dict(eval_dataset_dict)
            tokenized_eval_dataset = eval_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=eval_dataset.column_names
            )
        
        tokenized_test_dataset = None
        if test_sentences and test_labels:
            test_dataset_dict = {
                "tokens": test_sentences,
                "ner_tags": test_labels
            }
            test_dataset = Dataset.from_dict(test_dataset_dict)
            tokenized_test_dataset = test_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=test_dataset.column_names
            )
        
        def compute_metrics(p):
            predictions, labels_batch = p
            predictions = np.argmax(predictions, axis=2)
            
            true_predictions = [
                [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels_batch)
            ]
            true_labels = [
                [ID2LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels_batch)
            ]
            
            total = 0
            correct = 0
            for pred_seq, label_seq in zip(true_predictions, true_labels):
                for pred, label in zip(pred_seq, label_seq):
                    total += 1
                    if pred == label:
                        correct += 1
            
            accuracy = correct / total if total > 0 else 0
            
            return {
                "accuracy": accuracy,
                "total_tokens": total,
                "correct_tokens": correct
            }
        
        if model_type == "relation":
            output_dir = self.relation_models_dir / model_name
        else:
            output_dir = self.entity_models_dir / model_name
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            evaluation_strategy="epoch" if tokenized_eval_dataset else "no",
            logging_steps=10,
            remove_unused_columns=False,
            load_best_model_at_end=True if tokenized_eval_dataset else False,
            metric_for_best_model="accuracy" if tokenized_eval_dataset else None,
        )
        
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        
        logger.info(f"Starting training for model '{model_name}'...")
        train_result = trainer.train()
        
        test_metrics = None
        if tokenized_test_dataset:
            logger.info("Evaluating on test set...")
            test_results = trainer.evaluate(tokenized_test_dataset)
            test_metrics = {
                "test_loss": test_results.get("eval_loss"),
                "test_accuracy": test_results.get("eval_accuracy"),
                "test_total_tokens": test_results.get("eval_total_tokens"),
                "test_correct_tokens": test_results.get("eval_correct_tokens"),
            }
            logger.info(f"Test results: {test_metrics}")
        
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        
        metadata = {
            "name": model_name,
            "base_model": base_model,
            "model_type": model_type,
            "labels": ENTITY_LABELS,
            "created_at": datetime.now().isoformat(),
            "training_config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "train_samples": len(train_sentences),
                "dev_samples": len(dev_sentences) if dev_sentences else 0,
                "test_samples": len(test_sentences) if test_sentences else 0
            }
        }
        
        with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        self.load_model(model_name)
        
        training_metrics = {
            "loss": train_result.training_loss,
            "epochs": epochs,
            "train_samples": len(train_sentences),
            "dev_samples": len(dev_sentences) if dev_sentences else 0,
            "test_samples": len(test_sentences) if test_sentences else 0
        }
        
        if test_metrics:
            training_metrics.update(test_metrics)
        
        return {
            "success": True,
            "model_name": model_name,
            "model_path": str(output_dir),
            "training_metrics": training_metrics
        }
    
    def _parse_iob_data(
        self,
        iob_data: str
    ) -> Tuple[List[List[str]], List[List[int]]]:
        import csv
        from io import StringIO
        
        sentences = []
        labels = []
        
        lines = iob_data.strip().split("\n")
        
        first_line = lines[0].strip().lower() if lines else ""
        is_csv = first_line.startswith("words,") or ",sentence_id," in first_line
        
        if is_csv:
            sentence_dict = {}
            
            reader = csv.DictReader(StringIO(iob_data))
            for row in reader:
                word = row.get('words', '').strip()
                sentence_id = int(row.get('sentence_id', 0))
                label = row.get('labels', 'O').strip()
                
                if sentence_id not in sentence_dict:
                    sentence_dict[sentence_id] = ([], [])
                
                sentence_dict[sentence_id][0].append(word)
                label_normalized = self._normalize_label(label)
                label_id = LABEL2ID.get(label_normalized, 0)
                sentence_dict[sentence_id][1].append(label_id)
            
            for sid in sorted(sentence_dict.keys()):
                tokens, label_ids = sentence_dict[sid]
                if tokens:
                    sentences.append(tokens)
                    labels.append(label_ids)
        else:
            current_sentence = []
            current_labels = []
            
            for line in lines:
                line = line.strip()
                
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        label = parts[-1]
                        
                        current_sentence.append(token)
                        label_normalized = self._normalize_label(label)
                        label_id = LABEL2ID.get(label_normalized, 0)
                        current_labels.append(label_id)
            
            if current_sentence:
                sentences.append(current_sentence)
                labels.append(current_labels)
        
        return sentences, labels
    
    def _normalize_label(self, label: str) -> str:

        if label == 'O' or label == 'o':
            return 'O'
        
        label_check = label.upper()
        if label_check.startswith('B-') or label_check.startswith('I-'):
            prefix = label[:2].upper()
            entity_type = label[2:]
            
            import re
            entity_type_snake = re.sub(r'(?<!^)(?=[A-Z])', '_', entity_type).lower()
            
            type_mapping = {
                'disease': 'disease',
                'symptom': 'symptom',
                'finding': 'finding',
                'organ': 'organ',
                'imaging_procedure': 'imaging_procedure',
                'imagingprocedure': 'imaging_procedure',
                'examination_procedure': 'examination_procedure',
                'examinationprocedure': 'examination_procedure',
                'therapeutic_procedure': 'therapeutic_procedure',
                'therapeuticprocedure': 'therapeutic_procedure',
                'imaging_result': 'imaging_result',
                'imagingresult': 'imaging_result',
                'examination_measure': 'examination_measure',
                'examinationmeasure': 'examination_measure',
                'parameter': 'parameter',
                'score': 'score',
                'therapy': 'therapy',
                'substance': 'substance',
                'adverse_event': 'adverse_event',
                'adverseevent': 'adverse_event',
                'chemical': 'substance',
                'drug': 'substance',
                'medication': 'substance',
                'procedure': 'therapeutic_procedure',
                'treatment': 'therapy',
                'quantitative_measure': 'parameter',
                'quantitativemeasure': 'parameter',
                'measure': 'parameter',
            }
            
            normalized_type = type_mapping.get(entity_type_snake, entity_type_snake)
            return f"{prefix}{normalized_type}"
        
        return label
    
    def fine_tune_incremental(
        self,
        model_name: str,
        new_data: str,
        epochs: int = 1,
        learning_rate: float = 1e-5
    ) -> Dict[str, Any]:
        model_path = self.entity_models_dir / model_name
        model_type = "entity"
        if not model_path.exists():
            model_path = self.relation_models_dir / model_name
            model_type = "relation"
        if not model_path.exists():
            model_path = self.models_dir / model_name
            model_type = "entity"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model '{model_name}' not found")
        
        return self.train(
            model_name=model_name,
            base_model=str(model_path),
            training_data=new_data,
            epochs=epochs,
            learning_rate=learning_rate,
            model_type=model_type
        )


_bert_ner_service: Optional[BERTNERService] = None


def get_bert_ner_service() -> BERTNERService:
    """Get the global BERT NER service instance."""
    global _bert_ner_service
    if _bert_ner_service is None:
        _bert_ner_service = BERTNERService()
    return _bert_ner_service
