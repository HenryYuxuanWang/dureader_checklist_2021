import os
import time
import json
import logging

import torch
import numpy as np
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer, BertTokenizerFast, AdamW, set_seed, get_linear_schedule_with_warmup
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from progress_bar import ProgressBar
from squad import SquadProcessor
from training_args import training_args
from early_stopping import EarlyStopping
from squad import compute_prediction_checklist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.classifier_cls = nn.Linear(config.hidden_size, 2)
        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)
        sequence_output, pooled_output = outputs[0], outputs[1]

        logits = self.classifier(sequence_output)
        logits = logits.permute(2, 0, 1)
        start_logits, end_logits = logits[0], logits[1]
        cls_logits = self.classifier_cls(pooled_output)

        return start_logits, end_logits, cls_logits


class CrossEntropyLossForChecklist(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLossForChecklist, self).__init__()

    def forward(self, y, label, mask=None):
        start_logits, end_logits, cls_logits = y
        start_position, end_position, answerable_label = label
        if mask is not None:
            start_logits = start_logits - (1.0 - mask) * 1e12
            end_logits = end_logits - (1.0 - mask) * 1e12
        start_loss = F.cross_entropy(start_logits, start_position)
        end_loss = F.cross_entropy(end_logits, end_position)
        cls_loss = F.cross_entropy(cls_logits, answerable_label)
        mrc_loss = (start_loss + end_loss) / 2
        loss = (mrc_loss + cls_loss) / 2
        return loss


class Model:
    def __init__(self, path='./chinese-roberta-wwm-ext'):
        self.model_dir = path
        self.model = None
        self.processor = SquadProcessor()
        self.args = training_args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_dir)

    @staticmethod
    def sequence_padding(inputs, length=None, padding=0, mode='post'):
        if length is None:
            length = max([len(x) for x in inputs])

        pad_width = [(0, 0) for _ in np.shape(inputs[0])]
        outputs = []
        for x in inputs:
            x = x[:length]
            if mode == 'post':
                pad_width[0] = (0, length - len(x))
            elif mode == 'pre':
                pad_width[0] = (length - len(x), 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=padding)
            outputs.append(x)

        return np.array(outputs)

    def prepare_training_features(self, examples):
        contexts = [examples[i]['context'] for i in range(len(examples))]
        questions = [examples[i]['question'] for i in range(len(examples))]

        tokenized_examples = self.tokenizer(
            questions,
            contexts,
            stride=self.args.doc_stride,
            truncation=True,
            max_length=self.args.max_seq_length
        )

        new_tokenized_examples = []
        for n in range(len(tokenized_examples['input_ids'])):
            example = tokenized_examples[n]
            example.sample_index = n
            new_tokenized_examples.append(example)
            if example.overflowing:
                for sample in example.overflowing:
                    sample.sample_index = n
                    new_tokenized_examples.append(sample)

        tokenized_examples = new_tokenized_examples
        for i, tokenized_example in enumerate(tokenized_examples):
            input_ids = tokenized_example.ids
            cls_index = input_ids.index(self.tokenizer.cls_token_id)
            offsets = tokenized_example.offsets
            sequence_ids = tokenized_example.type_ids
            sample_index = tokenized_example.sample_index
            answers = examples[sample_index]['answers']
            answer_starts = examples[sample_index]['answer_starts']

            if len(answer_starts) == 0:
                tokenized_examples[i].start_positions = cls_index
                tokenized_examples[i].end_positions = cls_index
                tokenized_examples[i].answerable_label = 0
            else:
                start_char = answer_starts[0]
                end_char = start_char + len(answers[0])

                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                token_end_index = len(input_ids) - 2
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples[i].start_positions = cls_index
                    tokenized_examples[i].end_positions = cls_index
                    tokenized_examples[i].answerable_label = 0
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples[i].start_positions = token_start_index - 1
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples[i].end_positions = token_end_index + 1
                    tokenized_examples[i].answerable_label = 1
        return tokenized_examples

    def prepare_validation_features(self, examples):
        contexts = [examples[i]['context'] for i in range(len(examples))]
        questions = [examples[i]['question'] for i in range(len(examples))]

        tokenized_examples = self.tokenizer(
            questions,
            contexts,
            stride=self.args.doc_stride,
            truncation=True,
            max_length=self.args.max_seq_length,
        )

        new_tokenized_examples = []
        for n in range(len(tokenized_examples['input_ids'])):
            example = tokenized_examples[n]
            example.sample_index = n
            new_tokenized_examples.append(example)
            if example.overflowing:
                for sample in example.overflowing:
                    sample.sample_index = n
                    new_tokenized_examples.append(sample)

        tokenized_examples = new_tokenized_examples

        for i, tokenized_example in enumerate(tokenized_examples):
            sequence_ids = tokenized_example.type_ids

            sample_index = tokenized_example.sample_index
            tokenized_examples[i].example_id = examples[sample_index]['id']

            tokenized_examples[i].offset_mapping = [(o if sequence_ids[k] == 1 else None) for k, o in enumerate(tokenized_example.offsets)]

        return tokenized_examples

    def prepare_dataset(self, data, args, is_training=True):
        input_ids = torch.tensor(self.sequence_padding([f.ids for f in data], length=args.max_seq_length), dtype=torch.long)
        attention_mask = torch.tensor(self.sequence_padding([f.attention_mask for f in data], length=args.max_seq_length), dtype=torch.long)
        token_type_ids = torch.tensor(self.sequence_padding([f.type_ids for f in data], length=args.max_seq_length), dtype=torch.long)
        inputs = [input_ids, attention_mask, token_type_ids]
        if is_training:
            start_positions = torch.tensor([f.start_positions for f in data], dtype=torch.long)
            end_positions = torch.tensor([f.end_positions for f in data], dtype=torch.long)
            answerable_label = torch.tensor([f.answerable_label for f in data], dtype=torch.long)
            inputs += [start_positions, end_positions, answerable_label]
        dataset = TensorDataset(*inputs)
        return dataset

    def train(self, train_data, eval_data=None, args=None):
        if args is None:
            args = self.args
        self.load(self.model_dir)
        early_stopping = EarlyStopping(verbose=True)
        dataset = self.prepare_dataset(train_data, args)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size)

        if args.max_steps > 0:
            num_training_steps = args.max_steps
            args.num_train_epochs = args.max_steps // (len(dataloader) // args.gradient_accumulation_steps) + 1
        else:
            num_training_steps = len(dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        args.warmup_steps = int(num_training_steps * args.warmup_ratio)
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
        # multi-gpu training
        if args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", num_training_steps)

        criterion = CrossEntropyLossForChecklist()
        global_step = 0
        self.model.zero_grad()
        set_seed(args.seed)  # Added here for reproducibility (even between python 2 and 3)
        print('total epochs : {}'.format(args.num_train_epochs))
        print('train_dataloader length : {}'.format(len(dataloader)))
        for epoch in range(int(args.num_train_epochs)):
            pbar = ProgressBar(n_total=len(dataloader), desc='Training')
            self.model.train()
            losses = []
            for step, batch in enumerate(dataloader):
                batch = tuple(t.to(args.device) for t in batch)
                input_ids, attention_mask, segment_ids, start_positions, end_positions, answerable_label = batch
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
                loss = criterion(logits, (start_positions, end_positions, answerable_label), mask=attention_mask)
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                losses.append(loss.cpu().detach().numpy())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    global_step += 1
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()

                if global_step % args.save_steps == 0 or global_step == num_training_steps:
                    output_dir = os.path.join(args.output_dir, "model_%d" % global_step)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    model_to_save.save_pretrained(output_dir)
                    print('Saving checkpoint to:', output_dir)

                pbar(step, {'loss': loss.item()})
            print(" ")
            if 'cuda' in str(args.device):
                torch.cuda.empty_cache()
            if eval_data:
                self.evaluate(eval_data, prefix="dev")
                filename = os.path.join(args.data_dir, 'dev.json')
                metrics = json.loads(
                    os.popen("python ./evaluate.py %s %s" % (filename, os.path.join(args.output_dir, "dev_predictions.json"))).read().strip()
                )
                metrics['F1'] = float(metrics['F1'])
                metrics['EM'] = float(metrics['EM'])
                print(" ")
                print('Epoch:', epoch, 'F1:', metrics['F1'], 'EM:', metrics['EM'])
                early_stopping(metrics['F1'], self.model, args.output_dir)

            output_dir = os.path.join(args.output_dir, "model_epoch_%d" % epoch)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_save.save_pretrained(output_dir)

    def evaluate(self, data, args=None, prefix=""):
        if not self.model:
            self.load(self.model_dir)
        if not args:
            args = self.args
        if args.n_gpu > 1 and device.type == 'cuda':
            if not isinstance(self.model, nn.DataParallel):
                self.model = torch.nn.DataParallel(self.model)
        examples = data['examples']
        features = data['features']
        dataset = self.prepare_dataset(features, args, is_training=False)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)

        pbar = ProgressBar(n_total=len(dataloader), desc="Evaluating")
        self.model.eval()
        all_start_logits = []
        all_end_logits = []
        all_cls_logits = []
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                input_ids, attention_mask, segment_ids = batch
                start_logits_tensor, end_logits_tensor, cls_logits_tensor = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                                                                       token_type_ids=segment_ids)
                for idx in range(start_logits_tensor.shape[0]):
                    all_start_logits.append(start_logits_tensor.cpu().detach().numpy()[idx])
                    all_end_logits.append(end_logits_tensor.cpu().detach().numpy()[idx])
                    all_cls_logits.append(cls_logits_tensor.cpu().detach().numpy()[idx])
            pbar(step)

        features = [feature.__dict__ for feature in features]
        all_predictions, all_nbest_json, all_cls_predictions = compute_prediction_checklist(
            examples, features,
            (all_start_logits, all_end_logits, all_cls_logits), True, args.n_best_size,
            args.max_answer_length, args.cls_threshold)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        with open(os.path.join(args.output_dir, prefix + '_predictions.json'), "w", encoding='utf-8') as writer:
            writer.write(json.dumps(all_predictions, ensure_ascii=False, indent=4) + "\n")

        with open(os.path.join(args.output_dir, prefix + '_nbest_predictions.json'), "w", encoding="utf8") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + u"\n")

        if all_cls_predictions:
            with open(os.path.join(args.output_dir, prefix + "_cls_preditions.json"), "w") as f_cls:
                for cls_predictions in all_cls_predictions:
                    qas_id, pred_cls_label, no_answer_prob, answerable_prob = cls_predictions
                    f_cls.write('{}\t{}\t{}\t{}\n'.format(qas_id, pred_cls_label, no_answer_prob, answerable_prob))

    def load(self, path):
        self.model = BertForQuestionAnswering.from_pretrained(path)
        self.model.to(device)

    def save(self, output_dir):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
