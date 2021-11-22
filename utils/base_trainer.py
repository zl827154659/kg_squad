# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from . import Vn
from . import mkdir_if_notexist

import torch
from . import logger
from transformers.optimization import AdamW
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers.file_utils import WEIGHTS_NAME, CONFIG_NAME

import os
from tqdm.autonotebook import tqdm
import pdb


class BaseTrainer:
    def __init__(self, model, multi_gpu, device, print_step, gradient_accumulation_steps, output_model_dir, vn):
        self.model = model.to(device)
        self.device = device
        self.multi_gpu = multi_gpu
        self.print_step = print_step
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_model_dir = output_model_dir

        self.vn = vn
        self.train_record = Vn(vn)

    def set_optimizer(self, optimizer):
        if self.fp16:
            model, optimizer = amp.initialize(self.model, optimizer, opt_level='O1')
            self.model = model
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def train(self, epoch_num, train_dataloader, dev_dataloader,
              save_last=True, freeze_lm_epochs=0):

        best_dev_loss = float('inf')
        best_dev_acc = 0
        self.completed_step = 0
        self.train_record.init()
        self.model.zero_grad()
        if freeze_lm_epochs > 0:
            if self.multi_gpu:
                self.model.module.freeze_lm()
            else:
                self.model.freeze_lm()

        for epoch in range(int(epoch_num)):
            if epoch == freeze_lm_epochs and freeze_lm_epochs > 0:
                if self.multi_gpu:
                    self.model.module.unfreeze_lm()
                else:
                    self.model.unfreeze_lm()
            print(f'---- Epoch: {epoch+1:02} ----')
            for step, batch in enumerate(tqdm(train_dataloader, desc='Train')):
                self.model.train()
                results = self._forward(batch, self.train_record)
                loss = results[0]
                acc = results[1]
                loss = loss / self.gradient_accumulation_steps
                acc = acc / self.gradient_accumulation_steps
                self.train_record.inc([loss.item(), acc.item()])
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1)  # max_grad_norm = 1

                if step % self.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    self._step(batch)

                if self.completed_step % self.print_step == 0:

                    dev_record = self.evaluate(dev_dataloader)
                    self.model.zero_grad()

                    self._report(self.train_record, dev_record)
                    current_acc = dev_record.list()[1]
                    if current_acc > best_dev_acc:
                        best_dev_acc = current_acc
                        self.save_best(best_acc=best_dev_acc)

                    self.train_record.init()

        dev_record = self.evaluate(dev_dataloader)
        self._report(self.train_record, dev_record)

        dloss, dacc, _ = dev_record.avg()
        if save_last:
            self.save_last(last_acc=dacc)

    def _forward(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        all_results = self.model(*batch)
        results = all_results[:3]
        results = tuple([results[0].mean(), results[1].sum(), results[2].sum()])
        return results

    def _mean(self, tuples):
        if self.multi_gpu:
            return tuple(v.mean() for v in tuples)
        return tuples

    def _mean_sum(self, tuples):
        """
        mean if float, sum if int
        """
        if self.multi_gpu:
            return tuple(v.mean() if v.is_floating_point() else v.sum() for v in tuples)
        return tuples

    def _step(self, batch):
        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad()
        self.completed_step += 1

    def evaluate(self, dataloader, desc='Eval'):
        record = Vn(self.vn)
        print('model eval')
        for batch in dataloader:
            self.model.eval()
            with torch.no_grad():
                results = self._forward(batch)
                results = tuple([results[0],
                                 results[1],
                                 results[2]])
                record.inc([it.item() for it in results])

        return record

    def _report(self, train_record, devlp_record):
        tloss, tacc = train_record.avg()
        dloss, dacc = devlp_record.avg()
        print("\t\tTrain loss %.4f acc %.4f | Dev loss %.4f acc %.4f" % (
                tloss, tacc, dloss, dacc))

    def make_optimizer(self, weight_decay, lr):
        params = list(self.model.named_parameters())

        no_decay_keywords = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        def _no_decay(n):
            return any(nd in n for nd in no_decay_keywords)

        parameters = [
            {'params': [p for n, p in params if _no_decay(n)], 'weight_decay': 0.0},
            {'params': [p for n, p in params if not _no_decay(n)],
             'weight_decay': weight_decay}
        ]

        optimizer = AdamW(parameters, lr=lr, eps=1e-8)
        return optimizer

    def make_scheduler(self, optimizer, warmup_proportion, t_total):
        return get_cosine_with_hard_restarts_schedule_with_warmup(
          optimizer, num_warmup_steps=warmup_proportion * t_total,
          num_training_steps=t_total)

    def save_model(self):
        mkdir_if_notexist(self.output_model_dir)
        logger.info('saving model {}'.format(self.output_model_dir))
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.output_model_dir)

    def save_best(self, best_acc):
        best_model_dir = os.path.join(self.output_model_dir, 'best_model')
        mkdir_if_notexist(os.path.join(best_model_dir, 'best_acc.txt'))
        with open(os.path.join(best_model_dir, 'best_acc.txt'), 'w') as f:
            f.write(f'best dev acc: {best_acc}')
        logger.info('saving model {}'.format(best_model_dir))
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(best_model_dir)

    def save_last(self, last_acc):
        last_model_dir = os.path.join(self.output_model_dir, 'last_model')
        mkdir_if_notexist(os.path.join(last_model_dir, 'last_acc.txt'))
        with open(os.path.join(last_model_dir, 'last_acc.txt'), 'w') as f:
            f.write(f'last dev acc: {last_acc}')
        logger.info('saving model {}'.format(last_model_dir))
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(last_model_dir)