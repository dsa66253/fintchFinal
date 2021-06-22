import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn
import os
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from .configs import configs
import nni


class Trainer():
    def __init__(self, epoch, batch_size, logger, device, train_dataloader, valid_dataloader, optimizer, scheduler, lr_scheduler, model, num_train_steps):
        self.logger = logger
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.epoch = epoch
        self.gradient_accumulation_steps = 1
        self.global_step = 0
        self.model = model
        pos_weight = torch.tensor([1,1000], device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.earlystop_cnt = 0
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        # self.writer = SummaryWriter("train_visual")
        self.model.to(self.device)
        self.logger.info('device:{}'.format(self.device))
        self.logger.info('there are {} steps / epoch'.format(len(train_dataloader)))

    def train(self):
        min_avg_loss = 1000000

        self.logger.info("Training Start")
        for epoch in range(self.epoch):
            self.logger.info("Training epoch %s" %(epoch))
            self._train_epoch(epoch)
            valid_loss = self._valid_epoch(epoch)
            # nni.report_intermediate_result(eval_matrix)
            if valid_loss < min_avg_loss:
                self.earlystop_cnt = 0
                min_avg_loss = valid_loss
                #=========store the ckpt==========
                self.logger.info('save the model ckpt {}'.format(configs.ckpt_simplemodel_path))
                torch.save(
                    {
                        'state_dict': self.model.state_dict(),
                        'epoch': epoch,
                    },
                    configs.ckpt_simplemodel_path
                )
            else:
                self.earlystop_cnt+=1
            
            if self.earlystop_cnt >= 5:
                self.logger.info('early stop')
                break
        # nni.report_final_result(eval_matrix)
    
    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loss_cnt = 0
        N_step = 100
        for batch_idx, (x_batch, y_batch) in enumerate(self.train_dataloader):
            x_batch = x_batch.to(self.device)
            one_hot_target = torch.zeros((y_batch.shape[0], 2))
            one_hot_target[y_batch == 1, 1] = 1
            one_hot_target[y_batch == 0, 0] = 1
            one_hot_target = one_hot_target.to(self.device)
            logits = self.model(inputs=x_batch)
            loss = self.criterion(logits,one_hot_target)
            total_loss+=(loss.item())
            loss_cnt+=1
            # if self.gradient_accumulation_steps > 1:
            #     loss = loss / self.gradient_accumulation_steps
            loss.backward()
            avg_loss = total_loss / loss_cnt

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.lr_scheduler.batch_step(training_step = self.global_step)
                self.optimizer.zero_grad()
                self.global_step += 1
                # self.writer.add_scalar('Loss/train', avg_loss, self.global_step)
            
            if batch_idx % N_step == 0 and batch_idx > 0:
                self.logger.info('Epoch: {} step: {}, loss: {}'.format(epoch+1, self.global_step, avg_loss))
                total_loss = 0
                loss_cnt = 0
    
    def _valid_epoch(self, epoch):
        self.logger.info("*"*10)
        self.logger.info("Dev Start")
        self.logger.info("*"*10)
        self.model.eval()
        avg_loss = 0
        N_step = len(self.valid_dataloader)
        for x_batch, y_batch in self.valid_dataloader:
            with torch.no_grad():
                x_batch = x_batch.to(self.device)
                one_hot_target = torch.zeros((y_batch.shape[0], 2))
                one_hot_target[y_batch == 1, 1] = 1
                one_hot_target[y_batch == 0, 0] = 1
                one_hot_target = one_hot_target.to(self.device)
                logits = self.model(inputs=x_batch)
                loss = self.criterion(logits,one_hot_target)
                avg_loss+=(loss.item())

        avg_loss = avg_loss / N_step
        self.logger.info('Epoch: {} step: {}, val_loss: {}'.format(epoch+1, self.global_step, avg_loss))
        # self.writer.add_scalar('Loss/valid', avg_loss, self.global_step)
        return avg_loss
        