import time
import torch
import numpy as np
import matplotlib
import mlflow
matplotlib.use('agg')  # 에러발생방지
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from pprint import pformat
from dataset import FoodDataset
from torch.nn.modules.loss import CrossEntropyLoss
from custom_layers import *
from utils import split_train_valid

import models
LOADER = {
    'model': {
        'SwinV2_Base_256': models.load_SwinV2_Base_256,
    },
    
    'optimizer': {
        'AdamW': torch.optim.AdamW,
        'SGD': torch.optim.SGD,
    },
    
    'scheduler': {
        'NoneScheduler': NoneScheduler,
        'CustomCosineAnnealingWarmupRestarts': CustomCosineAnnealingWarmupRestarts,
        'ExpTargetIterScheduler': ExpTargetIterScheduler,
    }
}


class Trainer:
    # ------------------------------------Initialization------------------------------------------------
    def __init__(self, config, config_select_dict, test_mode=False):
        # Init variables
        self.config = config
        self.config_select_dict = config_select_dict
        self.test_mode = test_mode
        self.img_shape = tuple(config['dataset']['img_shape'])
        self.cut_iter = self.config['trainer']['cut_iter']
        self.data_parallel = True if len(self.config['trainer']['cuda_device'].split(','))>=2 else False

        # Load model
        self.model = torch.nn.DataParallel(self.load_model()).cuda() if self.data_parallel else self.load_model().cuda()
        self.total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total Parameters : {self.total_params:,}')
        if not test_mode: mlflow.log_param('Parameters', self.total_params)
        
        # Load Dataset
        label_to_idx, trainset_paths, validset_paths = split_train_valid(self.config['dataset']['dataset_path'], self.config['dataset']['train_valid_split_ratio'], self.config['dataset']['random_sate'])
        self.trainset = FoodDataset(trainset_paths, label_to_idx=label_to_idx, apply_aug=config['trainer']['apply_aug'], **config['dataset'])
        self.validset = FoodDataset(validset_paths, label_to_idx=label_to_idx, apply_aug=False, **config['dataset'])

        # Load Sampler
        custom_weights = self.config['trainer']['train_custom_weights']
        self.train_sampler = ImbalancedDatasetSampler(self.trainset, custom_weights=custom_weights) if custom_weights else RandomSampler(self.trainset)
        self.valid_sampler = SequentialSampler(self.validset)

        # Load dataloader
        num_workers = self.config['trainer']['num_workers']
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=self.config['trainer']['train_batch_size'],
                                num_workers=num_workers, drop_last=True, sampler=self.train_sampler)
        self.valid_loader = torch.utils.data.DataLoader(self.validset, batch_size=self.config['trainer']['valid_batch_size'],
                                num_workers=num_workers, drop_last=False, sampler=self.valid_sampler)

        # Load loss, optimizer, scheduler
        self.loss = CrossEntropyLoss()
        self.optimizer = self.load_optimizer()
        self.scheduler = self.load_scheduler()

        # init training variables
        self.epoch = self.config['trainer']['start_epoch']
        self.chk_metric_scores = []
        
    def __str__(self):
        str_total_params = f'Total Parameters : {self.total_params:,}'
        str_config_select_dict = pformat(self.config_select_dict)
        str_config = pformat(self.config)

        return '\n\n'.join([str_total_params, str_config_select_dict, str_config, ''])

    # ------------------------------------Load Layers------------------------------------------------        
    def load_model(self):
        model_load_func = LOADER['model'][self.config_select_dict['model']]
        model_kwargs = self.config['model'][self.config_select_dict['model']]
        model_kwargs['img_shape'] = self.img_shape

        return model_load_func(**model_kwargs)

    def load_optimizer(self):
        optim_load_func = LOADER['optimizer'][self.config_select_dict['optimizer']]
        optim_kwargs = self.config['optimizer'][self.config_select_dict['optimizer']]

        return optim_load_func(lr=self.config['trainer']['lr'], params=self.model.parameters(), **optim_kwargs)

    def load_scheduler(self):
        scheduler_load_func = LOADER['scheduler'][self.config_select_dict['scheduler']]
        scheduler_kwargs = self.config['scheduler'][self.config_select_dict['scheduler']]

        if self.config_select_dict['scheduler'] == 'CustomCosineAnnealingWarmupRestarts':
            # Set max_lr & first_cycle_stemps
            scheduler_kwargs['max_lr'] = self.config['trainer']['lr']
            scheduler_kwargs['first_cycle_steps'] = self.cut_iter if self.cut_iter!=-1 else len(self.train_loader)
            # Calc gamma
            init_lr = self.config['trainer']['lr']; gamma_last_lr = scheduler_kwargs['gamma_last_lr']
            target_epoch = self.config['trainer']['target_epoch']
            scheduler_kwargs['gamma'] = (gamma_last_lr/init_lr)**(1/target_epoch)

        return scheduler_load_func(optimizer=self.optimizer, **scheduler_kwargs)

    # ------------------------------------Train Methods------------------------------------------------
    def training_step(self, cut_iter=-1, verbose=True):
        max_iter_num = cut_iter if cut_iter!=-1 else len(self.train_loader)
        loss_list = []
        correct_sum, total_sum = 0, 0

        self.model.train()
        for iter_num, data in enumerate(self.train_loader):
            # forward
            input_batch, label_batch = data['input'].cuda(), data['label'].cuda()
            pred = self.model(input_batch)
            
            # calc loss
            loss = self.loss(pred, label_batch)
            loss_list.append(loss.item())

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # calc Acc
            _, indices = torch.max(pred, axis=1)
            correct_sum += (indices==label_batch).sum().item()
            total_sum += len(label_batch)
            acc = correct_sum/total_sum

            # Print info
            if verbose:
                print('\rEpoch: {}, Train_Iter: {}/{}, loss: {:.8f}, acc: {:.2f}%, lr: {:.8f}'
                    .format(self.epoch, iter_num+1, max_iter_num, np.mean(loss_list).item(),
                        acc, self.optimizer.param_groups[0]['lr']
                    ), end=' ')

            # Cut epoch
            if cut_iter!=-1 and (iter_num+1)>=cut_iter:
                break

        return np.mean(loss_list).item(), acc

    def validation_step(self, verbose=True):
        loss_list = []
        correct_sum, total_sum = 0, 0

        with torch.no_grad():
            self.model.eval()
            for iter_num, data in enumerate(self.valid_loader):
                # forward
                input_batch, label_batch = data['input'].cuda(), data['label'].cuda()
                pred = self.model(input_batch)
                
                # calc loss
                loss = self.loss(pred, label_batch)
                loss_list.append(loss.item())

                # calc Acc
                _, indices = torch.max(pred, axis=1)
                correct_sum += (indices==label_batch).sum().item()
                total_sum += len(label_batch)
                acc = correct_sum/total_sum

                # Print info
                if verbose:
                    print('\rEpoch: {}, Valid_Iter: {}/{}, loss: {:.8f}, acc: {:.2f}%, lr: {:.8f}'
                        .format(self.epoch, iter_num+1, len(self.valid_loader), np.mean(loss_list).item(),
                            acc, self.optimizer.param_groups[0]['lr']
                        ), end=' ')

        return np.mean(loss_list).item(), acc

    # ------------------------------------Utils------------------------------------------------

    # ------------------------------------Run Train------------------------------------------------
    def run_train(self):
        # config variables
        target_epoch = self.config['trainer']['target_epoch']
        min_chkpoint_epoch = self.config['trainer']['min_chkpoint_epoch']
        
        metric = self.config['trainer']['metric']
        
        # Set Chkpoint metric variables
        chkpoint_metric = self.config['trainer']['chkpoint_metric']
        if 'loss' in chkpoint_metric:
            compare_type = 'min'
            self.chk_metric_scores.append(float('inf'))
        elif 'acc' in chkpoint_metric:
            compare_type = 'max'
            self.chk_metric_scores.append(float('-inf'))
        else:
            raise Exception(f'Unknown chkpoint metric type : {chkpoint_metric}')

        # Start Training
        run_time_per_epoch = []
        while self.epoch <= target_epoch:
            epoch_start_time = time.time()

            # --------------------------------------------Train-----------------------------------------------------
            # Train & Valid step
            metric_dict = {}
            metric_dict['train_loss'], metric_dict['train_acc'] = self.training_step(cut_iter=self.cut_iter, verbose=True)
            metric_dict['valid_loss'], metric_dict['valid_acc'] = self.validation_step(verbose=True)

            # Chkpoint Save
            if (compare_type == 'min' and metric_dict[chkpoint_metric] < min(self.chk_metric_scores)) or \
                (compare_type == 'max' and metric_dict[chkpoint_metric] > max(self.chk_metric_scores)):
                mlflow.log_metric('chkpoint', metric_dict[chkpoint_metric], step=self.epoch)
                if self.epoch >= min_chkpoint_epoch:
                    mlflow.pytorch.log_model(self.model.module if self.data_parallel else self.model, 'chkpoint')
            
            # Log chkpoint scores
            if self.epoch >= min_chkpoint_epoch:
                self.chk_metric_scores.append(metric_dict[chkpoint_metric])

            # MLFlow logging
            mlflow.log_metrics(metric_dict, step=self.epoch)
            # -------------------------------------------------------------------------------------------------------

            # calc remaining time
            run_time_per_epoch.append(time.time()-epoch_start_time)
            per_epoch_time = np.mean(run_time_per_epoch[-50:]).item() ; remaining_time = (target_epoch-self.epoch) * per_epoch_time
            h, m = divmod(round(remaining_time), 3600); m, s = divmod(round(m), 60)

            # Print Info
            print('\rEpoch: {}/{}, train_loss: {:.6f}, valid_loss: {:.6f}, train_acc: {:.2f}%, valid_acc: {:.2f}%, remain_time: {}'
                .format(self.epoch, target_epoch, metric_dict['train_loss'], metric_dict['valid_loss'],
                    metric_dict['train_acc'], metric_dict['valid_acc'], f'{h:02}H_{m:02}M'
                )
            )

            self.epoch += 1
