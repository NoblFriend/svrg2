from utils import set_seed
from utils import get_device 
from utils import get_resnet18, get_simple_model
from utils import Logger
from utils import get_data_loaders
from utils import loader_to_device
import torch
from tqdm import tqdm
import json
import os
import numpy as np
NCOLS = 100


import time  # Для измерения времени


import os
import json
import torch
from tqdm import tqdm
from utils import Logger  # Предполагается, что Logger импортируется из utils
from utils import top_k_accuracy

class BaseOptimizer:
    def __init__(self, model, data_loaders, loss_fn, device, lambda_value=None, scheduler=None, scheduler_config=None):
        self.model = model
        self.train_data_loader = data_loaders[0]
        self.test_data_loader = data_loaders[1]
        self.train_batch_count = len(self.train_data_loader)
        self.test_batch_count = len(self.test_data_loader)
        self.loss_fn = loss_fn
        self.loggers = {}
        self.train_logger = Logger()
        self.test_logger = Logger()
        self.grads_epochs_computed = 0
        self.regularizer = lambda_value
        self.device = device
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config or {}

    def run(self, epochs, lr, exp_name=None, scheduler_func=None):
        self.start_time = time.time()  # Засекаем время начала всего обучения
        if exp_name is None:
            exp_name = f'{self.__class__.__name__}_{lr=}_{epochs=}'
            if self.regularizer is not None:
                exp_name += f'_lambda={self.regularizer}'

        if scheduler_func is not None:
            config = {
                'warmup_epochs': 5,
                'total_epochs': epochs,
                'base_lr': lr,
                'warmup_lr': lr * 0.1,
                'min_lr': 1e-5,
                **self.scheduler_config 
            }
            self.scheduler = scheduler_func(**config)
        else:
            self.scheduler = lambda epoch: lr 

        for epoch in tqdm(range(epochs), desc=exp_name, ncols=100):
            current_lr = self.scheduler(epoch)
            self._train_epoch(current_lr)
            tqdm.write(f'Train Loss: {self.train_logger.loss[-1]:.4f},\t Train Accuracy: {self.train_logger.accuracy[-1]:.4f}')
            self._test_epoch()
            tqdm.write(f'Test  Loss: {self.test_logger.loss[-1]:.4f},\t Test  Accuracy: {self.test_logger.accuracy[-1]:.4f}')
            tqdm.write(f'Grads epochs computed: {self.grads_epochs_computed:.2f}')
            total_time = time.time() - self.start_time
            tqdm.write(f'Training Time: {total_time:.2f}s')

        self.loggers[exp_name] = {
            'train': self.train_logger.to_dict(),
            'test': self.test_logger.to_dict(),
        }
        self.train_logger = Logger()
        self.test_logger = Logger()


    def _forward_backward(self, model, inputs, targets, zero_grad=True, is_test=False):
        if zero_grad:
            model.zero_grad()
        if is_test:
            model.eval()
        else:
            model.train()
        outputs = model(inputs)
        loss = self.loss_fn(outputs, targets)
        # Добавление L2 регуляризации
        if self.regularizer is not None:
            for param in model.parameters():
                loss += (self.regularizer/2) * torch.sum(param ** 2)

        if not is_test:
            loss.backward()
        if not is_test:
            self.grads_epochs_computed += 1 / self.train_batch_count
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        batch_accuracy = correct / targets.size(0)

        acc_at_k = top_k_accuracy(outputs, targets, ks=(1, 2, 3, 4, 5))
        return loss, batch_accuracy, acc_at_k

    def _test_epoch(self):
        test_loss = 0
        test_accuracy = 0
        test_data = loader_to_device(self.test_data_loader, self.device)
        topk_aggregate = {f'top@{k}': 0.0 for k in range(1, 6)}
        for inputs, targets in tqdm(test_data, desc='Testing', ncols=100, leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            loss, acc, acc_at_k = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=True)
            test_loss += loss.item()
            test_accuracy += acc
            for k in acc_at_k:
                topk_aggregate[k] += acc_at_k[k]
        avg_topk = {k: v / self.test_batch_count for k, v in topk_aggregate.items()}

        self.test_logger.append(test_loss / self.test_batch_count,
                                test_accuracy / self.test_batch_count,
                                self.grads_epochs_computed,
                                time.time() - self.start_time,
                                avg_topk)


    def dump_json(self, directory='experiments'):
        os.makedirs(directory, exist_ok=True)
        for exp_name, data in self.loggers.items():
            with open(f'{directory}/{exp_name}.json', 'w') as f:
                json.dump(data, f)

    def _train_epoch(self, lr):
        """Метод, реализующий обновление параметров для одной эпохи.
        Должен быть реализован в наследниках."""
        raise NotImplementedError("Метод _train_epoch должен быть реализован в наследнике.")
    

    def _compute_full_grad(self):
        self.model.zero_grad()
        for inputs, targets in tqdm(self.train_data_loader, desc='Training', ncols=100, leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self._forward_backward(self.model, inputs, targets, zero_grad=False, is_test=False)
        # Возвращаем усреднённый градиент по всему датасету
        return [param.grad.detach().clone() / self.train_batch_count for param in self.model.parameters() if param.requires_grad]


class SGD(BaseOptimizer):
    def _train_epoch(self, lr):
        self.model.train()
        for inputs, targets in tqdm(self.train_data_loader, desc='Training', ncols=100, leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            loss, accuracy, acc_at_k = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
            self.train_logger.append(loss.item(), accuracy, self.grads_epochs_computed, time.time() - self.start_time, acc_at_k)
            # Обновление параметров по SGD
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.requires_grad:
                        param.add_(param.grad, alpha=-lr)


class SVRG_Pick(BaseOptimizer):
    def __init__(self, model, model_ref, data_loaders, loss_fn, device, lambda_value=None, freq=3.5, scheduler=None, scheduler_config=None):
        super().__init__(model, data_loaders, loss_fn, device, lambda_value, scheduler_config=scheduler_config)
        self.model_ref = model_ref
        self.freq = freq
        self.m = int(self.train_batch_count/self.freq)
        self._g_ref = None

    def _train_epoch(self, lr):
        self._g_ref = self._compute_full_grad()
        self.model_ref.load_state_dict(self.model.state_dict())
        for inputs, targets in tqdm(self.train_data_loader, desc='Training', ncols=100, leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # Вычисляем градиенты для текущего батча на основной модели
            loss, accuracy, acc_at_k = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
            self.train_logger.append(loss.item(), accuracy, self.grads_epochs_computed, time.time() - self.start_time, acc_at_k)
            # Вычисляем градиенты для того же батча на эталонной модели
            _, _, _ = self._forward_backward(self.model_ref, inputs, targets, zero_grad=True, is_test=False)
            # Обновление параметров по схеме SVRG
            with torch.no_grad():
                for param, param_ref, grad_ref in zip([p for p in self.model.parameters() if p.requires_grad],
                                                      [p for p in self.model_ref.parameters() if p.requires_grad],
                                                      self._g_ref):
                    param.add_(param.grad - param_ref.grad + grad_ref, alpha=-lr)


class NFGSVRG_AVG(BaseOptimizer):
    def __init__(self, model, model_ref, model_buffer, data_loaders, loss_fn, device, lambda_value=None, scheduler=None, scheduler_config=None):
        super().__init__(model, data_loaders, loss_fn, device, lambda_value, scheduler_config=scheduler_config)
        self.model_ref = model_ref
        self.model_ref.load_state_dict(self.model.state_dict())
        self.model_buffer = model_buffer
        self.model_buffer.load_state_dict(self.model.state_dict())
        self._g_ref = [torch.zeros_like(param) for param in self.model.parameters() if param.requires_grad] # v
        self._g_avg = [torch.zeros_like(param) for param in self.model.parameters() if param.requires_grad] # v_tilde
        
        
        with torch.no_grad():
            for param in self.model_buffer.parameters():
                param.data.zero_()

    def _train_epoch(self, lr):
        train_data = loader_to_device(self.train_data_loader, self.device)

        for batch_num, batch_idx in enumerate(tqdm(torch.randperm(self.train_batch_count).tolist(),
                        desc='Training', ncols=NCOLS, leave=False)):
            inputs, targets = train_data[batch_idx]
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # Вычисляем градиенты для текущего батча на основной модели
            loss, accuracy , acc_at_k = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
            self.train_logger.append(loss.item(), accuracy, self.grads_epochs_computed, time.time()-self.start_time, acc_at_k)
            # Вычисляем градиенты для того же батча на эталонной модели
            _, _, _ = self._forward_backward(self.model_ref, inputs, targets, zero_grad=True, is_test=False)
            # Обновление среднего градиента
            g_cur = [param.grad.detach().clone() for param in self.model.parameters() if param.requires_grad]
            for idx, grad in enumerate(g_cur):
                self._g_avg[idx]  +=   grad / self.train_batch_count
            # Обновление параметров по схеме SVRG
            with torch.no_grad():
                for param, param_ref, grad_ref in zip([p for p in self.model.parameters() if p.requires_grad],
                                                      [p for p in self.model_ref.parameters() if p.requires_grad],
                                                      self._g_ref):
                    param.add_(param.grad - param_ref.grad + grad_ref, alpha=-lr)
                    
            with torch.no_grad():
                for param_buffer, param in zip(self.model_buffer.parameters(), self.model.parameters()):
                    param_buffer.data.mul_(batch_num / (batch_num + 1))
                    param_buffer.data.add_(param.data / (batch_num + 1))
                    

            self._g_ref = [grad.clone() for grad in self._g_avg]
            for grad in self._g_avg:
                grad.zero_()
            self.model_ref.load_state_dict(self.model_buffer.state_dict())
            with torch.no_grad():
                for param in self.model_buffer.parameters():
                    param.data.zero_()


class NFGSARAH(BaseOptimizer):
    def __init__(self, model, model_prev, data_loaders, loss_fn, device, lambda_value=None, freq=1, scheduler=None, scheduler_config=None):
        super().__init__(model, data_loaders, loss_fn, device, lambda_value, scheduler_config=scheduler_config)
        self.model_prev = model_prev
        self.model_prev.load_state_dict(self.model.state_dict())
        self.freq = freq
        self.m = int(self.train_batch_count/self.freq)
        self._g_ref = [torch.zeros_like(param) for param in self.model.parameters() if param.requires_grad] # v
        self._g_avg = [torch.zeros_like(param) for param in self.model.parameters() if param.requires_grad] # v_tilde


    def _train_epoch(self, lr):
        train_data = loader_to_device(self.train_data_loader, self.device)
        for batch_num, batch_idx in enumerate(tqdm(np.random.choice(self.train_batch_count, self.m).tolist(),
                        desc='Training', ncols=NCOLS, leave=False)):
            inputs, targets = train_data[batch_idx]
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # Вычисляем градиенты для текущего батча на основной модели
            loss, accuracy, acc_at_k = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
            g_cur = [param.grad.detach().clone() for param in self.model.parameters() if param.requires_grad]
            self.train_logger.append(loss.item(), accuracy, self.grads_epochs_computed, acc_at_k=acc_at_k)

            for idx, grad in enumerate(g_cur):
                self._g_avg[idx]  +=  grad / self.train_batch_count

            loss, accuracy, _ = self._forward_backward(self.model_prev, inputs, targets, zero_grad=True, is_test=False)
            g_prev = [param.grad.detach().clone() for param in self.model_prev.parameters() if param.requires_grad]
            self.model_prev.load_state_dict(self.model.state_dict())
            
            for idx, _ in enumerate(self._g_ref):
                self._g_ref[idx].add_((g_cur[idx] - g_prev[idx]) / self.train_batch_count)

            with torch.no_grad():
                for param, grad_ref in zip([p for p in self.model.parameters() if p.requires_grad], self._g_ref):
                    param.add_(grad_ref, alpha=-lr)
        self._g_ref = [grad.clone() for grad in self._g_avg]
        self._g_avg = [torch.zeros_like(param) for param in self.model.parameters() if param.requires_grad] # v_tilde

class NFGSARAHnon(BaseOptimizer):
    def __init__(self, model, model_prev, data_loaders, loss_fn, device, lambda_value=None, freq=1, scheduler=None, scheduler_config=None):
        super().__init__(model, data_loaders, loss_fn, device, lambda_value, scheduler_config=scheduler_config)
        self.model_prev = model_prev
        self.model_prev.load_state_dict(self.model.state_dict())
        self.freq = freq
        self.m = int(self.train_batch_count/self.freq)
        self._g_ref = [torch.zeros_like(param) for param in self.model.parameters() if param.requires_grad] # v
        self._g_avg = [torch.zeros_like(param) for param in self.model.parameters() if param.requires_grad] # v_tilde


    def _train_epoch(self, lr):
        train_data = loader_to_device(self.train_data_loader, self.device)
        for batch_num, batch_idx in enumerate(tqdm(np.random.choice(self.train_batch_count, self.m).tolist(),
                        desc='Training', ncols=NCOLS, leave=False)):
            inputs, targets = train_data[batch_idx]
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # Вычисляем градиенты для текущего батча на основной модели
            loss, accuracy, acc_at_k = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
            g_cur = [param.grad.detach().clone() for param in self.model.parameters() if param.requires_grad]
            self.train_logger.append(loss.item(), accuracy, self.grads_epochs_computed, acc_at_k=acc_at_k)

            for idx, grad in enumerate(g_cur):
                self._g_avg[idx]  +=  grad / self.train_batch_count

            loss, accuracy, _ = self._forward_backward(self.model_prev, inputs, targets, zero_grad=True, is_test=False)
            g_prev = [param.grad.detach().clone() for param in self.model_prev.parameters() if param.requires_grad]
            self.model_prev.load_state_dict(self.model.state_dict())
            
            for idx, _ in enumerate(self._g_ref):
                self._g_ref[idx].add_((g_cur[idx] - g_prev[idx]))

            with torch.no_grad():
                for param, grad_ref in zip([p for p in self.model.parameters() if p.requires_grad], self._g_ref):
                    param.add_(grad_ref, alpha=-lr)
        self._g_ref = [grad.clone() for grad in self._g_avg]
        self._g_avg = [torch.zeros_like(param) for param in self.model.parameters() if param.requires_grad] # v_tilde


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, required=True, choices=['sgd', 'svrg', 'nfg_svrg', 'nfg_sarah', 'nfg_sarah_non'])
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()


set_seed(152)
DEVICE = get_device(args.device)
BATCH_SIZE = 256
from models import get_swin_tiny_from_timm
MODEL = get_swin_tiny_from_timm

data_loaders = get_data_loaders(BATCH_SIZE)
EPOCHS = 20
LAMBDA_VALUE =  1e-8
from utils import get_warmup_cosine_scheduler

if args.method == 'sgd':
    opt = SGD(
        model=MODEL(DEVICE), 
        data_loaders=data_loaders, 
        loss_fn=torch.nn.CrossEntropyLoss(), 
        device=DEVICE, 
        lambda_value=LAMBDA_VALUE,
        scheduler_config = {
            'warmup_epochs': 4,
            'base_lr': 0.05,
            'warmup_lr': 0.005 ,
            'min_lr': 0.005,
        }
    )
    opt.run(
        EPOCHS, 
        args.lr,
        scheduler_func=get_warmup_cosine_scheduler
    )
    opt.dump_json()
elif args.method == 'svrg':
    opt = SVRG_Pick(
        model=MODEL(DEVICE), 
        model_ref=MODEL(DEVICE), 
        data_loaders=data_loaders, 
        loss_fn=torch.nn.CrossEntropyLoss(), 
        device=DEVICE, 
        lambda_value=LAMBDA_VALUE, 
        freq=1,
        scheduler_config = {
            'warmup_epochs': 3,
            'base_lr': 0.05,
            'warmup_lr': 0.005 ,
            'min_lr': 1e-5,
        }
    )
    opt.run(
        EPOCHS // 3 + 1, 
        args.lr, 
        scheduler_func=get_warmup_cosine_scheduler
    )
    opt.dump_json()
elif args.method == 'nfg_svrg':
    opt = NFGSVRG_AVG(
        model=MODEL(DEVICE), 
        model_ref=MODEL(DEVICE), 
        model_buffer=MODEL(DEVICE), 
        data_loaders=data_loaders, 
        loss_fn=torch.nn.CrossEntropyLoss(), 
        device=DEVICE, 
        lambda_value=LAMBDA_VALUE, 
        scheduler_config = {
            'warmup_epochs': 5,
            'base_lr': 0.05,
            'warmup_lr': 0.005 ,
            'min_lr': 1e-5,
        }
    )
    opt.run(
        EPOCHS // 2 + 1, 
        args.lr, 
        scheduler_func=get_warmup_cosine_scheduler
    )
    opt.dump_json()
elif args.method == 'nfg_sarah':
    opt = NFGSARAH(
        model=MODEL(DEVICE), 
        model_prev=MODEL(DEVICE), 
        data_loaders=data_loaders, 
        loss_fn=torch.nn.CrossEntropyLoss(), 
        device=DEVICE, 
        lambda_value=LAMBDA_VALUE, 
        scheduler_config = {
            'warmup_epochs': 5,
            'base_lr': 0.005,
            'warmup_lr': 0.0005 ,
            'min_lr': 1e-6,
        }
    )
    opt.run(
        EPOCHS // 2 + 1, 
        args.lr, 
        scheduler_func=get_warmup_cosine_scheduler
    )
    opt.dump_json()
elif args.method == 'nfg_sarah_non':
    opt = NFGSARAHnon(
        model=MODEL(DEVICE), 
        model_prev=MODEL(DEVICE), 
        data_loaders=data_loaders, 
        loss_fn=torch.nn.CrossEntropyLoss(), 
        device=DEVICE, 
        lambda_value=LAMBDA_VALUE, 
        scheduler_config = {
            'warmup_epochs': 5,
            'base_lr': 0.005,
            'warmup_lr': 0.0005 ,
            'min_lr': 1e-6,
        }
    )
    opt.run(
        EPOCHS // 2 + 1, 
        args.lr, 
        scheduler_func=get_warmup_cosine_scheduler
    )
    opt.dump_json()