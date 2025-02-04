from utils import set_seed
from utils import get_device 
from utils import get_resnet18
from utils import Logger
from utils import get_data
import torch
from tqdm import tqdm
import json
import os

class SGD():
    def __init__(self, model, data, loss_fn):
        self.model = model
        self.train_data = data[0]
        self.test_data = data[1]
        self.train_batch_count = len(self.train_data)
        self.test_batch_count = len(self.test_data)
        self.loss_fn = loss_fn
        self.loggers = {}
        self.train_logger = Logger()
        self.test_logger = Logger()
        self.grads_epochs_computed = 0

    def run(self, epochs, lr, exp_name=None):
        if exp_name is None:
            exp_name = f'SGD_{lr=}_{epochs=}'
        for _ in tqdm(range(epochs), desc=exp_name, ncols=100):
            self._train_epoch(lr)
            tqdm.write(f'Train Loss: {self.train_logger.loss[-1]:.4f},\t Train Accuracy: {self.train_logger.accuracy[-1]:.4f}')
            self._test_epoch()
            tqdm.write(f'Test  Loss: {self.test_logger.loss[-1]:.4f},\t Test  Accuracy: {self.test_logger.accuracy[-1]:.4f}')
            tqdm.write(f'Grads epochs computed: {self.grads_epochs_computed:.2f}')
        self.loggers[exp_name] = {
            'train': self.train_logger.to_dict(),
            'test': self.test_logger.to_dict()
        }
        self.test_logger = Logger()
        self.train_logger = Logger()

        
    def _forward_backward(self, model, inputs, targets, zero_grad=True, is_test=False):
        if zero_grad:
            model.zero_grad()
        if is_test:
            model.eval()
        else:
            model.train()
        outputs = model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        if not is_test:
            self.grads_epochs_computed += 1/self.train_batch_count
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        batch_accuracy = correct / targets.size(0)
        return loss, batch_accuracy

    def _train_epoch(self, lr):
        train_loss = 0
        train_accuracy = 0
        for idx in tqdm(torch.randperm(self.train_batch_count).tolist(), desc='Training', ncols=100, leave=False):
            inputs, targets = self.train_data[idx]
            loss, accuracy = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
            train_loss += loss.item()
            train_accuracy += accuracy
            for param in self.model.parameters():
                param.data.add_(param.grad, alpha=-lr)
        self.train_logger.append(train_loss/self.train_batch_count, train_accuracy/self.train_batch_count, self.grads_epochs_computed)
                
    def _test_epoch(self):
        test_loss = 0
        test_accuracy = 0
        for inputs, targets in tqdm(self.test_data, desc='Testing', ncols=100, leave=False):
            loss, accuracy = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=True)
            test_loss += loss.item()
            test_accuracy += accuracy
        self.test_logger.append(test_loss/self.test_batch_count, test_accuracy/self.test_batch_count, self.grads_epochs_computed)
    
    def dump_json(self, directory='experiments'):
        os.makedirs(directory, exist_ok=True)
        for exp_name, data in self.loggers.items():
            with open(f'{directory}/{exp_name}.json', 'w') as f:
                json.dump(data, f)


class SVRG():
    def __init__(self, model, model_ref, data, loss_fn, freq=3.5):
        self.model = model
        self.model_ref = model_ref
        self.train_data = data[0]
        self.test_data = data[1]
        self.train_batch_count = len(self.train_data)
        self.test_batch_count = len(self.test_data)
        self.loss_fn = loss_fn
        self.freq = freq
        self.p = 1/(freq*self.train_batch_count)
        self.loggers = {}
        self.train_logger = Logger()
        self.test_logger = Logger()
        self.grads_epochs_computed = 0
        self._g_ref = None

    def run(self, epochs, lr, exp_name=None):
        if exp_name is None:
            exp_name = f'SVRG_{lr=}_{self.freq=}_{epochs=}'
        for _ in tqdm(range(epochs), desc=exp_name, ncols=100):
            self._train_epoch(lr)
            tqdm.write(f'Train Loss: {self.train_logger.loss[-1]:.4f},\t Train Accuracy: {self.train_logger.accuracy[-1]:.4f}')
            self._test_epoch()
            tqdm.write(f'Test  Loss: {self.test_logger.loss[-1]:.4f},\t Test  Accuracy: {self.test_logger.accuracy[-1]:.4f}')
            tqdm.write(f'Grads epochs computed: {self.grads_epochs_computed:.2f}')
        self.loggers[exp_name] = {
            'train': self.train_logger.to_dict(),
            'test': self.test_logger.to_dict()
        }
        self.test_logger = Logger()
        self.train_logger = Logger()

        
    def _forward_backward(self, model, inputs, targets, zero_grad=True, is_test=False):
        if zero_grad:
            model.zero_grad()
        if is_test:
            model.eval()
        else:
            model.train()
        outputs = model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        if not is_test:
            self.grads_epochs_computed += 1/self.train_batch_count
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        batch_accuracy = correct / targets.size(0)
        return loss, batch_accuracy

    def _train_epoch(self, lr):
        for idx in tqdm(torch.randperm(self.train_batch_count).tolist(), desc='Training', ncols=100, leave=False):
            inputs, targets = self.train_data[idx]
            if torch.rand(1) < self.p or self._g_ref is None:
                self._g_ref = self._compute_full_grad()
                self.model_ref.load_state_dict(self.model.state_dict())
            loss, accuracy = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
            self.train_logger.append(loss.item(), accuracy, self.grads_epochs_computed)
            _, _ = self._forward_backward(self.model_ref, inputs, targets, zero_grad=True, is_test=False)
            for param, param_ref, grad_ref in zip(self.model.parameters(), self.model_ref.parameters(), self._g_ref):
                param.data.add_(param.grad - param_ref.grad + grad_ref, alpha=-lr)
                
    def _test_epoch(self):
        test_loss = 0
        test_accuracy = 0
        for inputs, targets in tqdm(self.test_data, desc='Testing', ncols=100, leave=False):
            loss, accuracy = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=True)
            test_loss += loss.item()
            test_accuracy += accuracy
        self.test_logger.append(test_loss/self.test_batch_count, test_accuracy/self.test_batch_count, self.grads_epochs_computed)

    def _compute_full_grad(self):
        self.model.zero_grad()
        for inputs, targets in tqdm(self.train_data, desc='Computing Full Gradient', ncols=100, leave=False):
            self._forward_backward(self.model, inputs, targets, zero_grad=False, is_test=False)
        return [param.grad.detach().clone()/self.train_batch_count for param in self.model.parameters()]
    
    def dump_json(self, directory='experiments'):
        os.makedirs(directory, exist_ok=True)
        for exp_name, data in self.loggers.items():
            with open(f'{directory}/{exp_name}.json', 'w') as f:
                json.dump(data, f)


set_seed(52)
DEVICE = get_device()
BATCH_SIZE = 128
EPOCHS = 40
FREQ = 3
LR = 0.05

# svrg = SVRG(
#     model=get_resnet18(DEVICE), 
#     model_ref=get_resnet18(DEVICE), 
#     data=get_data(BATCH_SIZE, DEVICE),
#     loss_fn=torch.nn.CrossEntropyLoss(), 
#     freq=3
#     )

# svrg.run(int(EPOCHS/(2+1/FREQ))+1, LR)

# svrg.dump_json()

sgd = SGD(
    model=get_resnet18(DEVICE), 
    data=get_data(BATCH_SIZE, DEVICE),
    loss_fn=torch.nn.CrossEntropyLoss()
    )

sgd.run(EPOCHS, LR)

sgd.dump_json()