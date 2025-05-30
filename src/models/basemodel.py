from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import os
import numpy as np
from h5py import File

class BaseModel:
    def get_trainer(self, **kwargs):
        raise NotImplementedError("Subclasses must implement the get_trainer method.")

class BaseTrainer:
    def __init__(self, model, **kwargs):
        """Initializes the trainer. To be extended by subclasses."""
        self.model = model
        self.is_trained = False

        self.train_loss_history = []
        self.val_loss_history = []

        self.hyperparameters = kwargs
        
    def fit(self, train_data, val_data=None):
        """Fits the model to the training data. To be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement the fit method.")
    
    def validate(self, val_data):
        """Validates the model on the validation data. To be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement the validate method.")
    
    def predict_proba(self, test_data):
        """Predicts probabilities for the test data. To be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement the predict_proba method.")
    
    def save_train_data(self, result_dir):
        os.makedirs(result_dir, exist_ok=True)
        data_path = os.path.join(result_dir, 'data.hdf5')
        with File(data_path, 'a') as f:
            if 'hyperparameters' not in f:
                for key, value in self.hyperparameters.items():
                    f.attrs[key] = str(value)
            
            if 'train_loss_history' not in f:
                f.create_dataset('train_loss_history', data=self.train_loss_history)
            elif len(f['train_loss_history']) < len(self.train_loss_history):
                f['train_loss_history'].resize(self.train_loss_history.shape)
                f['train_loss_history'][:] = self.train_loss_history
            
            if len(self.val_loss_history) > 0:
                if 'val_loss_history' not in f:
                    f.create_dataset('val_loss_history', data=self.val_loss_history)
                elif len(f['val_loss_history']) < len(self.val_loss_history):
                    f['val_loss_history'].resize(self.val_loss_history.shape)
                    f['val_loss_history'] = self.val_loss_history
    
    def save_test_data(self, result_dir, partition_id, labels, logits, loss):
        """Saves the predictions to a file."""
        os.makedirs(result_dir, exist_ok=True)
        data_path = os.path.join(result_dir, 'data.hdf5')
        with File(data_path, 'a') as f:
            f.attrs['hyperparameters'] = str(self.hyperparameters)
            partition = f.create_group(partition_id)
            partition.create_dataset('logits', data=logits)
            partition.create_dataset('labels', data=labels)
            partition.attrs['loss'] = loss
    
class TorchModel(BaseModel, nn.Module):

    def __init__(self, device='cpu', **kwargs):
        super().__init__()
        self.set_model_parameters(**kwargs)
        self.device = device
        self.to(device)

    def set_model_parameters(self):
        """Sets the model parameters. To be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement the set_model_parameters method.")

    def forward(self, x):
        """Performs a forward pass through the model. To be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement the forward method.")
    
    def get_trainer(self, **kwargs):
        return TorchTrainer(self, **kwargs)

class TorchTrainer(BaseTrainer):
    def __init__(self, model, optimizer, criterion, lr, weight_decay, batch_size, max_epochs, scheduler=None, gamma=None, milestones=None, transforms=None):
        super().__init__(model=model, 
                         optimizer=optimizer, 
                         criterion=criterion, 
                         lr=lr, 
                         weight_decay=weight_decay, 
                         batch_size=batch_size, 
                         max_epochs=max_epochs, 
                         scheduler=scheduler, 
                         gamma=gamma, 
                         milestones=milestones, 
                         transforms=transforms)

        self.optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = criterion()

        self.batch_size = batch_size
        self.max_epochs = max_epochs        

        self.scheduler = scheduler
        if self.scheduler and gamma and milestones:
            self.gamma = gamma
            self.milestones = milestones
            self.scheduler = self.scheduler(self.optimizer, gamma=self.gamma, milestones=self.milestones)

        self.epochs_trained = 0

    def extract_inputs(self, data):
        """Extracts inputs and labels from the data."""
        *inputs, labels = [_.to(self.model.device) for _ in data]
        return inputs, labels
    
    def extract_outputs(self, inputs):
        """Extracts outputs from the model given the inputs."""
        return self.model(*inputs)
    
    def train(self, train_loader):

        self.model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = self.extract_inputs(data)
            self.optimizer.zero_grad()
            outputs = self.extract_outputs(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        return train_loss
    
    def validate(self, dataloader):

        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = self.extract_inputs(data)
                outputs = self.extract_outputs(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
        val_loss = running_loss / len(dataloader)
        return val_loss
   
    def fit(self, train_data, val_data = None):
        train_loader = self.build_dataloader(train_data, shuffle=True)
        val_loader = self.build_dataloader(val_data, shuffle=False) if val_data else None

        if self.is_trained:
            raise RuntimeError("The model is already trained.")
        
        for _ in range(self.epochs_trained, self.max_epochs):

            train_loss = self.train(train_loader)
            self.train_loss_history.append(train_loss)
            self.epochs_trained += 1

            if val_loader:
                val_loss = self.validate(val_loader)
                self.val_loss_history.append(val_loss)

            if self.scheduler:
                self.scheduler.step()
 
        self.is_trained = True
        self.train_loss_history = np.array(self.train_loss_history)

        if val_loader:
            self.val_loss_history = np.array(self.val_loss_history)
            return self.train_loss_history, self.val_loss_history
        
        return self.train_loss_history
        
    def predict_proba(self, test_data):
        test_loader = self.build_dataloader(test_data, shuffle=False)
        
        y_true = []
        y_pred = []
        running_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = self.extract_inputs(data)
                logits = self.extract_outputs(inputs)
                loss = self.criterion(logits, labels)

                running_loss += loss.item()
                y_true.append(labels)
                y_pred.append(logits)
        
        running_loss /= len(test_loader)
        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()

        return y_true, y_pred, running_loss
    
    def save_train_data(self, result_dir):
        super().save_train_data(result_dir)
        model_path = os.path.join(result_dir, f'checkpoint.pth')
        torch.save(self.model.state_dict(), model_path)

    def load_train_data(self, result_dir):
        data_path = os.path.join(result_dir, 'data.hdf5')
        if os.path.exists(data_path):
            with File(data_path, 'r') as f:
                train_loss_history = list(f['train_loss_history'])
                
                if 'val_loss_history' in f:
                    val_loss_history = list(f['val_loss_history'])
                else:
                    val_loss_history = []

            epochs_trained = len(self.train_loss_history)
            model_path = os.path.join(result_dir, f'checkpoint.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Results, but no model found in {result_dir}. Unable to resume experiment.")
            self.model.load_state_dict(torch.load(model_path, map_location=self.model.device))
            self.epochs_trained = epochs_trained
            self.train_loss_history = train_loss_history
            self.val_loss_history = val_loss_history

    def build_dataloader(self, data, shuffle = True):
        return DataLoader(TensorDataset(*data.values()), batch_size=self.batch_size, shuffle=shuffle)

class ScikitModel(BaseModel):
    def __init__(self, clf, device='cpu', **kwargs):
        self.clf = clf(**kwargs)
    
    def __call__(self, x):
        """Performs a forward pass through the model."""
        return self.clf.decision_function(x)
    
    def get_trainer(self, **kwargs):
        return ScikitTrainer(self, **kwargs)
    
class ScikitTrainer(BaseTrainer):
    def __init__(self, model, criterion, n_components=None, transforms=None, **kwargs):
        super().__init__(model=model, n_components=n_components, transforms=transforms)
        
        self.model = model

        self.criterion = criterion()

        self.n_components = n_components
        self.transforms = transforms

        self.train_loss_history = []
        self.val_loss_history = []

    def validate(self, val_data):
        inputs, labels = val_data.values()

        if self.n_components:
            inputs = inputs(self.n_components)
        else:
            inputs = inputs.reshape(inputs.size(0), -1)

        inputs = inputs.cpu().numpy()
        logits = self.model(inputs)
        loss = self.criterion(torch.tensor(logits).to(labels.device), labels).item()
        return loss
    
    def fit(self, train_data, val_data=None):
        inputs, labels = train_data.values()
        if self.n_components:
            inputs = inputs(self.n_components)
        else:
            inputs = inputs.reshape(inputs.size(0), -1)
        inputs = inputs.cpu().numpy()

        self.model.clf.fit(inputs, labels.cpu().numpy())

        train_loss = self.validate(train_data)
        self.train_loss_history = np.array([train_loss])

        if val_data is not None:
            val_loss = self.validate(val_data)
            self.val_loss_history = np.array([val_loss])
            return self.train_loss_history, self.val_loss_history

        return self.train_loss_history
    
    def predict_proba(self, test_data):
        inputs, labels = test_data.values()
        if self.n_components:
            inputs = inputs(self.n_components)
        else:
            inputs = inputs.reshape(inputs.size(0), -1)
        inputs = inputs.cpu().numpy()

        logits = self.model(inputs)
        logits = torch.tensor(logits).to(labels.device)
        loss = self.criterion(logits, labels).item()

        return labels.cpu().numpy(), logits.cpu().numpy(), loss

class TorchGeometricModel(TorchModel):
    def forward(self, x, edge_index, edge_attr):
        """Performs a forward pass through the model. To be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement the forward method.")
    
    def get_trainer(self, optimizer, criterion, lr, weight_decay, batch_size, max_epochs, scheduler=None, gamma=None, milestones=None, transforms=None):
        return TorchGeometricTrainer(self, optimizer, criterion, lr, weight_decay, batch_size, max_epochs, scheduler=scheduler, gamma=gamma, milestones=milestones, transforms=transforms)
    
class TorchGeometricTrainer(TorchTrainer):
    def __init__(self, model, optimizer, criterion, lr, weight_decay, batch_size, max_epochs, scheduler=None, gamma=None, milestones=None, transforms=None):
        super().__init__(model=model, 
                         optimizer=optimizer, 
                         criterion=criterion, 
                         lr=lr, 
                         weight_decay=weight_decay, 
                         batch_size=batch_size, 
                         max_epochs=max_epochs,
                         scheduler=scheduler,
                         gamma=gamma,
                         milestones=milestones,
                         transforms=transforms)
        
    def extract_inputs(self, data):
        *inputs, labels = data.x, data.edge_index, None, data.y
        return inputs, labels

    def build_dataloader(self, data, shuffle=True):
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        EEG, edge_index, labels = data.values()
        data = [Data(x=EEG[i], edge_index=edge_index[i], y=labels[i]) for i in range(len(EEG))]
        return DataLoader(data, batch_size=self.batch_size, shuffle=shuffle)
        
