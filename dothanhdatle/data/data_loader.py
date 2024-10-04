import pandas as pd
from torch.utils.data import DataLoader
from data.data_sets import NIRS_Dataset_CVAE, NIRS_Dataset_DVAE, NIRS_Dataset, Batch_Sampler, collate_func, split_data, remove_data


class Data_Loader:
    def __init__(self, geno_column='GenoID', test_size=0.2, remove_percentage=0.2, batch_size=32, num_vars=10, random_state=42):
        self.geno_column = geno_column
        self.test_size = test_size
        self.remove_percentage = remove_percentage
        self.batch_size = batch_size
        self.num_vars = num_vars
        self.random_state = random_state

    def data_split(self, data):
        # Split the data into training and validation sets
        self.train_gt, self.test_gt = split_data(data, geno_column=self.geno_column, 
                                                 test_size=self.test_size, random_state=self.random_state)
        return self.train_gt, self.test_gt
    
    def create_missing(self, train_data, test_data):
        # Create missing datasets by removing a percentage of data
        self.train_miss = remove_data(train_data, percentage=self.remove_percentage)
        self.test_miss = remove_data(test_data, percentage=self.remove_percentage)
        return self.train_miss, self.test_miss

    def load_train(self, train_data, model_type):
        if model_type=='cvae':
            train_loader = self.cvae_dataloader(data=train_data, batch_size=self.batch_size)
        elif model_type=='dvae':
            train_loader = self.dvae_dataloader(data=train_data, batch_size=self.batch_size)
        elif model_type=='hvae':
            train_loader = self.group_dataloader(data=train_data, num_vars=self.num_vars, collate_func= collate_func)
        else:
            train_set = NIRS_Dataset(data=train_data)
            train_loader = DataLoader(train_set, self.batch_size, shuffle=True)        
        return train_loader
    
    def load_test(self, test_data, model_type):
        if model_type=='cvae':
            test_loader = self.cvae_dataloader(data=test_data, batch_size=self.batch_size)
        elif model_type=='dvae':
            test_loader = self.dvae_dataloader(data=test_data, batch_size=self.batch_size)
        elif model_type=='hvae':
            test_loader = self.group_dataloader(data=test_data, num_vars=self.num_vars, collate_func= collate_func)
        else:
            test_set = NIRS_Dataset(data=test_data)
            test_loader = DataLoader(test_set, self.batch_size, shuffle=True)        
        return test_loader

    def cvae_dataloader(self, file_path=None, data=None, batch_size=32, shuffle=True, one_hot_encode=True):
        data_set = NIRS_Dataset_CVAE(file_path=file_path, data=data, one_hot_encode=one_hot_encode)

        return DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)

    def dvae_dataloader(self, file_path = None, data=None, batch_size=32, shuffle=True):
        data_set = NIRS_Dataset_DVAE(file_path=file_path, data=data)
        return DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)

    def group_dataloader(self, file_path = None, data=None, num_vars = 10, collate_func = collate_func):
        data_set = NIRS_Dataset(file_path=file_path, data=data)
        batch_samp = Batch_Sampler(data_set, num_vars)
        return DataLoader(data_set, sampler=batch_samp, collate_fn=collate_func)