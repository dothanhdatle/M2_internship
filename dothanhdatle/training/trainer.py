from utils.baseline import baseline_prediction_data, mse_miss_recon

class Trainer:
    def __init__(self, model_type, model, data, data_loader, optimizer, preprocessor = None):
        self.model_type = model_type
        self.model = model
        self.data = data
        self.preprocessor = preprocessor
        self.data_loader = data_loader
        self.optimizer = optimizer
        if preprocessor != None:
            self.preprocessed_data = self.preprocessor.process(self.data)
            self.train_gt, self.test_gt = self.data_loader.data_split(self.preprocessed_data)
            self.train_miss, self.test_miss = self.data_loader.create_missing(self.train_gt, self.test_gt)
        else:
            self.train_gt = self.data_loader.train_gt
            self.test_gt = self.data_loader.test_gt
            self.train_miss = self.data_loader.train_miss
            self.test_miss = self.data_loader.test_miss

        self.train_loader = self.data_loader.load_train(self.train_miss,self.model_type)

    def train(self, nb_epochs, device, print_rate, 
              lr_decay = False, step_size= 10, factor = 0.1, model_save_path='saved_models/best_model.pth'):
        
        
        loss_train, train_loss_dict, \
            mse_train, best_epoch = self.model.train_(self.train_loader, train_data=self.train_miss, train_gt=self.train_gt, 
                                                      optimizer=self.optimizer, nb_epochs=nb_epochs,
                                                      device=device, print_rate = print_rate, 
                                                      lr_decay = lr_decay, step_size = step_size, factor = factor, 
                                                      model_save_path=model_save_path)
        return loss_train, train_loss_dict, mse_train, best_epoch

    def test(self,device):
        mse_test, mse_test_df = self.model.test_(self.test_miss, self.test_gt, device)
        return mse_test, mse_test_df
    
    def baseline_mse_train(self):
        rec_base_train = baseline_prediction_data(self.train_miss)
        mse_base_train, mse_base_train_df = mse_miss_recon(rec_base_train, self.train_gt)
        return rec_base_train, mse_base_train, mse_base_train_df
    
    def baseline_mse_test(self):
        rec_base_test = baseline_prediction_data(self.test_miss)
        mse_base_test, mse_base_test_df = mse_miss_recon(rec_base_test, self.test_gt)
        return rec_base_test, mse_base_test, mse_base_test_df
        

    