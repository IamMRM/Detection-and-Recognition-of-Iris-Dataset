from os.path import join, exists
from os import makedirs

import torch.cuda
from torch import nn, optim
from torchvision.utils import save_image, make_grid
from sklearn.metrics import f1_score
import wandb
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt

from siamese_net import SiameseNet
from contrastive_loss import ContrastiveLoss
from utils import save_model, load_model, save_pickle_file, load_pickle_file


class SiameseTrainer:

    def __init__(self, train_data_loader, test_data_loader, save_dir, run_id,
                 test_evaluation_data_loader=None,
                 train_evaluation_data_loader=None,
                 config_dict=None):
        if config_dict is None:
            self.config_dict = {
                'lr': 10 ** -4,
                'patience': 5,
                'gamma': 0.7,
                'epochs': 40,
                'resume': 'allow',
                'margin': 2
            }
        else:
            self.config_dict = config_dict
        self.run_id = run_id
        self.save_dir = save_dir
        self.current_save_dir = join(save_dir, run_id)
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

        self.loss_function = ContrastiveLoss(self.config_dict['margin'])
        self.model = SiameseNet()
        self.optimizer = optim.Adam(self.model.parameters(), self.config_dict['lr'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                              factor=self.config_dict['gamma'],
                                                              patience=self.config_dict['patience'],
                                                              verbose=True, threshold=0.05, min_lr=10 ** -6)

        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.test_evaluation_data_loader = test_evaluation_data_loader
        self.train_evaluation_data_loader = train_evaluation_data_loader

    def forward_pass_batch(self, images1, images2, labels):
        images1 = images1.to(self.device)
        images2 = images2.to(self.device)
        labels = labels.to(self.device)

        out1, out2 = self.model(images1, images2)
        loss = self.loss_function(out1, out2, labels)

        assert loss >= 0, f'The loss is: {loss}'
        if loss == 0:
            print("loss is zero")
        return out1, out2, loss

    def train_epoch(self, epoch):
        progress_bar = tqdm(total=len(self.train_data_loader),
                            desc=f'Batches/Epoch {epoch}')

        running_loss = 0
        batch_id = 1

        for images1_batch, images2_batch, labels_batch in self.train_data_loader:
            out1, out2, loss = self.forward_pass_batch(images1_batch, images2_batch, labels_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.detach().item()

            wandb.log({'batch_loss': loss.detach().item()})

            batch_id += 1
            progress_bar.update(1)

        progress_bar.close()

        epoch_loss = running_loss / len(self.train_data_loader)
        return epoch_loss

    @staticmethod
    def get_euclidean_distance(x1, x2):
        return torch.sqrt(torch.sum(torch.pow(x1 - x2, 2), dim=1, keepdim=True))

    @torch.no_grad()
    def validate_epoch(self, epoch=1):
        if epoch == 20:
            print('hi')

        progress_bar = tqdm(total=len(self.test_data_loader),
                            desc=f'Batches/Validate Loss')

        running_loss = 0
        batch_id = 1

        for images1_batch, images2_batch, labels_batch in self.test_data_loader:
            images1_batch = images1_batch.to(self.device)
            images2_batch = images2_batch.to(self.device)
            labels_batch = labels_batch.to(self.device)

            out1, out2 = self.model(images1_batch, images2_batch)
            loss = self.loss_function(out1, out2, labels_batch)
            running_loss += loss.detach().item()

            if batch_id % 5 == 0:
                self.save_batch_output(images1=images1_batch,
                                       images2=images2_batch,
                                       similarity_distance=SiameseTrainer.get_euclidean_distance(out1, out2),
                                       labels=labels_batch,
                                       batch_id=batch_id,
                                       epoch=epoch)
            batch_id += 1
            progress_bar.update(1)

        progress_bar.close()

        epoch_loss = running_loss / len(self.test_data_loader)
        return epoch_loss

    def train(self):
        run = wandb.init(project='Iris Segmentation',
                         id=self.run_id,
                         resume=self.config_dict['resume'],
                         config=self.config_dict,
                         reinit=True)

        if wandb.run.resumed:
            status_dict = self.load_training_status()
            starting_epoch = status_dict['epoch'] + 1
        else:
            starting_epoch = 0

        self.model.to(self.device)

        for epoch in range(starting_epoch, self.config_dict['epochs']):
            epoch_train_loss = self.train_epoch(epoch)
            epoch_validate_loss = self.validate_epoch(epoch)

            epoch_validate_f1_score = self.evaluate_epoch(self.test_evaluation_data_loader)
            epoch_train_f1_score = self.evaluate_epoch(self.train_evaluation_data_loader)

            self.scheduler.step(epoch_validate_loss)

            wandb.log({'loss/train': epoch_train_loss,
                       'loss/test': epoch_validate_loss,
                       'f1/train': epoch_train_f1_score,
                       'f1/test': epoch_validate_f1_score,
                       'lr': SiameseTrainer.get_lr(self.optimizer),
                       'epoch': epoch})

            self.save_training_status(epoch)
        run.finish()

    # def evaluate_epoch(self, outputs_tensor, classes_tensor):
    #     distance_matrix = torch.cdist(outputs_tensor, outputs_tensor, p=2)
    #     values, indices = torch.topk(distance_matrix, k=10, largest=False)
    #     predictions = torch.zeros_like(distance_matrix)
    #     for i in range(indices.shape[0]):
    #         for j in range(indices.shape[1]):
    #             row_index = indices[i, j]
    #             predictions[i, row_index] = 1
    #
    #     temp = classes_tensor.reshape(-1, 1) - classes_tensor.reshape(-1)
    #     ground_truth = torch.where(temp == 0, 1, 0)
    #
    #     assert predictions.shape == ground_truth.shape
    #     validation_f1_score = f1_score(y_true=ground_truth.reshape(-1), y_pred=predictions.reshape(-1), pos_label=1)
    #
    #     return validation_f1_score

    # @torch.no_grad()
    # def evaluate_epoch(self, eval_dataloader):
    #
    #     original_features_list = []
    #     original_classes_list = []
    #     target_features_list = []
    #     target_classes_list = []
    #
    #     progress_bar = tqdm(total=len(eval_dataloader),
    #                         desc=f'Batches/Evaluation: ')
    #
    #     for image1, image2, labels, classes in eval_dataloader:
    #         out1, out2 = self.model(image1.to(self.device), image2.to(self.device))
    #         out1, out2 = out1.cpu(), out2.cpu()
    #
    #         original_features_list.append(out1)
    #         original_classes_list.append(classes)
    #         target_features_list.append(out2)
    #         target_classes_list.append(classes)
    #
    #         progress_bar.update(1)
    #
    #     progress_bar.close()
    #
    #     original_features_tensor = torch.cat(original_features_list, dim=0)
    #     original_classes_tensor = torch.cat(original_classes_list, dim=0)
    #     target_features_tensor = torch.cat(target_features_list, dim=0)
    #     target_classes_tensor = torch.cat(target_classes_list, dim=0)
    #
    #     distance_matrix = torch.cdist(target_features_tensor, original_features_tensor, p=2)
    #
    #     # print(f'Distance matrix {distance_matrix.shape}: {distance_matrix}')
    #
    #     values, indices = torch.topk(distance_matrix, k=1, largest=False)
    #
    #     # print(f'Values {values.shape}: {values}\nIndices {indices.shape}: {indices}')
    #
    #     predictions = torch.zeros_like(target_classes_tensor)
    #     for i in range(indices.shape[0]):
    #         predictions[i] = 1 if indices[i] == i else 0
    #
    #     # print(f'Predictions {predictions.shape}: {predictions}')
    #
    #     ground_truth = torch.ones_like(predictions)
    #
    #     # print(f'Ground truth {ground_truth.shape}: {ground_truth}')
    #
    #     assert predictions.shape == ground_truth.shape
    #     f1 = f1_score(y_true=ground_truth.reshape(-1), y_pred=predictions.reshape(-1), pos_label=1)
    #     # print(f'F1: {validation_f1_score}')
    #     return f1

    @torch.no_grad()
    def evaluate_epoch(self, eval_dataloader):

        original_features_list = []
        original_classes_list = []
        target_features_list = []
        target_classes_list = []

        progress_bar = tqdm(total=len(eval_dataloader),
                            desc=f'Batches/Evaluation: ')

        for image1, image2, labels, classes in eval_dataloader:
            out1, out2 = self.model(image1.to(self.device), image2.to(self.device))
            out1, out2 = out1.cpu(), out2.cpu()

            original_features_list.append(out1)
            original_classes_list.append(classes)
            target_features_list.append(out2)
            target_classes_list.append(classes)

            progress_bar.update(1)

        progress_bar.close()

        original_features_tensor = torch.cat(original_features_list, dim=0)
        original_classes_tensor = torch.cat(original_classes_list, dim=0)
        target_features_tensor = torch.cat(target_features_list, dim=0)
        target_classes_tensor = torch.cat(target_classes_list, dim=0)

        distance_matrix = torch.cdist(target_features_tensor, original_features_tensor, p=2)

        # print(f'Distance matrix {distance_matrix.shape}: {distance_matrix}')

        values, indices = torch.topk(distance_matrix, k=1, largest=False)

        # print(f'Values {values.shape}: {values}\nIndices {indices.shape}: {indices}')

        predictions = torch.zeros_like(target_classes_tensor)
        for i in range(indices.shape[0]):
            predictions[i] = 1 if indices[i] == i else 0

        # print(f'Predictions {predictions.shape}: {predictions}')

        ground_truth = torch.ones_like(predictions)

        # print(f'Ground truth {ground_truth.shape}: {ground_truth}')

        assert predictions.shape == ground_truth.shape
        f1 = f1_score(y_true=ground_truth.reshape(-1), y_pred=predictions.reshape(-1), pos_label=1)
        # print(f'F1: {validation_f1_score}')
        return f1

    def save_batch_output(self, images1, images2, similarity_distance, labels, batch_id, epoch):
        images1 = images1.detach().cpu()
        images2 = images2.detach().cpu()
        similarity_distance = similarity_distance.detach().cpu()
        labels = labels.detach().cpu()

        output_images_path = join(self.current_save_dir, 'images')
        makedirs(output_images_path, exist_ok=True)

        labels_string = '-'.join([f'd{d.item():.2f}' for l, d in zip(labels, similarity_distance)])
        labels_mask = torch.where(labels == 1, torch.tensor([0, 1, 0]), torch.tensor([1, 0, 0]))

        output_tensor = torch.cat([images1, images2], dim=3)
        output_tensor = 0.7 * output_tensor + 0.3 * labels_mask.reshape((*labels_mask.shape, 1, 1))

        save_image(output_tensor, fp=join(output_images_path, f'img_{epoch}_{batch_id} {labels_string}.jpg'), nrow=2)

    def save_training_status(self, epoch):
        current_saving_path = self.current_save_dir
        models_path = join(current_saving_path, 'models')
        dict_path = join(current_saving_path, 'status.pkl')
        makedirs(models_path, exist_ok=True)

        model_path = join(models_path, f'model_{epoch}.pth')
        save_model(self.model, model_path)

        status_dict = {'epoch': epoch,
                       'optimizer': self.optimizer,
                       'scheduler': self.scheduler}
        save_pickle_file(dict_path, status_dict)

    def load_training_status(self, old_saving_dir=None):
        current_saving_path = self.current_save_dir if old_saving_dir is None else old_saving_dir
        models_path = join(current_saving_path, 'models')
        dict_path = join(current_saving_path, 'status.pkl')

        if not exists(dict_path):
            return {'epoch': -1}

        status_dict = load_pickle_file(dict_path)
        epoch = status_dict['epoch']
        self.optimizer = status_dict['optimizer']
        self.scheduler = status_dict['scheduler']

        model_path = join(models_path, f'model_{epoch}.pth')
        load_model(self.model, model_path)

        return status_dict

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']


if __name__ == '__main__':
    from iris_train_dataset import SiameseIrisTrainDataset
    from iris_test_dataset import SiameseIrisTestDataset
    from siamese_net import SiameseNet

    import pandas as pd
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split

    b = '/home/mahmoudk/iris/dataset'
    p = '/home/mahmoudk/iris/dataset/labels.csv'
    s = '/home/mahmoudk/iris/save_dir'
    run_id = 'test_siamese_V0'
    labels_df = pd.read_csv(p)

    train_df, test_df = train_test_split(labels_df, test_size=0.2, random_state=41, shuffle=True,
                                         stratify=labels_df['classes'].values)

    # train_df = labels_df[labels_df['classes'] <= 112]
    # test_df = labels_df[labels_df['classes'] > 112]

    train_data_loader = DataLoader(SiameseIrisTrainDataset(b, train_df), batch_size=8, shuffle=True, num_workers=3)
    test_data_loader = DataLoader(SiameseIrisTrainDataset(b, test_df), batch_size=8, shuffle=False, num_workers=3)
    train_eval_data_loader = DataLoader(SiameseIrisTestDataset(b, train_df), batch_size=8, shuffle=False, num_workers=3)
    test_eval_data_loader = DataLoader(SiameseIrisTestDataset(b, test_df), batch_size=8, shuffle=False, num_workers=3)

    lr_list = [10**-3, 10**-4, 10**-5]
    margins_list = [2, 3]
    config = {
        'lr': 10 ** -4,
        'patience': 5,
        'gamma': 0.7,
        'epochs': 40,
        'resume': 'allow',
        'margin': 2
    }

    # for margin in margins_list:
    #     for lr in lr_list:
    #         config['lr'] = lr
    #         config['margin'] = margin
    #
    #         run_id = f's_model2_exploration_lr_{lr},margin_{margin}'
    #
    #         trainer = SiameseTrainer(train_data_loader,
    #                                  test_data_loader,
    #                                  save_dir=s,
    #                                  run_id=run_id,
    #                                  test_evaluation_data_loader=test_eval_data_loader,
    #                                  train_evaluation_data_loader=train_eval_data_loader,
    #                                  config_dict=config)
    #         try:
    #             trainer.train()
    #         except:
    #             print('Exception')

    config = {
        'lr': 10 ** -3,
        'patience': 7,
        'gamma': 0.7,
        'epochs': 80,
        'resume': 'allow',
        'margin': 2
    }
    run_id = f"Siamese_train_without_sigmoid_128_V5"

    trainer = SiameseTrainer(train_data_loader,
                             test_data_loader,
                             save_dir=s,
                             run_id=run_id,
                             test_evaluation_data_loader=test_eval_data_loader,
                             train_evaluation_data_loader=train_eval_data_loader,
                             config_dict=config)
    trainer.train()
