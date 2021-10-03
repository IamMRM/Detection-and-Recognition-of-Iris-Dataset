from os.path import join, exists
from os import makedirs

import cv2
import torch.cuda
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torchvision.utils import save_image
from sklearn.metrics import f1_score
import wandb
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt

from u_net_model import UNet
from utils import save_model, load_model, save_pickle_file, load_pickle_file


class Trainer:

    def __init__(self, train_data_loader, test_data_loader, save_dir, run_id, loss_function=None, model=None):
        self.config_dict = {
            'lr': 10 ** -4,
            'epochs': 60,
            'class_threshold': 0.5,
            'resume': 'allow'
        }
        self.run_id = run_id
        self.save_dir = save_dir
        self.current_save_dir = join(save_dir, run_id)
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

        if loss_function is None:
            self.segmentation_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2], device=self.device))
        else:
            self.segmentation_loss = loss_function

        if model is None:
            self.model = UNet()
        else:
            self.model = model

        self.optimizer = optim.Adam(self.model.parameters(), self.config_dict['lr'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.3, patience=10,
                                                              verbose=True, threshold=0.05, min_lr=10 ** -6)

        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

    def forward_pass_batch(self, images, masks):
        weights_masks = []
        for i in range(masks.shape[0]):
            weights_masks.append(torch.tensor(Trainer.get_weighted_loss(masks[i].clone().detach())))
        weights_masks_tensor = torch.stack(weights_masks, dim=0)

        images = images.to(self.device)
        masks = masks.to(self.device)
        weights_masks_tensor = weights_masks_tensor.to(self.device)

        out_masks, predicted_masks = self.model(images)
        loss = self.segmentation_loss(out_masks, masks)

        if len(loss.shape) != 0:
            loss = torch.mean(loss * weights_masks_tensor)

        assert loss > 0
        return predicted_masks, loss

    @staticmethod
    def get_weighted_loss(image):
        from skimage.morphology import erosion, square
        import matplotlib.pyplot as plt

        depth_level = 3
        temp_image = np.squeeze(image.numpy())
        weight_mask = np.zeros_like(temp_image) + 0.5
        for level in range(depth_level):
            eroded = erosion(temp_image, square(7))
            diff = temp_image - eroded
            weight_mask = np.where(diff != 0, 1 - level * 0.1, weight_mask)

            temp_image -= diff

        # plt.imshow(weight_mask, cmap='gray')
        # plt.show()

        return weight_mask

    def train_epoch(self, epoch):
        progress_bar = tqdm(total=len(self.train_data_loader),
                            desc=f'Batches/Epoch {epoch}')

        running_loss = 0
        running_f1_score = 0
        batch_id = 1
        for images_batch, masks_batch in self.train_data_loader:
            predicted_masks, loss = self.forward_pass_batch(images_batch, masks_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.detach().item()

            f1_batch_score = self.evaluate_batch(masks_batch, predicted_masks)
            running_f1_score += f1_batch_score

            wandb.log({'batch_loss': loss.detach().item(), 'f1_batch': f1_batch_score})

            if batch_id % 5 == 0:
                self.save_batch_output(images=images_batch,
                                       true_masks=masks_batch,
                                       predicted_masks=predicted_masks,
                                       batch_id=batch_id,
                                       epoch=epoch)

            batch_id += 1
            progress_bar.update(1)

        progress_bar.close()

        epoch_loss = running_loss / len(self.train_data_loader)
        epoch_f1_score = running_f1_score / len(self.train_data_loader)
        return epoch_loss, epoch_f1_score

    @torch.no_grad()
    def validate_epoch(self):
        progress_bar = tqdm(total=len(self.test_data_loader),
                            desc=f'Batches/Validate')

        running_loss = 0
        running_f1_score = 0

        for images_batch, masks_batch in self.test_data_loader:
            predicted_masks, loss = self.forward_pass_batch(images_batch, masks_batch)

            running_loss += loss.detach().item()

            f1_batch_score = self.evaluate_batch(masks_batch, predicted_masks)
            running_f1_score += f1_batch_score

            progress_bar.update(1)

        progress_bar.close()

        epoch_loss = running_loss / len(self.test_data_loader)
        epoch_f1_score = running_f1_score / len(self.test_data_loader)
        return epoch_loss, epoch_f1_score

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
            epoch_train_loss, epoch_train_f1_score = self.train_epoch(epoch)
            epoch_validate_loss, epoch_validate_f1_score = self.validate_epoch()

            self.scheduler.step(epoch_validate_loss)

            wandb.log({'train_loss': epoch_train_loss,
                       'train_f1': epoch_train_f1_score,
                       'test_loss': epoch_validate_loss,
                       'test_f1': epoch_validate_f1_score,
                       'lr': Trainer.get_lr(self.optimizer),
                       'epoch': epoch})

            self.save_training_status(epoch)
        run.finish()

    def evaluate_batch(self, true_masks, predicted_masks):
        true_masks = true_masks.detach().cpu().numpy()
        predicted_masks = predicted_masks.detach().cpu().numpy()

        batch_size = true_masks.shape[0]
        running_f1_score = 0
        for image_id in range(batch_size):
            f1 = f1_score(true_masks[image_id].reshape(-1),
                          (predicted_masks[image_id].reshape(-1) > self.config_dict['class_threshold']).astype(np.int8))
            running_f1_score += f1

        return running_f1_score / batch_size

    # def save_batch_output(self, true_masks, predicted_masks, batch_id, epoch):
    #     true_masks = true_masks.detach().cpu()
    #     predicted_masks = predicted_masks.detach().cpu()
    #
    #     # predicted_masks_grid = make_grid(predicted_masks, nrow=3)
    #     # true_masks_grid = make_grid(true_masks, nrow=3)
    #     images_path = join(self.current_save_dir, 'images')
    #     makedirs(images_path, exist_ok=True)
    #
    #     true_mask_path = join(images_path, f'true_{epoch}_{batch_id}.jpg')
    #     predicted_mask_path = join(images_path, f'pred_{epoch}_{batch_id}.jpg')
    #
    #     save_image(true_masks, fp=true_mask_path)
    #     save_image(predicted_masks, fp=predicted_mask_path)

    def save_batch_output(self, images, true_masks, predicted_masks, batch_id, epoch):
        images = images.detach().cpu()
        true_masks = true_masks.detach().cpu()
        predicted_masks = predicted_masks.detach().cpu()

        # predicted_masks_grid = make_grid(predicted_masks, nrow=3)
        # true_masks_grid = make_grid(true_masks, nrow=3)
        images_path = join(self.current_save_dir, 'images')
        makedirs(images_path, exist_ok=True)

        image_path = join(images_path, f'img_{epoch}_{batch_id}.jpg')
        alpha = 0.3
        true_colored_mask = true_masks.repeat_interleave(repeats=3, dim=1) * torch.Tensor([1, 0, 0]).reshape(
            (1, 3, 1, 1))
        predicted_colored_masks = predicted_masks.repeat_interleave(repeats=3, dim=1) * torch.Tensor([0, 1, 0]).reshape(
            (1, 3, 1, 1))

        output_image = (1 - alpha) * images + torch.multiply(true_colored_mask, alpha) + torch.multiply(
            predicted_colored_masks, alpha)

        # save_image(true_masks, fp=true_mask_path)
        # save_image(predicted_masks, fp=predicted_mask_path)

        save_image(output_image, fp=image_path)

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
    from iris_dataset import IrisDataset
    from iru_net import IRUNet

    import pandas as pd
    from torch.utils.data import DataLoader

    b = '/home/mahmoudk/iris/dataset'
    p = '/home/mahmoudk/iris/dataset/labels.csv'
    s = '/home/mahmoudk/iris/save_dir'
    run_id = 'weighted_element_loss_weighted_loss_diff_classes_V2'
    labels_df = pd.read_csv(p)
    train_df, test_df = train_test_split(labels_df, test_size=0.2, random_state=41, shuffle=True,
                                         stratify=labels_df['classes'].values)

    # train_df = labels_df[labels_df['classes'] <= 112]
    # test_df = labels_df[labels_df['classes'] > 112]

    loss_function = nn.BCEWithLogitsLoss()
    model = IRUNet()
    train_data_loader = DataLoader(IrisDataset(b, train_df), batch_size=8, shuffle=True, num_workers=3)
    test_data_loader = DataLoader(IrisDataset(b, test_df), batch_size=8, shuffle=False, num_workers=3)
    trainer = Trainer(train_data_loader, test_data_loader, save_dir=s, run_id="web_project_deployment2",
                      loss_function=loss_function, model=model)
    trainer.train()

    # loss_functions_list = [
    #     nn.BCEWithLogitsLoss(),
    #     nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2], device='cuda:0')),
    #     nn.BCEWithLogitsLoss(
    #         pos_weight=torch.tensor([2], device='cuda:0'),
    #         reduction='none')]
    #
    # run_id_names = ['t_IRU_baseline_diff_classes_V1',
    #                 't_IRU_weighted_loss_diff_classes_V1',
    #                 't_IRU_weighted_element_loss_weighted_loss_diff_classes_V1']
    #
    # import albumentations as A
    # from albumentations.pytorch import ToTensorV2
    #
    # train_transform = A.Compose([A.Flip(p=0.5),
    #                              A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=10, interpolation=cv2.INTER_CUBIC,
    #                                                 p=0.5),
    #                              A.CenterCrop(height=200, width=280, p=0.5),
    #                              A.Resize(height=240, width=320, always_apply=True, interpolation=cv2.INTER_CUBIC),
    #                              ToTensorV2()], additional_targets={'image1': 'image'})
    #
    # for run_id, loss_function in zip(run_id_names, loss_functions_list):
    #     model = IRUNet()
    #     train_data_loader = DataLoader(IrisDataset(b, train_df, transform=train_transform),
    #                                    batch_size=8, shuffle=True, num_workers=3)
    #     test_data_loader = DataLoader(IrisDataset(b, test_df), batch_size=8, shuffle=False, num_workers=3)
    #     trainer = Trainer(train_data_loader, test_data_loader, save_dir=s, run_id=run_id, loss_function=loss_function,
    #                       model=model)
    #     trainer.train()
