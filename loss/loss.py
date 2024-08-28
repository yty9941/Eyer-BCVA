import torch
import torch.nn.functional as F
from torch import nn

class ContrastiveLoss(nn.Module):
    def __init__(self, cfgs):
        super(ContrastiveLoss, self).__init__()
        self.temperature = cfgs['train_cfg']['incomplete']['tau']
        self.Lambda = cfgs['train_cfg']['incomplete']['lambda']
        self.isLossMask = cfgs['base_cfg']['isLossMask']

    def forward(self, image, text, missModalTag, type):
        assert type == "OCT" or type == "SLO", "Contrastive Learning type is error!!!"
        if type == "OCT":
            mask =  missModalTag[:, 1]
        else:
            mask =  missModalTag[:, 2]
        image, text = image.mean(1), text.mean(1)
        image = F.normalize(image, p = 2, dim = 1)
        text = F.normalize(text, p = 2, dim = 1)
        batch_size = image.shape[0]

        similarity_matrix = torch.exp(F.cosine_similarity(image.unsqueeze(1), text.unsqueeze(0), dim = 2) / self.temperature)

        sum_row = torch.sum(similarity_matrix, dim = 1)
        sum_col = torch.sum(similarity_matrix, dim = 0)

        loss_it = torch.diag(- torch.log(torch.div(similarity_matrix, sum_row[:, None])))
        loss_ti = torch.diag(- torch.log(torch.div(similarity_matrix, sum_col[:, None])))
        if self.isLossMask:
            loss = torch.sum(mask * (self.Lambda * loss_ti + (1 - self.Lambda) * loss_it)) / batch_size
        else:
            loss = torch.sum(self.Lambda * loss_ti + (1 - self.Lambda) * loss_it) / batch_size
        return loss

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        mse = torch.mean(torch.square(y_pred - y_true))
        return mse






