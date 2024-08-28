from model.IM_BCVA import IncompleteBCVA
from dataloader.dataset import AierDataset, dataloader
from utils.common import config_loader, summary
from utils.trainUtils import Utils
import torch

if __name__ == '__main__':
    cfgs = config_loader()
    torch.cuda.set_device(cfgs['base_cfg']['gpu_ids'])
    trainer = Utils(cfgs)
    trainer.ensureReproduce()
    trainDataset = AierDataset(cfgs, "train")
    valDataset = AierDataset(cfgs, "val")
    train_dataloader = dataloader(trainDataset, cfgs)
    val_dataloader = dataloader(valDataset, cfgs)
    model = IncompleteBCVA(cfgs)
    summary(model)
    if(cfgs['base_cfg']['isMultiGpu']):
        model = torch.nn.DataParallel(model, device_ids = [])
    trainer.train(model, train_dataloader, val_dataloader)
