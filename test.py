from model.IM_BCVA import IncompleteBCVA
from dataloader.dataset import AierDataset, dataloader
from utils.common import config_loader, summary
from utils.trainUtils import Utils
import torch
if __name__ == '__main__':
    cfgs = config_loader()
    torch.cuda.set_device(cfgs['base_cfg']['gpu_ids'])
    tester = Utils(cfgs)
    tester.ensureReproduce()
    testDataset = AierDataset(cfgs, "test")
    test_dataloader = dataloader(testDataset, cfgs)
    model = IncompleteBCVA(cfgs)
    summary(model)
    if (cfgs['base_cfg']['isMultiGpu']):
        model = torch.nn.DataParallel(model, device_ids = [])
    tester.test(model, test_dataloader)