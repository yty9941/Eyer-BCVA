import random
import os
import time
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from collections import defaultdict
from utils.earlyStopping import EarlyStopping
from loss.loss import ContrastiveLoss, MSELoss

class Utils:
    def __init__(self, cfgs):
        self.BCVA_loss = MSELoss()
        self.ImageTextContrastive_loss = ContrastiveLoss(cfgs)
        self.seed = cfgs['base_cfg']['seed']
        self.model_save_path = os.path.join(cfgs['base_cfg']['model_save'], cfgs['model_cfg']['image']['image_encoder'], str(cfgs['train_cfg']['Epochs']))
        self.figure_save_path = os.path.join(cfgs['base_cfg']['figure_save'], cfgs['model_cfg']['image']['image_encoder'], str(cfgs['train_cfg']['Epochs']))
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.exists(self.figure_save_path):
            os.makedirs(self.figure_save_path)
        self.cfgs = cfgs
    def calAccuracy(self, y_pred, y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        correct = (np.absolute(
            y_true - y_pred) <= self.cfgs['train_cfg']['threshold'])
        accuracy = np.sum(correct) * 1.0 / y_true.shape[0]
        return accuracy

    def calMAE(self, y_pred, y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        mae = np.mean(np.absolute(y_true - y_pred))
        return mae

    def ensureReproduce(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        torch.use_deterministic_algorithms(False)
        torch.autograd.set_detect_anomaly(True)

    def train(self, model, train_loader, val_loader):
        print("------Training Start!-------")
        train_start_time = time.time()
        model = model.cuda()
        optimizer = optim.Adam(params = model.parameters(),
                               lr = self.cfgs['train_cfg']['Learning_Rate'],
                               betas = (0.9, 0.99))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.5)
        max_acc = 0
        early_stopping = EarlyStopping(self.cfgs['train_cfg']['Early_Stopping_Patience'])
        for epoch in range(1, self.cfgs['train_cfg']['Epochs'] + 1):
            epoch_start_time = time.time()
            model.train()
            batch_dict = defaultdict(list)
            for batch in tqdm(train_loader):
                OctImage, SloImage, label, patientMessage, diagOct, diagSlo, missModalTag, OTSU, ROI = self.unpackToGpu(batch)
                BCVA, predOct, predSlo, octEmbed, sloEmbed = model.forward(OctImage, OTSU, patientMessage, SloImage, missModalTag, diagOct, diagSlo, ROI)
                optimizer.zero_grad()
                predBCVALoss = self.BCVA_loss(BCVA, label)
                predOctLoss = self.BCVA_loss(predOct, label)
                predSloLoss = self.BCVA_loss(predSlo, label)

                if self.cfgs['base_cfg']['isCL']:
                    octCtrLoss = self.ImageTextContrastive_loss(octEmbed, diagOct, missModalTag, "OCT")
                    sloCtrLoss = self.ImageTextContrastive_loss(sloEmbed, diagSlo, missModalTag, "SLO")
                    ctrLoss  = self.cfgs['train_cfg']['incomplete']['gamma'] * octCtrLoss + self.cfgs['train_cfg']['incomplete']['delta'] * sloCtrLoss
                else:
                    ctrLoss = 0
                batch_dict['train_BCVA'] += BCVA.clone().detach().cpu().numpy().tolist()
                batch_dict['train_label'] += label.clone().detach().cpu().numpy().tolist()
                trainLoss = predBCVALoss + predOctLoss * self.cfgs['train_cfg']['incomplete']['alpha'] + \
                            predSloLoss * self.cfgs['train_cfg']['incomplete']['beta'] + ctrLoss
                trainLoss.backward()
                optimizer.step()
                batch_dict["train_loss"].append(trainLoss.clone().detach().cpu())
            trainAccuracy = self.calAccuracy(batch_dict['train_BCVA'], batch_dict['train_label'])
            trainMae = self.calMAE(batch_dict['train_BCVA'], batch_dict['train_label'])
            loss_train_mean = np.mean(np.array(batch_dict["train_loss"]))

            scheduler.step()
            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    OctImage, SloImage, label, patientMessage, \
                    diagOct, diagSlo, missModalTag, OTSU, ROI = self.unpackToGpu(batch)
                    BCVA, predOct, predSlo, octEmbed, sloEmbed = model.forward(OctImage, OTSU, patientMessage, SloImage, missModalTag, diagOct, diagSlo, ROI)
                    predBCVALoss = self.BCVA_loss(BCVA, label)
                    predOctLoss = self.BCVA_loss(predOct, label)
                    predSloLoss = self.BCVA_loss(predSlo, label)
                    if self.cfgs['base_cfg']['isCL']:
                        octCtrLoss = self.ImageTextContrastive_loss(octEmbed, diagOct, missModalTag, "OCT")
                        sloCtrLoss = self.ImageTextContrastive_loss(sloEmbed, diagSlo, missModalTag, "SLO")
                        ctrLoss  = self.cfgs['train_cfg']['incomplete']['gamma'] * octCtrLoss + self.cfgs['train_cfg']['incomplete']['delta'] * sloCtrLoss
                    else:
                        ctrLoss = 0
                    valLoss = predBCVALoss + predOctLoss * self.cfgs['train_cfg']['incomplete']['alpha'] +\
                              predSloLoss * self.cfgs['train_cfg']['incomplete']['beta'] + ctrLoss
                    batch_dict['val_BCVA'] += BCVA.clone().detach().cpu().numpy().tolist()
                    batch_dict['val_label'] += label.clone().detach().cpu().numpy().tolist()
                    batch_dict["val_loss"].append(valLoss.clone().detach().cpu())
                valAccuracy = self.calAccuracy(batch_dict['val_BCVA'], batch_dict['val_label'])
                loss_val_mean = np.mean(np.array(batch_dict["val_loss"]))

            print("Epoch: {}/{}".format(epoch, self.cfgs['train_cfg']['Epochs']))
            print(f"[ Train | {epoch:03d}/{self.cfgs['train_cfg']['Epochs']:03d} ] \n"
                  f"loss = {loss_train_mean:.5f}, acc = {trainAccuracy:.5f}, mae = {trainMae:.5f}\n")
            if (valAccuracy > max_acc):
                max_acc = valAccuracy
                torch.save(model.state_dict(), self.model_save_path + '/Best.pt')
                print("Epoch " + str(epoch) + " save model!")
            epoch_end_time = time.time()
            epoch_time_interval = epoch_end_time - epoch_start_time
            print('Time Cost: {:.0f}m {:.0f}s'.format(epoch_time_interval // 60, epoch_time_interval % 60))
            early_stopping(loss_val_mean)
            if early_stopping.early_stop:
                print("Epoch " + str(epoch) + " early stopping！")
                break
        print("------Training Finish!-------")
        train_end_time = time.time()
        train_time_interval = train_end_time - train_start_time
        print('Total Time Cost: {:.0f}m {:.0f}s'.format(train_time_interval // 60, train_time_interval % 60))
    def test(self, model, test_loader):
        print("------Test Start!-------")
        test_start_time = time.time()
        model.load_state_dict(torch.load(self.model_save_path + '/Best.pt'))
        model = model.cuda()
        model.eval()
        result_dict = defaultdict(list)
        with torch.no_grad():
            for batch in tqdm(test_loader):
                OctImage, SloImage, label, patientMessage, diagOct, \
                diagSlo, missModalTag, OTSU, ROI = self.unpackToGpu(batch)

                BCVA, predOct, predSlo, octEmbed, sloEmbed = model.forward(OctImage, OTSU, patientMessage, SloImage, missModalTag, diagOct, diagSlo, ROI)
                postProcess = []
                for key in label:
                    postProcess.append(self.getValue(key)[0])
                postProcess = torch.Tensor(postProcess).cuda()
                BCVA += postProcess
                predBCVALoss = self.BCVA_loss(BCVA, label)
                predOctLoss = self.BCVA_loss(predOct, label)
                predSloLoss = self.BCVA_loss(predSlo, label)
                if self.cfgs['base_cfg']['isCL']:
                    octCtrLoss = self.ImageTextContrastive_loss(octEmbed, diagOct, missModalTag, "OCT")
                    sloCtrLoss = self.ImageTextContrastive_loss(sloEmbed, diagSlo, missModalTag, "SLO")
                    ctrLoss  = self.cfgs['train_cfg']['incomplete']['gamma'] * octCtrLoss + self.cfgs['train_cfg']['incomplete']['delta'] * sloCtrLoss
                else:
                    ctrLoss = 0
                testLoss = predBCVALoss + predOctLoss * self.cfgs['train_cfg']['incomplete']['alpha'] + \
                           predSloLoss * self.cfgs['train_cfg']['incomplete']['beta'] + ctrLoss
                result_dict['test_BCVA'] += BCVA.clone().detach().cpu().numpy().tolist()
                result_dict['test_label'] += label.clone().detach().cpu().numpy().tolist()
                result_dict["test_loss"].append(testLoss.clone().detach().cpu())
            testAccuracy = self.calAccuracy(result_dict['test_BCVA'], result_dict['test_label'])
            testMae = self.calMAE(result_dict['test_BCVA'], result_dict['test_label'])
            loss_test_mean = np.mean(np.array(result_dict["test_loss"]))
        print("------Test Finish!-------")
        print(f"test_loss = {loss_test_mean:.5f}, test_acc = {testAccuracy:.5f}, test_mae = {testMae:.5f}\n")
        test_end_time = time.time()
        test_time_interval = test_end_time - test_start_time
        print('Test Cost {:.0f}m {:.0f}s'.format(test_time_interval // 60, test_time_interval % 60))
        
    def unpackToGpu(self, batch):
        OctImage, SloImage, label, patientMessage, \
        diagOct, diagSlo, missModalTag, OTSU, ROI = self.unpackToGpu(batch)
        
        OctImage = OctImage.cuda()
        SloImage = SloImage.cuda()
        label = label.cuda()
        patientMessage = patientMessage.cuda()
        diagOct = diagOct.cuda()
        diagSlo = diagSlo.cuda()
        missModalTag = missModalTag.cuda()
        OTSU = OTSU.cuda()
        ROI = ROI.cuda()
        
        return OctImage, SloImage, label, patientMessage, \
        diagOct, diagSlo, missModalTag, OTSU, ROI












