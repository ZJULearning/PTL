import numpy as np
from scipy.spatial.distance import cdist
import torch
import os
from torch.optim import Adam, lr_scheduler
from opt import opt
from data import Data
from mgn_ptl import MGN_PTL
from mgn import MGN
from loss import Loss
from functions import mean_ap, cmc, re_ranking

def usegpu(element):
    if opt.usegpu:
        return element.cuda()
    else:
        return element

class Main():
    def __init__(self, reid_model, loss_fn, data_loader):
        self.train_loader = data_loader.train_loader
        self.test_loader = data_loader.test_loader
        self.query_loader = data_loader.query_loader
        self.testset = data_loader.testset
        self.queryset = data_loader.queryset
        self.loss_fn = loss_fn
        self.model = reid_model
        self.optimizer = self.get_optimizer(reid_model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)

    def train(self):
        self.model.train()
        self.scheduler.step()
        if opt.arch == 'mgn_ptl':
            self.model.resetalllatentstates()
        for batch, (inputs, labels) in enumerate(self.train_loader):
            print(self.scheduler.get_lr())
            inputs = usegpu(inputs)
            labels = usegpu(labels)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            train_loss = self.loss_fn(outputs, labels)
            train_loss.backward()
            self.optimizer.step()

    def test(self):

        test_epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.model.eval()
        if opt.arch == 'mgn_ptl':
            self.model.val = True
        qf = self.extract_feature(self.query_loader).numpy()
        gf = self.extract_feature(self.test_loader).numpy()

        # re rank
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                separate_camera_set=False,
                single_gallery_shot=False,
                first_match_break=True)
        m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

        print('epoch:{:d} lr:{:.6f} [   re_rank] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(test_epoch, lr, m_ap, r[0], r[2], r[4], r[9]))

        # no re rank
        dist = cdist(qf, gf)
        r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                separate_camera_set=False,
                single_gallery_shot=False,
                first_match_break=True)
        m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

        print('epoch:{:d} lr:{:.6f} [no re_rank] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(test_epoch, lr, m_ap, r[0], r[2], r[4], r[9]))

    @staticmethod
    def get_optimizer(net):
        if opt.freeze:
            for p in net.parameters():
                p.requires_grad = True
            for q in net.backbone.parameters():
                q.requires_grad = False

            optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()),
                             lr=opt.lr, weight_decay=5e-4, amsgrad=True)

        else:

            optimizer = Adam(net.parameters(), lr=opt.lr, weight_decay=5e-4, amsgrad=True)
        return optimizer

    @staticmethod
    def fliphor(inputs):
        inv_idx = torch.arange(inputs.size(3) - 1, -1, -1).long()  # N x C x H x W
        return inputs.index_select(3, inv_idx)

    def extract_feature(self, data_loader):
        features = torch.FloatTensor()
        for (inputs, labels) in data_loader:

            ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
            for i in range(2):
                if i == 1:
                    inputs = self.fliphor(inputs)
                input_img = usegpu(inputs)
                outputs = self.model(input_img)
                f = outputs[0].data.cpu()
                ff = ff + f

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            features = torch.cat((features, ff), 0)
        return features


if __name__ == '__main__':
    assert opt.project_name is not None
    print(opt)
    loader = Data()
    if opt.arch == 'mgn_ptl':
        model = usegpu(MGN_PTL())
    elif opt.arch == 'mgn':
        model = usegpu(MGN())
    else:
        ValueError('Only mgn & mgn_ptl are supported')

    loss = Loss()
    reid = Main(model, loss, loader)

    if opt.mode == 'train':
        if not os.path.exists('weights/{}/'.format(opt.project_name)):
            os.makedirs('weights/{}/'.format(opt.project_name))
        for epoch in range(1, opt.epoch+1):
            print('\nepoch', epoch)
            reid.train()
            if epoch % 50 == 0 or epoch == 10 or epoch == 1:
                print('\nstart evaluate')
                reid.test()
                torch.save(model.state_dict(), ('weights/{}/model_{}.pt'.format(opt.project_name, epoch)))
        reid.test()
        torch.save(model.state_dict(), ('weights/{}/model_final.pt'.format(opt.project_name)))

    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load('{}'.format(opt.weight)))
        reid.test()