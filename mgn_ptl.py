import copy
import torch
import torch.nn as nn
from opt import opt
from torchvision.models.resnet import resnet50, resnet101, Bottleneck
from bconv_cell import BConvCell
import torch.nn.init as init


class PTL(nn.Module):
    """ Init the Progressive Transfer Learning Network.
        'Progressive transfer learning for person re-identification' by Yu et al.
    """
    def __init__(self):
        super(PTL, self).__init__()
        self.bconv1 = BConvCell(3, 16, kernel_size=3, stride=2, padding=1)
        self.bconv2 = BConvCell(16, 32, kernel_size=3, stride=2, padding=1)
        self.bconv3 = BConvCell(64, 64, kernel_size=3, padding=1)
        self.bconv4 = BConvCell(128, 256, kernel_size=3, stride=2, padding=1)
        self.bconv5 = BConvCell(256, 512, kernel_size=3, stride=2, padding=1)
        self.bconv6 = BConvCell(512, 128, kernel_size=3, stride=2, padding=1)
        self.bconv7 = BConvCell(512, 128, kernel_size=3, padding=1)
        self.bconv8 = BConvCell(512, 128, kernel_size=3, padding=1)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(64 + 32, 64, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(256 + 64, 128, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.fuse4 = nn.Sequential(
            nn.Conv2d(1024 + 512, 512, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 8))

    def resetalllatentstates(self):
        self.bconv1.resetlatentstate()
        self.bconv2.resetlatentstate()
        self.bconv3.resetlatentstate()
        self.bconv4.resetlatentstate()
        self.bconv5.resetlatentstate()
        self.bconv6.resetlatentstate()
        self.bconv7.resetlatentstate()
        self.bconv8.resetlatentstate()

    def init_param(self):
        print('\nInit PTL Net Params\n')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_in')


class MGN_PTL(nn.Module):
    """ Init the Multiple Granularities Network.
    Implement of paper: 'Learning Discriminative Features with Multiple Granularities for Person Re-Identification' proposed by Wang et al.
    Refer url: https://github.com/GNAYUOHZ/ReID-MGN
    """
    def __init__(self):
        super(MGN_PTL, self).__init__()
        num_classes = opt.classn
        feats = 256
        self.val = False
        self.ptl = PTL()
        self.ptl.init_param()
        if opt.backbone == 'resnet50':
            self.backbone = resnet50(pretrained=True)
        elif opt.backbone == 'resnet101':
            self.backbone = resnet101(pretrained=True)
        res_conv4 = nn.Sequential(*self.backbone.layer3[1:])
        res_g_conv5 = self.backbone.layer4
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(self.backbone.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 8))

        self.reduction = nn.Sequential(nn.Conv2d(2048+128, feats, 1, bias=True), nn.BatchNorm2d(feats), nn.ReLU())

        self._init_reduction(self.reduction)

        self.fc_id_2048_0 = nn.Linear(feats, num_classes)
        self.fc_id_2048_1 = nn.Linear(feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(feats, num_classes)
        self.fc_id_256_1_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(feats, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)
        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_in')
        nn.init.constant_(fc.bias, 0.)

    def resetalllatentstates(self):
        self.ptl.resetalllatentstates()

    def forward(self, input):
        if self.val:
            self.resetalllatentstates()
        x1 = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(input))))
        bconv_1 = self.ptl.bconv1(input)
        bconv_2 = self.ptl.bconv2(bconv_1)
        z0 = self.ptl.fuse1(torch.cat([bconv_2, x1], dim=1))
        bconv_3 = self.ptl.bconv3(z0)
        x2 = self.backbone.layer1(x1)
        z1 = self.ptl.fuse2(torch.cat([x2, bconv_3], dim=1))
        bconv_4 = self.ptl.bconv4(z1)
        x3 = self.backbone.layer2(x2)
        z2 = self.ptl.fuse3(torch.cat([x3, bconv_4], dim=1))
        bconv_5 = self.ptl.bconv5(z2)
        x4 = self.backbone.layer3[0](x3)
        p1 = self.p1(x4)
        p2 = self.p2(x4)
        p3 = self.p3(x4)
        z3 = self.ptl.fuse4(torch.cat([x4, bconv_5], dim=1))
        bconv_6 = self.ptl.bconv6(z3)
        bconv_7 = self.ptl.bconv7(z3)
        bconv_8 = self.ptl.bconv8(z3)
        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)
        zg_bconv_6 = self.ptl.maxpool_zg_p1(bconv_6)
        zg_bconv_7 = self.ptl.maxpool_zg_p2(bconv_7)
        zg_bconv_8 = self.ptl.maxpool_zg_p3(bconv_8)
        zp2_bconv_7 = self.ptl.maxpool_zp2(bconv_7)
        z0_p2_bconv_7 = zp2_bconv_7[:, :, 0:1, :]
        z1_p2_bconv_7 = zp2_bconv_7[:, :, 1:2, :]
        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp2_bconv_8 = self.ptl.maxpool_zp3(bconv_8)
        z0_p3_bconv_8 = zp2_bconv_8[:, :, 0:1, :]
        z1_p3_bconv_8 = zp2_bconv_8[:, :, 1:2, :]
        z2_p3_bconv_8 = zp2_bconv_8[:, :, 2:3, :]
        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]
        fg_p1 = self.reduction(torch.cat([zg_p1, zg_bconv_6], dim=1)).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction(torch.cat([zg_p2, zg_bconv_7], dim=1)).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction(torch.cat([zg_p3, zg_bconv_8], dim=1)).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction(torch.cat([z0_p2, z0_p2_bconv_7], dim=1)).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction(torch.cat([z1_p2, z1_p2_bconv_7], dim=1)).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction(torch.cat([z0_p3, z0_p3_bconv_8], dim=1)).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction(torch.cat([z1_p3, z1_p3_bconv_8], dim=1)).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction(torch.cat([z2_p3, z2_p3_bconv_8], dim=1)).squeeze(dim=3).squeeze(dim=2)

        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)
        return predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3

