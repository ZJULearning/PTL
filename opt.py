import argparse

parser = argparse.ArgumentParser(description='reid')

parser.add_argument('--data_path',
                    default="path/to/Market-1501-v15.09.15",
                    help='path of Market-1501-v15.09.15')

parser.add_argument('--mode',
                    default='train', choices=['train', 'evaluate'],
                    help='train or evaluate ')

parser.add_argument('--backbone',
                    default='resnet50',
                    choices=['resnet50', 'resnet101'],
                    help='load weights ')

parser.add_argument('--freeze',action='store_true',
                    help='freeze backbone or not ')

parser.add_argument('--weight',
                    default='weights/model_400.pt',
                    help='load weights ')

parser.add_argument('--epoch',
                    default=400, type=int,
                    help='number of epoch to train')

parser.add_argument('--lr',
                    default=2e-4,
                    type=float,
                    help='learning_rate')

parser.add_argument('--lr_scheduler',
                    default=[320, 360,400],
                    help='MultiStepLR')

parser.add_argument('--classn',
                    type=int,
                    default=751)

parser.add_argument('--rep',
                    type=float,
                    default=0.5)

parser.add_argument("--batchid",
                    default=4,
                    type=int,
                    help='the batch for id')

parser.add_argument("--batchimage",
                    default=4,
                    help='the batch of per id')

parser.add_argument("--batchtest",
                    default=8,
                    help='the batch size for test')

parser.add_argument("--project_name",
                    default='tmp_project',
                    type=str,
                    help='project_name')

parser.add_argument("--arch",
                    default='mgn_ptl',
                    help='use MGN or MGN_PTL')

parser.add_argument("--usegpu",
                    action="store_true",
                    help='set gpu flag')

opt = parser.parse_args()