import os
import sys
import argparse
import os.path as osp
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from torch.utils import data
from torch.nn import functional as F

from utils.eval_utils import *
from networks.baseline import Res_Deeplab
from dataset.dataset import ISPRSDataSet, ISPRSDataValSet

from utils.criterion import CriterionDSN
from utils.metrics import AverageMeter, Result
from utils.encoding import DataParallelModel, DataParallelCriterion

torch_ver = torch.__version__[:3]

if torch_ver == '0.3':
    from torch.autograd import Variable

IGNORE_LABEL = 255
IMG_MEAN = np.array((120.47595769, 81.79931481, 81.19268267), dtype=np.float32)

DATA_DIRECTORY = '/workspace/dataset/data'
DATA_LIST_PATH = '/workspace/dataset/train.txt'
TEST_DATA_LIST_PATH = '/workspace/dataset/test.txt'

BATCH_SIZE = 6
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 0.0005

GPU_NUM = '0'
INPUT_SIZE = '512,512'

POWER = 0.9
MOMENTUM = 0.9
NUM_CLASSES = 1
NUM_STEPS = 80000
RANDOM_SEED = 111

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000

RESTORE_FROM = '/workspace/pretrain/resnet50-imagenet.pth'
SNAPSHOT_DIR = 'snapshots/baseline_results/'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")

    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY, help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH, help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL, help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE, help="Comma-separated string with height and width of images.")

    parser.add_argument("--is-training", action="store_true", help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Number of images sent to the network in one step.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM, help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true", help="Whether to not restore last (FC) layers.")
    parser.add_argument("--start-iters", type=int, default=0, help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS, help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER, help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=str, default=GPU_NUM, help="choose gpu device.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM, help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES, help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY, help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR, help="Where to save snapshots of the model.")

    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES, help="Number of classes to predict (including background).")
    parser.add_argument("--random-mirror", action="store_true", help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true", help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Random seed to have reproducible results.")
    parser.add_argument("--ft", type=bool, default=False, help="fine-tune the model with large input size.")
    parser.add_argument("--ohem", type=str2bool, default='False', help="use hard negative mining")
    parser.add_argument("--ohem-thres", type=float, default=0.6, help="choose the samples with correct probability under the threshold.")
    parser.add_argument("--ohem-keep", type=int, default=200000, help="choose the samples with correct probability under the threshold.")
    return parser.parse_args()

args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003


def main():
    """Create the model and start the training."""
    # writer = SummaryWriter(args.snapshot_dir)

    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    cudnn.enabled = True
    # -----------------------------
    # Create network.
    # -----------------------------
    net = Res_Deeplab(num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from, map_location='cpu')
    new_params = net.state_dict().copy()

    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[0] == 'fc':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
    net.load_state_dict(new_params)
    model = DataParallelModel(net)
    model.cuda()

    if args.ohem:
        criterion = CriterionOhemDSN(thresh=args.ohem_thres, min_kept=args.ohem_keep)
    else:
        criterion = CriterionDSN()

    criterion = DataParallelCriterion(criterion)
    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    traindataset = ISPRSDataSet(args.data_dir, args.data_list,
                                max_iters=args.num_steps * args.batch_size,
                         crop_size=input_size, scale=args.random_scale,
                                mirror=args.random_mirror, mean=IMG_MEAN)
    trainloader = data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    testdataset = ISPRSDataValSet(args.data_dir, TEST_DATA_LIST_PATH, mean=IMG_MEAN, scale=False, mirror=False)
    testloader = data.DataLoader(testdataset,batch_size=1, shuffle=False, pin_memory=True)

    optimizer = optim.SGD([{'params': filter(lambda p: p.requires_grad, net.parameters()), 'lr': args.learning_rate}],
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    for i_iter, batch in enumerate(trainloader):
        i_iter += args.start_iters

        images, dsms, labels, size, _ = batch

        images = images.cuda()
        labels = labels.long().cuda()
        dsms = dsms.cuda()

        model.train()
        model.float()

        if torch_ver == "0.3":
            images = Variable(images)
            labels = Variable(labels)
            dsms = Variable(dsms)

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter)

        preds = model(images)
        pred_up = F.upsample(input=preds[0], size=(labels.size(1), labels.size(2)), mode='bilinear', align_corners=True)
        loss = criterion(pred_up, dsms)

        loss.backward()
        optimizer.step()

        if i_iter % 100 == 0:
            print('======[Train]: iter:{}/{} | loss = {} | lr = {}======'.format(i_iter, args.num_steps, loss.data.cpu().numpy(), lr))

        if i_iter >= args.num_steps - 1:
            print('================Save model====================')
            torch.save(net.state_dict(), osp.join(args.snapshot_dir, 'ISPRS_lr1e-2_' + str(args.num_steps) + '.pth'))
            break

        if i_iter!= 0 and i_iter % args.save_pred_every == 0:
            print('----------------------------------------------')
            print('===================Eval=======================')
            h, w = map(int, args.input_size.split(','))
            input_size = (h, w)
            eval(testloader, model, input_size)
            torch.save(net.state_dict(), osp.join(args.snapshot_dir, 'ISPRS_lr1e-2_' + str(i_iter) + '.pth'))


def eval(testloader, model, input_size):
    average_meter = AverageMeter()
    model.eval()

    for index, batch in enumerate(tqdm(testloader)):
        image, label, ndsm, size, name = batch

        with torch.no_grad():
            output = predict_sliding(model, image.numpy(), input_size, 1, True)
        output = output * 183.17412
        seg_pred = output
        seg_pred = np.squeeze(seg_pred)

        dsm = np.array(ndsm)
        dsm = np.squeeze(dsm)
        result = Result()
        result.evaluate(seg_pred, dsm)
        average_meter.update(result, image.size(0))

    avg = average_meter.average()

    print('----------------------------------------------')
    print('[Eval]: DSM Estimate metrics:')
    print('==============================================')
    print('MAE={average.mae:.3f},    MSE={average.mse:.3f},    RMSE={average.rmse:.3f}\n'
          'Delta1={average.delta1:.3f}, Delta2={average.delta2:.3f}, Delta3={average.delta3:.3f}'.format(average=avg))
    print('==============================================')
    print('----------------------------------------------')



if __name__ == '__main__':
    main()
