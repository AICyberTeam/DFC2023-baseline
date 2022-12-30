import os
import timeit
import argparse

from tqdm import tqdm
from torch.utils import data

from utils.eval_utils import *
from networks.baseline import Res_Deeplab
from dataset.dataset import ISPRSDataValSet
from utils.metrics import AverageMeter, Result


start = timeit.default_timer()
IMG_MEAN = np.array((120.47595769, 81.79931481, 81.19268267), dtype=np.float32)

DATA_DIRECTORY = '/workspace/dataset/data/'
DATA_LIST_PATH = '/workspace/dataset/test.txt'
RESTORE_FROM = '/workspace/baseline_results/best.pth'

NUM_STEPS = 17
INPUT_SIZE = '512,512'



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--whole", type=bool, default=False,
                        help="use whole input size.")
    return parser.parse_args()




def main():
    """Create the model and start the evaluation process."""
    average_meter = AverageMeter()
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    h, w = map(int, args.input_size.split(','))

    if args.whole:
        input_size = (2000, 2000)
    else:
        input_size = (h, w)

    model = Res_Deeplab(num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()

    testdataset = ISPRSDataValSet(args.data_dir, args.data_list, mean=IMG_MEAN, scale=False, mirror=False)
    testloader = data.DataLoader(testdataset,batch_size=1, shuffle=False, pin_memory=True)


    print('----------------------------------------------')
    print('Testing:')
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    for index, batch in enumerate(tqdm(testloader)):
        image, label, ndsm, size, name = batch

        with torch.no_grad():
            if args.whole:
                output = predict_multiscale(model, image, (image.shape[2], image.shape[3]), [0.75, 1.0, 1.25],
                                            args.num_classes, True)
            else:
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
    end = timeit.default_timer()




    print('----------------------------------------------')
    print('DSM Estimate metrics:')
    print('**********************************************')
    print('MAE={average.mae:.3f}, MSE={average.mse:.3f},    RMSE={average.rmse:.3f}\n'
          'Delta1={average.delta1:.3f}, Delta2={average.delta2:.3f}, Delta3={average.delta3:.3f}'.format(average=avg))
    print('**********************************************')
    print('Model inference time:', end - start, 'seconds')
    print('----------------------------------------------')





if __name__ == '__main__':
    main()
