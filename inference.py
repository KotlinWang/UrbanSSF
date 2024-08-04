import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_supervision import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from geoseg.datasets.uavid_dataset import *
from geoseg.datasets.vaihingen_dataset import *

from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from tools.metric import Evaluator

import time


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    return parser.parse_args()

def model_dateset():
    seed_everything(42)
    args = get_args()
    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(os.path.join(config.weights_path, config.test_weights_name+'.ckpt'), config=config)
    model.cuda()
    model.eval()
    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()

    test_dataset = config.test_dataset

    return config, model, test_dataset, evaluator

def confusion_txt():
    config, model, test_dataset, evaluator = model_dateset()

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            raw_predictions = model(input['img'].cuda())

            image_ids = input["img_id"]
            masks_true = input['gt_semantic_seg']

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(masks_true.shape[0]):
                evaluator.add_batch(pre_image=predictions[i].cpu().numpy(), gt_image=masks_true[i].cpu().numpy())

    if config.weights_path.split('/')[1] != 'uavid':
        cm = evaluator.confusion_matrix[:-1, :-1].astype('float') / evaluator.confusion_matrix[:-1, :-1].sum(axis=1)[:, np.newaxis]
        np.savetxt('./{}_{}.txt'.format(config.weights_path.split('/')[1] ,config.test_weights_name), cm)
        mIoU = evaluator.Intersection_over_Union()
        print(np.nanmean(mIoU[:-1]))
    else:
        cm = evaluator.confusion_matrix.astype('float') / evaluator.confusion_matrix.sum(axis=1)[:, np.newaxis]
        np.savetxt('./{}_{}.txt'.format(config.weights_path.split('/')[1] ,config.test_weights_name), cm)

        mIoU = evaluator.Intersection_over_Union()
        print(np.nanmean(mIoU))
    
def performance(shape=(3, 1024, 1024)):
    config, model, test_dataset, evaluator = model_dateset()

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=12,
            pin_memory=True,
            drop_last=False,
        )

        time_list=[]
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            start = time.time()
            raw_predictions = model(input['img'].cuda())
            # torch.cuda.synchronize()
            end = time.time()
            time_list.append(end - start)

            # image_ids = input["img_id"]
            # masks_true = input['gt_semantic_seg']
            # raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            # predictions = raw_predictions.argmax(dim=1)
        

    img = torch.randn((1, *shape), device=next(model.parameters()).device)
    params = parameter_count(model)[""]

    supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            "aten::max_pool2d": None,  # as permute is in _IGNORED_OPS
            # "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,  # latter
        }
    
    Gflops, unsupported = flop_count(model=model, inputs=(img,), supported_ops=supported_ops)

    print("Params: {:.3f}M\nFPS: {:.3f}\nGflops: {:.3f}G".format(params / 1e6, 1 / (np.mean(time_list)), sum(Gflops.values())))


def visualization():
    if not os.path.exists('./save/visual'):
        os.makedirs('./save/visual')

    config, model, test_dataset, evaluator = model_dateset()
    # test_dataset = UAVIDDataset(data_root='./data', img_dir='images', mask_dir='masks', mode='val',
    #                        mosaic_ratio=0.0, transform=val_aug, img_size=(1024, 1024))
    test_dataset = VaihingenDataset(data_root='./data/vaihingen', transform=val_aug)


    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            raw_predictions = model(input['img'].cuda())


def hotmap():
    if not os.path.exists('./save/hotmap'):
        os.makedirs('./save/hotmap')

    config, model, test_dataset, evaluator = model_dateset()
    test_dataset = UAVIDDataset(data_root='./data/hotmap', img_dir='images', mask_dir='masks', mode='val',
                           mosaic_ratio=0.0, transform=val_aug, img_size=(1024, 1024))


    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            img = cv2.imread(os.path.join('./data/hotmap/images', input['img_id'][0]+'.png'))

            h, w, _ = img.shape
            pre = model(input['img'].cuda())

            import matplotlib.pyplot as plt
            plt.axis('off')
            plt.imshow(pre.sum(dim=1).contiguous().data.cpu().numpy()[0, :, :], cmap='jet')
            plt.savefig('./save/hotmap/8.jpg', bbox_inches='tight', pad_inches=0, dpi=1200)
            plt.close()
            exit()
            
            pre = pre[0].sum(dim=0).contiguous().data.cpu().numpy()
            pre = cv2.resize(pre, (h, w), cv2.INTER_NEAREST)

            pre = (pre - np.min(pre)) / (np.max(pre) - np.min(pre))

            pre = np.uint8(255 * pre)
            heatmap = cv2.applyColorMap(pre.astype(np.uint8), cv2.COLORMAP_JET)


            # 进行归一化
            # heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
            # heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
            result = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
            cv2.imwrite('./save/hotmap/trans_2.jpg', result)
            exit()


if __name__ == "__main__":
    # confusion_txt()
    # performance()
    visualization()
    # hotmap()
    