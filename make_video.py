import cv2
import time
import numpy as np

from train_supervision import *

import torch
from tqdm import tqdm
import albumentations as albu

def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [128, 0, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 64, 128]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 128, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [128, 128, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [64, 0, 128]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [192, 0, 192]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [64, 64, 0]
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [0, 0, 0]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb

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


def video2image(videoPath):#两个参数，视频源地址和图片保存地址
    cap = cv2.VideoCapture(videoPath)
    
    img_list = []
    start = time.time()
    while True:
        # 函数cv2.VideoCapture.grab()用来指向下一帧，其语法格式为：
        # 如果该函数成功指向下一帧，则返回值retval为True
        if cap.grab():
            # 函数cv2.VideoCapture.retrieve()用来解码，并返回函数cv2.VideoCapture.grab()捕获的视频帧。该函数的语法格式为：
            # retval, image = cv2.VideoCapture.retrieve()image为返回的视频帧，如果未成功，则返回一个空图像。retval为布尔类型，若未成功，返回False；否则返回True
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(frame, (2048, 1024))
                img_list.append(img)
        else:
            break
    end = time.time()
    print(end - start)
    return img_list

def image2video(img_list):
    fps=20.0
    videopath='./reslut.mp4'#图片保存地址及格式
    out1 = cv2.VideoWriter(videopath,cv2.VideoWriter_fourcc(*'mp4v'),fps, (2048, 1024))
    print(len(img_list))
    for img in img_list:
        #写成视频操作
        out1.write(img)
    out1.release()
    print("all is ok")

def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img):
    img = np.array(img)
    aug = get_val_transform()(image=img.copy())
    img= aug['image']
    return img


def predict():
    seed_everything(42)
    args = get_args()
    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(os.path.join(config.weights_path, config.test_weights_name+'.ckpt'), config=config)
    model.cuda()
    model.eval()

    img_list = video2image('/root/data/dataset/uavid/uavid_test/seq22/images.mp4')

    result_list = []
    for i, input in enumerate(tqdm(img_list)):
        org_img = input
        start = time.time()
        input = val_aug(input)
        input = torch.FloatTensor(input).permute(2, 0, 1).unsqueeze(0).cuda()
        raw_predictions = model(input)

        raw_predictions = nn.Softmax(dim=1)(raw_predictions)
        predictions = raw_predictions.argmax(dim=1)
        # torch.cuda.synchronize()
        end = time.time()
        # print(1 / (end - start))
        img = label2rgb(predictions[0].detach().cpu().numpy())

        result = cv2.addWeighted(org_img.astype(np.uint8), 0.3, img.astype(np.uint8), 0.7, 0)
        result_list.append(result)

    image2video(result_list)


if __name__ == "__main__":
    predict()