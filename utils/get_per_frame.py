import os

import cv2
import torch
import imageio.v3 as iio
from PIL import Image
import numpy

import common.common


def get_video_frame(video: str):
    videoFrames = []

    videoCapture = cv2.VideoCapture()
    videoCapture.open(video)

    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        frame =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        videoFrames.append(frame)
        file_name = os.path.basename(video)
        file_name_without_extension = os.path.splitext(file_name)[0]
        pic_save_path = os.path.join(common.common.PIC_SAVE_PATH_BASE+file_name_without_extension, f'frame_{i:04d}.png')
        # print(pic_save_path)
        cv2.imwrite(pic_save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # 保存为图像文件，转换回BGR
    print('The size of frames are ' + str(len(videoFrames)))
    return videoFrames
# get_video_frame("../videos/test.mp4")
def get_video_from_bytes(msg_bytes: bytes):
    videoFrames = []
    frames = iio.imiter(msg_bytes, format_hint=".mp4")
    for frame in frames:
        videoFrames.append(frame)
    return videoFrames
def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)
    return image.permute(2,0,1).contiguous()

# get_video_frame("../videos/test.mp4")