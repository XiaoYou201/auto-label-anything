import cv2
import torch
import imageio.v3 as iio
from PIL import Image
import numpy


def get_video_frame(video: str):
    videoFrames = []

    videoCapture = cv2.VideoCapture()
    videoCapture.open(video)

    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        frame =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        videoFrames.append(frame)

    print(len(videoFrames))
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