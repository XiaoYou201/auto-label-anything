import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything.utils.transforms import ResizeLongestSide

from PIL import Image
from utils import get_per_frame
from segment_anything import sam_model_registry, SamPredictor
from utils import binary_mask2seg
from common import common

sam_checkpoint = "sam/sam_vit_h.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
batched_input = []


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def get_mask(mask, num, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def mask_video(video_frames, box_list, batch_size, save_path):
    # load_sam_time1 = time.perf_counter()
    # sam_checkpoint = "../sam/sam_vit_b_01ec64.pth"
    # model_type = "vit_b"
    #
    # device = "cuda"
    #
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)
    #
    # predictor = SamPredictor(sam)
    # resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    # batched_input = []
    # load_sam_time2 = time.perf_counter()
    # print("load sam spend %s ms" % ((load_sam_time2 - load_sam_time1) * 1000))
    # image_boxes = torch.tensor([
    #     [500, 200, 750, 530],
    # ], device=sam.device)
    # print(image_boxes)
    # print(image_boxes.shape)

    # print(len(image_boxes), type(image_boxes))
    # print()
    # print(torch.reshape(image_boxes[1], (1,4)))
    # return
    batched_input_list = []
    batched_frame_list = []
    i_batch = 0
    for index, frame in enumerate(video_frames):
        image_boxes = torch.tensor([box_list[index]], device=sam.device)
        print(image_boxes)
        temp_map = {
            'image': get_per_frame.prepare_image(frame, resize_transform, sam),
            'boxes': resize_transform.apply_boxes_torch(image_boxes, frame.shape[:2]),
            'original_size': frame.shape[0:2]
        }
        batched_frame_list.append(frame)
        batched_input_list.append(temp_map)
        if index % batch_size != 0:
            i_batch = i_batch + 1
            print("this is no.%d batched_picture" % i_batch)
            batched_output = sam(batched_input_list, multimask_output=False)
            fig, ax = plt.subplots(1, batch_size, figsize=(20, 20))

            ax[0].imshow(batched_frame_list[index-1])
            ax[1].imshow(batched_frame_list[index])
            batch_length = min(batch_size, len(video_frames)-(batch_size*1))
            # draw_mask(image_id=j, image_boxes=image_boxes, batched_output=batched_output,
            #           ax=ax, batch_length=batch_length, save_path=save_path)

            #保存结果至磁盘
            for i in range(batch_size):
                mask = batched_output[i]['masks'][0]
                maskArray = (mask.cpu().numpy() + 0)[0]
                show_mask(maskArray, ax[i], random_color=True)
                # 保存mask图像
                maskArray[maskArray == 1] = 255
                maskArray[maskArray == 0] = 1
                cv2.imwrite(common.MASK_PATH_BASE + save_path + '\\' + str(index-batch_size+1+i) + '.png', maskArray)
                print("mask success")

                # mask.cpu() shape (1, 680, 1210)
                mask_cpu = mask.cpu().permute(2, 1, 0).squeeze().numpy()
                # print(type(mask_cpu))
                # print(mask_cpu.shape)
                polygon = binary_mask2seg.mask2polygon(mask_cpu)
                rle = binary_mask2seg.binary_mask_to_rle(mask_cpu)
                # rleTest = binary_mask2seg.mask2rle(maskArray)
                area = binary_mask2seg.get_area_from_binary_mask(mask_cpu)
                binary_mask2seg.write_annotation_to_json(id=1, image_id=index-batch_size+1+i, category_id=1, segmentation=polygon,
                                                         area=area, iscrowed=0, bbox=image_boxes.cpu().numpy(),
                                                         path='result\\annotations\\' + save_path + '\\' + 'polygon')
                binary_mask2seg.write_annotation_to_json(id=1, image_id=index-batch_size+1+i, category_id=1, segmentation=rle,
                                                         area=area, iscrowed=1, bbox=image_boxes.cpu().numpy(),
                                                         path='result\\annotations\\' + save_path + '\\' + 'rle')
            plt.show()
            batched_input_list.clear()

def draw_mask(image_id, image_boxes, batched_output, ax, batch_length, save_path):
    """
    画mask、box，将标注文件写入json
    :param box: 目标识别的bbox
    :param batch_output: sam的输出
    :param batch_length: 当前批次的实际长度
    :return:
    """
    for i in range(batch_length):
        for mask in batched_output[i]['masks']:
            show_mask(mask.cpu().numpy(), ax[i], random_color=True)
            # get_mask(mask.cpu().numpy(), i, random_color=True);
        # show_box(image_boxes[i].cpu().numpy(), ax[i])
        ax[i].axis('on')
        mask_cpu = mask.cpu().permute(2, 1, 0).squeeze().numpy()
        print(type(mask_cpu))
        print(mask_cpu.shape)
        rle = binary_mask2seg.binary_mask_to_rle(mask_cpu)
        # print(rle)
        area = binary_mask2seg.get_area_from_binary_mask(mask_cpu)
        # print('area=',area)
        binary_mask2seg.write_annotation_to_json(id=1, image_id=image_id, category_id=1, segmentation=rle,
                                          area=area, iscrowed=1, bbox=image_boxes.cpu().numpy(),
                                          path=save_path)

    pass


# if __name__ == '__main__':
#     mask_video(get_per_frame.get_video_frame("../videos/bird.mp4"), 'bird')
