import json

import numpy as np
import pycocotools.mask as mask_util
from skimage import measure
import cv2

def binary_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
    binary_mask: a 2D binary numpy array where '1's represent the object
    tolerance: Maximum distance from original points of polygon to approximated
    polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons

def get_area_from_binary_mask(mask):
    num = mask == 1
    area = mask[num]
    return area.size

def write_annotation_to_json(id, image_id, category_id, segmentation, area, bbox, iscrowed, path):
    """
    annotation{
        "id": int,
        "image_id": int,
        "category_id": int,
        "segmentation": RLE or [polygon],
        "area": float,
        "bbox": [x,y,width,height],
        "iscrowd": 0 or 1,
}
    :param bbox:
    """
    # bbox =[bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
    bbox = [bbox[0][0], bbox[0][1], bbox[0][2]-bbox[0][0], bbox[0][3]-bbox[0][1]]
    annotation = {
        "id": id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "area": area,
        "bbox": str(bbox),
        "iscrowd": iscrowed,
    }
    save_file = path +'/' +str(image_id) + '.json'
    with open(save_file, 'w') as f:
        json.dump(annotation, f)


def mask2polygon(mask):
    contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour_list = contour.flatten().tolist()
        if len(contour_list) > 4:# and cv2.contourArea(contour)>10000
            segmentation.append(contour_list)
    return segmentation

#未测试
def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape, dtype=np.uint8)
    polygons = np.asarray(polygons, np.int32) # 这里必须是int32，其他类型使用fillPoly会报错
    shape=polygons.shape
    polygons=polygons.reshape(shape[0],-1,2)
    cv2.fillPoly(mask, polygons,color=1) # 非int32 会报错
    return mask
#test------------------------------
# import numpy as np
# mask = np.ones((100, 100))
# for i in range(10):
#     for j in range(10):
#         mask[i][j]=0
# mask2polygon(mask)
#[[10, 0, 10, 9, 9, 10, 0, 10, 0, 99, 99, 99, 99, 0]]
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(rle, input_shape):
    width, height = input_shape[:2]

    mask = np.zeros(width * height).astype(np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1
        current_position += lengths[index]
    return mask.reshape(height, width).T