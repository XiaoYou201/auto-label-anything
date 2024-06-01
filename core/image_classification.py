# import skimage
from PIL import Image
from transformers import pipeline

from utils import get_per_frame

checkpoint = "google/owlvit-base-patch32"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
labels_set = ["human face", "rocket", "nasa badge", "star-spangled banner", "apple", "banana", "chair", "a bird"]


def get_labels_from_image(img):  #
    labels = []
    predictions = detector(
        img,
        candidate_labels=labels_set,
    )
    for prediction in predictions:
        label = prediction["label"]
        if label not in labels:
            labels.append(label)
    return labels


def get_label_list_from_video(video):
    label_list = []
    # labels_from_frame = []
    frames = get_per_frame.get_video_frame(video)
    length = len(frames) // 10
    for i in range(length):
        labels_from_frame = get_labels_from_image(Image.fromarray(frames[i * 10]))
        print("每帧的label:", labels_from_frame)
        # Image.fromarray(frames[10]).show()
        label_list.extend(labels_from_frame)
    label_list = list(set(label_list))
    return label_list


def get_label_box(frames: list, label: str):  # 图片是PIL的image类型
    box_list = []
    for index, frame in enumerate(frames):
        img = Image.fromarray(frame)
        predictions = detector(
            img,
            candidate_labels=[label],
        )
        if len(predictions) == 0:
            box_list.append(box_list[len(box_list)-1])
        for prediction in predictions:
            box = prediction["box"]
            # label = prediction["label"]
            xmin, ymin, xmax, ymax = box.values()
            box_list.append([xmin, ymin, xmax, ymax])
            # print(box_list)
    return box_list
    pass


if __name__ == '__main__':
    # frames = get_per_frame.get_video_frame("../videos/test.mp4")
    # print(type(Image.fromarray(frames[0])))
    # label_list = get_label_list_from_video('../videos/bird.mp4')
    box_list = get_label_box(get_per_frame.get_video_frame('../videos/bird.mp4'), 'a bird')
    print(box_list)
    # print(type(10//3), 10//3)
    pass
