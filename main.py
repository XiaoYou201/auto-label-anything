from core import image_classification
from core import seg_video
from utils import get_per_frame as get_frames
from kafka import KafkaConsumer, KafkaProducer
from common import common
from core import object_detection as od
import os

def accurate_seg(frames: list, label: str):
    boxes = image_classification.get_label_box(frames, label)
    print('The boxed len is: '+str(len(boxes)))
    seg_video.mask_video(frames, boxes, batch_size=2, save_path=file_name_without_extension)
#Temporary deprecated
def vague_seg(frames: list, boxes: list):
    seg_video.mask_video(frames, boxes, batch_size=2, save_path=file_name_without_extension)



if __name__ == '__main__':
    video_path = 'data\\videos\\short-cxk.mp4'

    file_name = os.path.basename(video_path)
    file_name_without_extension = os.path.splitext(file_name)[0]
    subfolders = [f.path for f in os.scandir('result') if f.is_dir()]
    for partition in subfolders:
        result_path = partition + '\\' + file_name_without_extension
        # print(partition+'\\'+file_name_without_extension)
        if not os.path.exists(result_path):
            # 如果文件夹不存在，创建文件夹
            os.makedirs(result_path)
            print(f"文件夹 '{result_path}' 已创建。")
            if partition == 'result\\annotations':
                polygon_path = result_path + '\\polygon'
                rle_path = result_path + '\\rle'
                if not os.path.exists(polygon_path):
                    os.makedirs(polygon_path)
                if not os.path.exists(rle_path):
                    os.makedirs(rle_path)
        else:
            print(f"文件夹 '{result_path}' 已存在。")
    pass
    # object detect -> objects in picture -> next week
    # user -> select zhun que lv准确率
    frames: list = get_frames.get_video_frame(video_path)
    # Labels correspond to box one by one
    labels, vague_boxes = od.obj_det(frames[4])
    # display to user
    label_distinct = set(labels)
    # User select label in labels ->
    accurate_seg(frames, labels[0])


def start_receive_data():
    producer = KafkaProducer(bootstrap_servers=['172.21.79.170:9092'])
    consumer = KafkaConsumer('segment', bootstrap_servers=['172.21.79.170:9092'])
    for msg in consumer:
        if (msg.value != "b'\x00\x00\x00\x00\x00\x00'"):
            print(msg)
            videoPath = str(msg.value)[2:-1]
            print("received video path is :" + videoPath)
            box_list = image_classification.get_label_box(get_frames.get_video_frame(videoPath), 'a person')
            seg_video.mask_video(get_frames.get_video_frame(videoPath), box_list, batch_size=2,
                                 save_path=common.ANNOTQTIONS_PATH)