from core import pic_detect
from core import seg_video
from utils import get_per_frame as get_frames
from kafka import KafkaConsumer, KafkaProducer
from common import common
# data_train = pd.read_csv("data/train/train.csv")
# data_train.info()
# data_train.describe()
# df_num = data_train[['Age', 'SibSp', 'Parch', 'Fare']]
# for i in df_num.columns:
#     plt.hist(df_num[i])
#     plt.title(i)
# print(df_num.corr())
if __name__ == '__main__':
    box_list = pic_detect.get_label_box(get_frames.get_video_frame('E:\pycharm_data\projects\seg-hq\data\videos\short-cxk.mp4'), 'a person')
    seg_video.mask_video(get_frames.get_video_frame('E:\pycharm_data\projects\seg-hq\data\videos\short-cxk.mp4'), box_list, batch_size=2,
                         save_path=common.ANNOTQTIONS_PATH)



    # video = "videos/onebird.mp4"
    # video = "data/videos/short-cxk.mp4"
    # # box_list = pic_detect.get_label_box(get_per_frame.get_video_frame(video), 'a bird')
    # box_list = pic_detect.get_label_box(get_per_frame.get_video_frame(video), 'a person')
    # seg_video.mask_video(get_per_frame.get_video_frame(video), box_list, batch_size=2,
    #                      save_path= common.ANNOTQTIONS_PATH)
def start_receive_data():
    producer = KafkaProducer(bootstrap_servers=['172.21.79.170:9092'])
    consumer = KafkaConsumer('segment', bootstrap_servers=['172.21.79.170:9092'])
    for msg in consumer:
        if (msg.value != "b'\x00\x00\x00\x00\x00\x00'"):
            print(msg)
            videoPath = str(msg.value)[2:-1]
            print("received video path is :" + videoPath)
            box_list = pic_detect.get_label_box(get_per_frame.get_video_frame(videoPath), 'a person')
            seg_video.mask_video(get_per_frame.get_video_frame(videoPath), box_list, batch_size=2,
                                 save_path=common.ANNOTQTIONS_PATH)