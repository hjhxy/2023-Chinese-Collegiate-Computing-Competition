# -----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# -----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolo_ssd.yolo import YOLO
import os
from tqdm import tqdm

import exifread
import re
import requests
import json

yolo = YOLO()


def predict_img(save_img_path, input_img_path, input_img_filename):
    image_path = input_img_path + input_img_filename
    image = Image.open(image_path)
    print(image_path)

    GPS_info = find_GPS_image(pic_path=image_path)
    address = find_address_from_GPS(GPS=GPS_info)

    img_property = {'Time_taken': GPS_info['date_information'],
                    'Longitude': GPS_info['GPS_information']['GPSLongitude'],
                    'Latitude': GPS_info['GPS_information']['GPSLatitude'],
                    'Address': address}

    r_image = yolo.detect_image(image, crop=False, count=False)
    r_image_path = save_img_path + input_img_filename
    r_image.save(r_image_path, quality=95, subsampling=0)
    return img_property


def predict_video(save_video_path, input_video_path, input_video_filename):
    video_save_path = save_video_path + input_video_filename
    video_fps = 30
    video_path = input_video_path + input_video_filename
    capture = cv2.VideoCapture(video_path)
    if video_save_path != "":
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # mp4v视频格式无法播放
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    ref, frame = capture.read()
    if not ref:
        raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

    fps = 0.0
    while (True):
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        if not ref:
            break
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测
        frame = np.array(yolo.detect_image(frame))
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        fps = 15 + (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % fps)
        frame = cv2.putText(frame, "fps= %.2f" % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # cv2.imshow("video", frame)
        c = cv2.waitKey(1) & 0xff
        if video_save_path != "":
            out.write(frame)

        if c == 27:
            capture.release()
            break

    print("Video Detection Done!")
    capture.release()
    if video_save_path != "":
        print("Save processed video to the path :" + video_save_path)
        out.release()
    cv2.destroyAllWindows()

    return video_save_path


def latitude_and_longitude_convert_to_decimal_system(*arg):
    return float(arg[0]) + ((float(arg[1]) + (float(arg[2].split('/')[0]) / float(arg[2].split('/')[-1]) / 60)) / 60)


def find_GPS_image(pic_path):
    GPS = {}
    date = ''
    with open(pic_path, 'rb') as f:
        tags = exifread.process_file(f)
        for tag, value in tags.items():
            if re.match('GPS GPSLatitudeRef', tag):
                GPS['GPSLatitudeRef'] = str(value)
            elif re.match('GPS GPSLongitudeRef', tag):
                GPS['GPSLongitudeRef'] = str(value)
            elif re.match('GPS GPSAltitudeRef', tag):
                GPS['GPSAltitudeRef'] = str(value)
            elif re.match('GPS GPSLatitude', tag):
                try:
                    match_result = re.match('\[(\w*),(\w*),(\w.*)/(\w.*)\]', str(value)).groups()
                    GPS['GPSLatitude'] = int(match_result[0]), int(match_result[1]), int(match_result[2])
                except:
                    deg, min, sec = [x.replace(' ', '') for x in str(value)[1:-1].split(',')]
                    GPS['GPSLatitude'] = latitude_and_longitude_convert_to_decimal_system(deg, min, sec)
            elif re.match('GPS GPSLongitude', tag):
                try:
                    match_result = re.match('\[(\w*),(\w*),(\w.*)/(\w.*)\]', str(value)).groups()
                    GPS['GPSLongitude'] = int(match_result[0]), int(match_result[1]), int(match_result[2])
                except:
                    deg, min, sec = [x.replace(' ', '') for x in str(value)[1:-1].split(',')]
                    GPS['GPSLongitude'] = latitude_and_longitude_convert_to_decimal_system(deg, min, sec)
            elif re.match('GPS GPSAltitude', tag):
                GPS['GPSAltitude'] = str(value)
            elif re.match('.*Date.*', tag):
                date = str(value)
    return {'GPS_information': GPS, 'date_information': date}


def find_address_from_GPS(GPS):
    # 使用Geocoding API把经纬度坐标转换为结构化地址。
    secret_key = 'zbLsuDDL4CS2U0M4KezOZZbGUY9iWtVf'
    if not GPS['GPS_information']:
        return '该照片无GPS信息'
    lat, lng = GPS['GPS_information']['GPSLatitude'], GPS['GPS_information']['GPSLongitude']
    baidu_map_api = "http://api.map.baidu.com/geocoder/v2/?ak={0}&callback=renderReverse&location={1},{2}s&output=json&pois=0".format(
        secret_key, lat, lng)
    response = requests.get(baidu_map_api)
    content = response.text.replace("renderReverse&&renderReverse(", "")[:-1]
    baidu_map_address = json.loads(content)
    formatted_address = baidu_map_address["result"]["formatted_address"]
    province = baidu_map_address["result"]["addressComponent"]["province"]
    city = baidu_map_address["result"]["addressComponent"]["city"]
    district = baidu_map_address["result"]["addressComponent"]["district"]
    return formatted_address, province, city, district

# if __name__ == "__main__":
#     predict_img(input_img="E:/课内学习备用文件夹/大三下/计算机设计比赛/flask_server/static/img/2094620417V1.jpg")
