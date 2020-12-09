import os
import numpy as np
import glob
import cv2
from mtcnn.mtcnn import MTCNN
import math
import pandas
from PIL import Image
import time
"""
reading all videos in given dir (including subdirs and subsubdirs)
return the aligend face images of each video
"""
detector = MTCNN()
import argparse
parser = argparse.ArgumentParser(description="MTCNN video face preprocessing")
parser.add_argument('-i', '--input_dir', type= str, default = None)
parser.add_argument('-o', '--output_dir', type= str, default = None)
parser.add_argument( '--alignment', action='store_false', help='default: face alignment')
parser.add_argument('--size', type=int, default = 100 , help='face size nxn')
parser.add_argument('--save_fl', action='store_false', help='default: save facial landmarks') 
parser.add_argument('-q', '--quiet', action='store_true', help='whether to output face detection results')
args = parser.parse_args()
def video_reader(input_dir, output_dir):
    video_ext = ['.avi', '.mp4', '.MP4']
    video_input_paths = []
    video_output_dirs = []
    for dirpath, dirname, filenames in os.walk(input_dir):
        if any([ext in filename for ext in video_ext for filename in filenames]):
            video_path = [os.path.join(dirpath, filename) for filename in filenames if any([ext in filename for ext in video_ext])]
            video_names = [os.path.splitext(filename)[0] for filename in filenames]
            prefix = dirpath.replace(input_dir, output_dir)
            output_video_dirs = [os.path.join(prefix, video_n) for video_n in video_names]
            
            video_input_paths.extend(video_path)
            video_output_dirs.extend(output_video_dirs)
    return video_input_paths, video_output_dirs

def crop_face(image, rotate = True, quiet_mode=True):
    height, width, channels = image.shape #cv2 image
    detections = detector.detect_faces(image)
    image = PIL_image_convert(image)

    if detections==None or len(detections)==0:
        if not quiet_mode:
            print("***No Face detected. ***") 
        return None, None
    if len(detections) > 1:
        if not quiet_mode:
            print("*** Multi Faces ,get the face with largest confidence ***")
    detection = sorted(detections, key=lambda x: x['confidence'], reverse=True)[0]
    bounding_box = detection['box']
    keypoints = detection['keypoints']
    lex, ley = keypoints['left_eye']
    rex, rey = keypoints['right_eye']
    rmx, rmy = keypoints['mouth_right']
    lmx, lmy = keypoints['mouth_left']
    nex, ney = keypoints['nose']
    # roation using PIL image

    if rotate:
        angle = calculate_angle(lex, ley, rex, rey)
        image, lex, ley, rex, rey, lmx, lmy, rmx, rmy \
            = image_rote(image, angle, lex, ley, rex, rey, lmx, lmy, rmx, rmy) 
    eye_width = rex - lex # distance between two eyes 
    ecx, ecy = (lex + rex) / 2.0, (ley + rey) / 2.0 # the center between two eyes 
    mouth_width = rmx - lmx  
    mcx, mcy = (lmx + rmx) / 2.0, (lmy + rmy) / 2.0 #mouth center coordinate
    em_height = mcy - ecy # height between mouth center to eyes center
    fcx, fcy = (ecx + mcx) / 2.0, (ecy + mcy) / 2.0 # face center
    # face 
    if eye_width > em_height: 
        alpha = eye_width 
    else: 
        alpha = em_height 
    g_beta = 2.0 
    g_left = fcx - alpha / 2.0 * g_beta 
    g_upper = fcy - alpha / 2.0 * g_beta 
    g_right = fcx + alpha / 2.0 * g_beta 
    g_lower = fcy + alpha / 2.0 * g_beta 
    g_face = image.crop((g_left, g_upper, g_right, g_lower)) 
    
    return g_face, keypoints

def image_rote(img, angle, elx, ely, erx, ery, mlx, mly, mrx, mry, expand=1):
    w,h= img.size
    img = img.rotate(angle, expand=expand)  #whether to expand after rotation
    if expand == 0: 
        elx, ely = pos_transform_samesize(angle, elx, ely, w, h) 
        erx, ery = pos_transform_samesize(angle, erx, ery, w, h) 
        mlx, mly = pos_transform_samesize(angle, mlx, mly, w, h) 
        mrx, mry = pos_transform_samesize(angle, mrx, mry, w, h) 
    if expand == 1: 
        elx, ely = pos_transform_resize(angle, elx, ely, w, h) 
        erx, ery = pos_transform_resize(angle, erx, ery, w, h) 
        mlx, mly = pos_transform_resize(angle, mlx, mly, w, h) 
        mrx, mry = pos_transform_resize(angle, mrx, mry, w, h) 
    return img, elx, ely, erx, ery, mlx, mly, mrx, mry

def calculate_angle(elx, ely, erx, ery): 
    """
    calculate image rotate angle
    :param elx: lefy eye x
    :param ely: left eye y
    :param erx: right eye x
    :param ery: right eye y
    :return: rotate angle
    """ 
    dx = erx - elx 
    dy = ery - ely 
    angle = math.atan(dy / dx) * 180 / math.pi 
    return angle

def pos_transform_resize(angle, x, y, w, h): 
    """
    after rotation, new coordinate with expansion
    :param angle:
    :param x:
    :param y:
    :param w:
    :param h:
    :return:
    """ 
    angle = angle * math.pi / 180 
    matrix = [ math.cos(angle), math.sin(angle), 0.0, -math.sin(angle), math.cos(angle), 0.0 ] 
    def transform(x, y, matrix=matrix): 
        (a, b, c, d, e, f) = matrix 
        return a * x + b * y + c, d * x + e * y + f # calculate output size 
    xx = [] 
    yy = [] 
    for x_, y_ in ((0, 0), (w, 0), (w, h), (0, h)): 
        x_, y_ = transform(x_, y_) 
        xx.append(x_) 
        yy.append(y_) 
    ww = int(math.ceil(max(xx)) - math.floor(min(xx))) 
    hh = int(math.ceil(max(yy)) - math.floor(min(yy))) 
    # adjust center 
    cx, cy = transform(w / 2.0, h / 2.0) 
    matrix[2] = ww / 2.0 - cx 
    matrix[5] = hh / 2.0 - cy 
    tx, ty = transform(x, y) 
    return tx, ty

def pos_transform_samesize(angle, x, y, w, h): 
    """
    after rotation, new coordinate without expansion
    :param angle:
    :param x:
    :param y:
    :param w:
    :param h:
    :return:
    """ 
    angle = angle * math.pi / 180 
    matrix = [ math.cos(angle), math.sin(angle), 0.0, -math.sin(angle), math.cos(angle), 0.0 ] 
    def transform(x, y, matrix=matrix): 
        (a, b, c, d, e, f) = matrix 
        return a * x + b * y + c, d * x + e * y + f 
    cx, cy = transform(w / 2.0, h / 2.0) 
    matrix[2] = w / 2.0 - cx 
    matrix[5] = h / 2.0 - cy 
    x, y = transform(x, y) 
    return x, y
def PIL_image_convert(cv2_im):
    cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    return pil_im
def parse_video_frames():
    input_dir = args.input_dir
    output_dir = args.output_dir
    alignment = args.alignment
    size = args.size
    save_fl = args.save_fl
    quiet_mode = args.quiet
    video_input_paths, video_output_dirs = video_reader(input_dir, output_dir)
    length = len(video_input_paths)
    video_index= 0
    total_frames = 0
    failed_frames = 0
    for video_input_file, video_output_dir in zip(video_input_paths, video_output_dirs):
        video_index +=1
        df = pandas.DataFrame()
        print("processing {}/{}\n".format(video_index, length), end='/r')
        if os.path.isfile(video_input_file):
            if not os.path.isdir(video_output_dir):
                os.makedirs(video_output_dir)
            else:
                if len(os.listdir(video_output_dir))!=0:
                    continue
            cap = cv2.VideoCapture(video_input_file)
            index = 0
            total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()
                index+=1
                if ret==True:
                    face_img, keypoints = crop_face(frame, rotate=alignment, quiet_mode=quiet_mode)
                    if (keypoints is None) :
                        keypoints = {'frame': index, 'detect': 0,'nose': (0, 0), 'mouth_right': (0, 0), 'right_eye': (0, 0), 'left_eye': (0, 0), 'mouth_left': (0, 0)}
                    else:
                        keypoints['frame'] = index
                        keypoints['detect'] = 1 
                    if face_img is not None:
                        face_name = '%6d.jpg'%index
                        face_path = os.path.join(video_output_dir, face_name)
                        face_img = face_img.resize((size, size))
                        face_img.save(face_path)
                    else:
                        failed_frames += 1
                    df = df.append(keypoints, ignore_index=True)
                else: 
                    break
            cap.release()
            if save_fl:
                df.to_csv(os.path.join(video_output_dir, 'keypoints.csv'), index=False)     
        else:
            print(video_input_file+'does not exist.')
        time.sleep(5)
    print("{}/{} frames failed".format(failed_frames, total_frames))

if __name__ == "__main__":
    parse_video_frames()
    

