# encoding=utf-8

import argparse
import os
import time
from collections import defaultdict

import cv2
import torch
from loguru import logger

from trackers.ocsort_tracker.ocsort import MCOCSort
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.tracker.byte_tracker import ByteTracker
from yolox.tracking_utils.timer import Timer
from yolox.utils import fuse_model, get_model_info, post_process
from yolox.utils.visualize import plot_tracking_sc, \
    plot_tracking_mc, plot_tracking_ocsort


####################

# New product를 구매한 사람은 10프레임 후의 위치를 기준으로 결정된다.
NPF = 10

# Disappeared product는 10프레임 후에도 없을 경우 사라진 것으로 처리한다.
DPF = 10

# Disappeared customer는 5프레임 후에도 없을 경우 사라진 것으로 처리한다.
DCF = 30


# 상품을 누가 가져갔는지 고르기 위해 필요한 함수들

import numpy as np
from numpy.linalg import norm

# 두 vector 간의 거리 계산


def angle(p1, p2):
    dot_product = np.dot(p1, p2)
    p1_norm = norm(p1)
    p2_norm = norm(p2)
    return dot_product / p1_norm / p2_norm

# 각도가 가장 작은 손님의 번호 반환
# origin: 상품이 처음에 있던 위치(=등장한 순간의 위치)
# product: 일정 시간 뒤의 상품의 위치
# customers_list: 손님들의 위치를 모아놓은 리스트(시점은 product와 동일...인데 딱히 상관 없을 듯.)
# 각 parameter는 1 x 2 모양의 numpy 배열임.


def purchased_customer(origin, product, customers_list):
    max_cos = -1
    target_id = ""
    for customer_id in customers_list:
        p_vec = product - origin
        c_pos = customers_list[customer_id][1]
        c_vec = c_pos - origin
        angle_temp = angle(p_vec, c_vec)
        if angle_temp > max_cos:
            max_cos = angle_temp
            target_id = customer_id
    return target_id

# 상품이 사라졌을 때, 그 상품이 선반 쪽에서 사라졌는지를 판단하는 함수.
# 좌표의 범위는 알아서 숫자를 바꿔서 설정한다.


def is_on_the_shelves(position):
    if 210 < position[0] < 920 and 50 < position[1] < 460:
        return True
    else:
        return False

def is_on_the_shelves2(position):
    if 300 < position[0] < 800 and 200 < position[1] < 460:
        return True
    else:
        return False

# Track의 위치 출력. tlwh: t_bboxes[idx][:]


def position(tlwh):
    x = (tlwh[0] + tlwh[2]) / 2
    y = (tlwh[1] + tlwh[3]) / 2
    return np.array([x, y])

def position2(x1y1x2y2):
    x1y1x2y2 = np.squeeze(x1y1x2y2)
    x1, y1, x2, y2 = x1y1x2y2
    x = (x1+x2)/2
    y = (y1+y2)/2
    return np.array([x, y])


def print_receipt(purchase_table, class_list):
    f = open("receipt.txt", 'w')
    M, N = np.shape(purchase_table)

    for i in range(M):
        customer_already_added = False      # 손님 이름이 이미 출력되었는지를 나타내는 flag
        for j in range(N):
            product_num = purchase_table[i][j]
            if product_num != 0:
                if not customer_already_added:
                    f.write("Customer #%d\n" % i)
                    customer_already_added = True
                f.write(
                    "  - {0:<24}{1:>4}\n".format(class_list[j], product_num))
        if customer_already_added:
            f.write("\n")

    f.close()


####################


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    """
    :return:
    """
    parser = argparse.ArgumentParser("ByteTrack Demo!")

    parser.add_argument("--demo",
                        default="video",  # image
                        help="demo type, eg. image, video, videos, and webcam")
    parser.add_argument("--tracker",
                        type=str,
                        default="oc",
                        help="byte | oc")
    parser.add_argument("-expn",
                        "--experiment-name",
                        type=str,
                        default=None)
    parser.add_argument("-n",
                        "--name",
                        type=str,
                        default=None,
                        help="model name")
    parser.add_argument("--reid",
                        type=bool,
                        default=False,  # True | False
                        help="")
    parser.add_argument("-debug",
                        type=bool,
                        default=True,  # True
                        help="")

    # ----- object classes
    parser.add_argument("--n_classes",
                        type=int,
                        default=5,
                        help="")  # number of object classes
    parser.add_argument("--class_names",
                        type=str,
                        default="car, bicycle, person, cyclist, tricycle",
                        help="")

    # ----- exp file, eg: yolox_x_ablation.py
    parser.add_argument("-f",
                        "--exp_file",
                        default="../exps/example/mot/yolox_tiny_track_c5_darknet.py",
                        type=str,
                        help="pls input your experiment description file")

    # ----- checkpoint file path, eg: ../pretrained/latest_ckpt.pth.tar, track_latest_ckpt.pth.tar
    parser.add_argument("-c",
                        "--ckpt",
                        default="../pretrained/latest_ckpt.pth.tar",
                        type=str,
                        help="ckpt for eval")

    parser.add_argument("--task",
                        type=str,
                        default="track",
                        help="Task mode: track or detect")

    # ----- videos dir path
    parser.add_argument("--video_dir",
                        type=str,
                        default="../videos",
                        help="")

    # "--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
    parser.add_argument("--path",
                        default="../videos/test_13.mp4",
                        help="path to images or video")

    # ----- Web camera's id
    parser.add_argument("--camid",
                        type=int,
                        default=0,
                        help="webcam demo camera id")
    parser.add_argument("--save_result",
                        type=bool,
                        default=True,
                        help="whether to save the inference result of image/video")

    parser.add_argument("--device",
                        default="gpu",
                        type=str,
                        help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf",
                        default=None,
                        type=float,
                        help="test conf")
    parser.add_argument("--nms",
                        default=None,
                        type=float,
                        help="test nms threshold")
    parser.add_argument("--tsize",
                        default=None,
                        type=int,
                        help="test img size")
    parser.add_argument("--fp16",
                        dest="fp16",
                        default=False,  # False
                        action="store_true",
                        help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse",
                        dest="fuse",
                        default=False,
                        action="store_true",
                        help="Fuse conv and bn for testing.")
    parser.add_argument("--trt",
                        dest="trt",
                        default=False,
                        action="store_true",
                        help="Using TensorRT model for testing.")

    # tracking args
    parser.add_argument("--track_thresh",
                        type=float,
                        default=0.5,
                        help="detection confidence threshold")
    parser.add_argument("--iou_thresh",
                        type=float,
                        default=0.3,
                        help="the iou threshold in Sort for matching")
    parser.add_argument("--match_thresh",
                        type=int,
                        default=0.8,
                        help="matching threshold for tracking")
    parser.add_argument("--track_buffer",
                        type=int,
                        default=240,  # 30
                        help="the frames for keep lost tracks")
    parser.add_argument('--min-box-area',
                        type=float,
                        default=10,
                        help='filter out tiny boxes')
    parser.add_argument("--mot20",
                        dest="mot20",
                        default=False,
                        action="store_true",
                        help="test mot20.")

    return parser


def get_image_list(path):
    """
    :param path:
    :return:
    """
    image_names = []
    for main_dir, sub_dir, file_name_list in os.walk(path):
        for file_name in file_name_list:
            apath = os.path.join(main_dir, file_name)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results_dict(f_path,
                       results_dict,
                       data_type,
                       num_classes=5):
    """
    :param f_path:
    :param results_dict:
    :param data_type:
    :param num_classes:
    :return:
    """
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,{cls_id},1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(f_path, "w", encoding="utf-8") as f:
        for cls_id in range(num_classes):  # process each object class
            cls_results = results_dict[cls_id]
            for fr_id, tlwhs, track_ids in cls_results:  # fr_id starts from 1
                if data_type == 'kitti':
                    fr_id -= 1

                for tlwh, track_id in zip(tlwhs, track_ids):
                    if track_id < 0:
                        continue

                    x1, y1, w, h = tlwh
                    # x2, y2 = x1 + w, y1 + h
                    line = save_format.format(frame=fr_id,
                                              id=track_id,
                                              x1=x1, y1=y1, w=w, h=h,
                                              cls_id=cls_id)
                    # if fr_id == 1:
                    #     print(line)

                    f.write(line)
                    # f.flush()

    logger.info('Save results to {}.\n'.format(f_path))


def write_results(file_path, results):
    """
    :param file_path:
    :param results:
    :return:
    """
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(file_path, "w", encoding="utf-8") as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):

                if track_id < 0:
                    continue

                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id,
                                          id=track_id,
                                          x1=round(x1, 1),
                                          y1=round(y1, 1),
                                          w=round(w, 1),
                                          h=round(h, 1),
                                          s=round(score, 2))
                f.write(line)

    logger.info('save results to {}'.format(file_path))


class Predictor(object):
    def __init__(self,
                 model,
                 exp,
                 trt_file=None,
                 decoder=None,
                 device="cpu",
                 fp16=False,
                 reid=False):
        """
        :param model:
        :param exp:
        :param trt_file:
        :param decoder:
        :param device:
        :param fp16:
        :param reid:
        """
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.n_classes
        self.conf_thresh = exp.test_conf
        self.nms_thresh = exp.nms_thresh
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.reid = reid

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        """
        :param img:
        :param timer:
        :return:
        """
        img_info = {"id": 0}

        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.mean, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()

        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()

            # ----- forward
            outputs = self.model.forward(img)
            # -----

            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())

            if self.reid:
                outputs, feature_map = outputs[0], outputs[1]
                outputs = post_process(
                    outputs, self.num_classes, self.conf_thresh, self.nms_thresh)
            else:
                if isinstance(outputs, tuple):
                    outputs, feature_map = outputs[0], outputs[1]
                outputs = post_process(
                    outputs, self.num_classes, self.conf_thresh, self.nms_thresh)
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))

        if self.reid:
            return outputs, feature_map, img_info
        else:
            return outputs, img_info


def image_demo(predictor, vis_folder, path, current_time, save_result):
    """
    :param predictor:
    :param vis_folder:
    :param path:
    :param current_time:
    :param save_result:
    :return:
    """
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]

    files.sort()
    tracker = ByteTracker(opt, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []

    for image_name in files:
        if frame_id % 30 == 0:
            if frame_id != 0:
                logger.info('Processing frame {} ({:.2f} fps)'
                            .format(frame_id,
                                    1.0 / max(1e-5, timer.average_time)))
            else:
                logger.info('Processing frame {} ({:.2f} fps)'
                            .format(frame_id,
                                    30.0))

        outputs, img_info = predictor.inference(image_name, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(
                outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

            # save results
            results.append((frame_id + 1, online_tlwhs,
                           online_ids, online_scores))
            timer.toc()
            online_im = plot_tracking_sc(img_info['raw_img'],
                                         online_tlwhs,
                                         online_ids,
                                         frame_id=frame_id + 1,
                                         fps=1.0 / timer.average_time)
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(vis_folder,
                                       time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(
                save_folder, os.path.basename(image_name))
            cv2.imwrite(save_file_name, online_im)
        ch = cv2.waitKey(0)

        frame_id += 1
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
    # write_results(result_filename, results)


################ 이 함수에 주목 ####################


def track_video(predictor, cap, vid_save_path, opt):
    """
    online or offline tracking
    :param predictor:
    :param cap:
    :param vid_save_path:
    :param opt:
    :return:
    """
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_ids = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # int

    vid_save_path = os.path.abspath(vid_save_path)
    vid_writer = cv2.VideoWriter(vid_save_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps,
                                 (int(width), int(height)))

    # ---------- define the tracker
    if opt.tracker == "byte":
        tracker = ByteTracker(opt, frame_rate=30)
    elif opt.tracker == "oc":
        tracker = MCOCSort(class_names=opt.class_names,
                           det_thresh=opt.track_thresh,
                           iou_thresh=opt.iou_thresh,
                           max_age=opt.track_buffer)
    # ----------

    # ----- class name to class id and class id to class name
    id2cls = defaultdict(str)
    cls2id = defaultdict(int)
    for cls_id, cls_name in enumerate(tracker.class_names):
        id2cls[cls_id] = cls_name
        cls2id[cls_name] = cls_id

    net_size = exp.test_size

    timer = Timer()

    frame_id = 0
    results = []


#######################################################

    # トラッキングID保持用変数
    track_id_dict = {}      # key: id, value: 새로 들어온 순번

    # 현재 매장에 있는 손님 리스트. 각 요소의 key는 id, value는 [tid, 위치 좌표]이다.
    customers_list = {}

    # 현재 화면에 잡히고 있는 상품 리스트. 각 요소의 key는 id, value는 [cid, tid, 구매한 손님 tid, 위치 좌표]이다.
    # 상품의 tid는 필요 없을 것 같긴 하지만... 혹시 모르니 일단 넣어 두었다.
    products_list = {}

    # 현재 손님들이 장바구니에 담은 각 상품의 수
    # "100"은, 많은 수의 손님에 대비해 충분히 큰 숫자로 설정한 것.
    # "11"은, 상품의 종류 수 + 1.
    purchase_table = np.zeros((100, 12), dtype=np.int32)

    new_product_flag = False    # 새 상품이 추가되면 flag가 True가 됨.
    disappeared_product_flag = False        # 화면에서 사라진 상품이 생기면 True가 됨.
    # new_customer_flag 는 필요가 없다. 화면에 잡히면, 그냥 추가하면 끝이라.
    disappeared_customer_flag = False       # 화면에서 사라진 손님이 생기면 True가 됨.

    new_product_id = ""
    disappeared_product_id = ""
    disappeared_customer_id = ""
    new_product_origin = np.zeros(
        (1, 2), dtype=np.float32)     # 상품이 처음으로 화면에 나타난 위치

    np_frame_count = 0
    dp_frame_count = 0
    dc_frame_count = 0


###################################################

    while True:
        if frame_id % 30 == 0:  # logging per 30 frames
            if frame_id != 0:
                logger.info('Processing frame {:03d}/{:03d} | fps {:.2f}'
                            .format(frame_id,
                                    frame_ids,
                                    1.0 / max(1e-5, timer.average_time)))
            else:
                logger.info('Processing frame {:03d}/{:03d} | fps {:.2f}'
                            .format(frame_id,
                                    frame_ids,
                                    30.0))

        # ----- read the video
        ret_val, frame = cap.read()

        if ret_val:
            if opt.reid:
                outputs, feature_map, img_info = predictor.inference(
                    frame, timer)
            else:
                outputs, img_info = predictor.inference(frame, timer)

            dets = outputs[0]

            if dets is not None:
                # ----- update the frame
                img_size = [img_info['height'], img_info['width']]
                # online_targets = tracker.update(dets, img_size, exp.test_size)

                if opt.tracker == "byte":
                    if opt.reid:
                        online_dict = tracker.update_mcmot_emb(dets,
                                                               feature_map,
                                                               img_size,
                                                               exp.test_size)
                    else:
                        # online_dict = tracker.update_mcmot_byte(dets, img_size, net_size)
                        # online_dict = tracker.update_byte_nk(dets, img_size, net_size)
                        online_dict = tracker.update_byte_enhance2(
                            dets, img_size, net_size)
                        # online_dict = tracker.update_oc_enhance2(dets, img_size, net_size)

                elif opt.tracker == "oc":
                    online_dict = tracker.update_frame(
                        dets, img_size, exp.test_size)

                # ----- plot single-class multi-object tracking results
                if tracker.n_classes == 1:
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for track in online_targets:
                        tlwh = track.tlwh
                        tid = track.track_id

                        # vertical = tlwh[2] / tlwh[3] > 1.6
                        # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:

                        if tlwh[2] * tlwh[3] > opt.min_box_area:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(track.score)

                    results.append((frame_id + 1, online_tlwhs,
                                   online_ids, online_scores))

                    timer.toc()
                    online_img = plot_tracking_sc(img_info['raw_img'],
                                                  online_tlwhs,
                                                  online_ids,
                                                  frame_id=frame_id + 1,
                                                  fps=1.0 / timer.average_time)


################## 여기가 실질적으로 수정해야 하는 부분 ######################

                # ----- plot multi-class multi-object tracking results
                elif tracker.n_classes > 1:
                    if opt.tracker == "byte":
                        # ---------- aggregate current frame's results for each object class
                        online_tlwhs_dict = defaultdict(list)
                        online_tr_ids_dict = defaultdict(list)

                        track_ids = []
                        t_bboxes = []
                        t_class_ids = []

                        # process each object class
                        for cls_id in range(tracker.n_classes):
                            online_targets = online_dict[cls_id]
                            for track in online_targets:
                                online_tlwhs_dict[cls_id].append(track.tlwh)
                                online_tr_ids_dict[cls_id].append(
                                    track.track_id)

                                track_ids.append(
                                    str(cls_id) + "_" + str(track.track_id))
                                t_bboxes.append(track.tlwh)
                                t_class_ids.append(cls_id)

                        # トラッキングIDと連番の紐付け
                        for idx, track_id in enumerate(track_ids):
                            if track_id not in track_id_dict:   # 리스트에 없을 때 ( = 새로 추가된 것일 때)
                                new_id = len(track_id_dict)
                                track_id_dict[track_id] = new_id




                        # 사라진 것이 있는지부터 확인.

                        if disappeared_product_flag:
                            # 화면에 다시 잡히면, 일시적인 오류였던걸로 간주함.
                            if disappeared_product_id in track_ids:
                                disappeared_product_flag = False
                            else:
                                dp_frame_count -= 1
                                # 60프레임 이후에도 여전히 화면에 안 잡히면, 사라진 것.
                                if dp_frame_count == 0:
                                    disappeared_product_flag = False
                                    product = products_list[disappeared_product_id]
                                    print(f"Frame: {frame_id}")
                                    print(f"{id2cls[product[0]]} disappeared.")
                                    if is_on_the_shelves(product[3]):       # position
                                        c_tid = product[2]
                                        p_cid = product[0]
                                        # 구매 목록에서 한 개 뺌.
                                        purchase_table[c_tid][p_cid] -= 1
                                        print(f"Customer #{c_tid} put {id2cls[p_cid]} back.")
                                    # 상품 목록에서 삭제.
                                    del products_list[disappeared_product_id]
                                    print()
                        else:
                            for product in products_list:
                                if product not in track_ids:
                                    disappeared_product_flag = True
                                    disappeared_product_id = product
                                    dp_frame_count = DPF

                        if disappeared_customer_flag:
                            # 화면에 다시 잡히면, 일시적인 오류였던걸로 간주함.
                            if disappeared_customer_id in track_ids:
                                disappeared_customer_flag = False
                            else:
                                dc_frame_count -= 1
                                # 60프레임 이후에도 여전히 화면에 안 잡히면, 사라진 것.
                                if dc_frame_count == 0:
                                    disappeared_customer_flag = False
                                    print(f"Frame: {frame_id}")
                                    print(f"Customer #{customers_list[disappeared_customer_id][0]} disappeared.")
                                    print()                                        
                                    del customers_list[disappeared_customer_id]

                        else:
                            for customer in customers_list:
                                if customer not in track_ids:
                                    disappeared_customer_flag = True
                                    disappeared_customer_id = customer
                                    dc_frame_count = DCF

                        # 이제, 새로 들어온 것들을 처리한다.
                        for idx, track_id in enumerate(track_ids):

                            # 화면에 잡히는 손님과 상품의 위치를 매 프레임마다 갱신해주어야 한다.
                            # 일시적으로 track이 안 될 때 발생할 수 있는 오류를 방지하기 위함이다.
                            class_id = int(t_class_ids[idx])

                            # Person
                            if class_id == 10:
                                # 기존의 것이든, 새로 들어온 것이든, 정보가 업데이트됨.
                                c_tid = int(track_id_dict[track_id])
                                c_pos = position(t_bboxes[idx])
                                if track_id not in customers_list:
                                    print(f"Frame: {frame_id}")
                                    print(f"Customer #{c_tid} entered.")
                                    print()
                                customers_list[track_id] = [c_tid, c_pos]

                            elif class_id in range(10):        # 물품 id들 중 하나이면...
                                p_pos = position(t_bboxes[idx])
                                # 기존에 리스트에 있던 상품이면, 위치 정보만 업데이트 해줌.
                                if track_id in products_list:
                                    products_list[track_id][3] = p_pos
                                    # 아니면, 새로 추가해야 함. owner는 일단 0으로 설정.
                                else:
                                    new_product_flag = True
                                    np_frame_count = NPF
                                    new_product_id = track_id
                                    new_product_origin = p_pos
                                    p_cid = int(t_class_ids[idx])
                                    p_tid = int(track_id_dict[track_id])
                                    products_list[track_id] = [p_cid, p_tid, 99, p_pos]
                                

                        # 새 상품의 구매자를 결정한다.
                        if new_product_flag:
                            np_frame_count -= 1
                            if np_frame_count == 0:
                                new_product_flag = False
                                new_product_after = products_list[new_product_id][3]
                                owner_tid = products_list[new_product_id][2]
                                if customers_list and not np.all((new_product_after == new_product_origin) == True):
                                    owner_id = purchased_customer(
                                        new_product_origin, new_product_after, customers_list)
                                    owner_tid = track_id_dict[owner_id]
                                    products_list[new_product_id][2] = owner_tid
                                p_cid = products_list[new_product_id][0]
                                purchase_table[owner_tid][p_cid] += 1
                                print(f"Frame: {frame_id}")
                                print(f"Customer #{owner_tid} picked {id2cls[p_cid]}.")
                                print()





                        timer.toc()
                        online_img = plot_tracking_mc(img=img_info['raw_img'],
                                                      tlwhs_dict=online_tlwhs_dict,
                                                      obj_ids_dict=online_tr_ids_dict,
                                                      num_classes=tracker.n_classes,
                                                      frame_id=frame_id + 1,
                                                      fps=1.0 / timer.average_time,
                                                      id2cls=id2cls)
                    elif opt.tracker == "oc":

                        track_ids = []
                        t_bboxes = []
                        t_class_ids = []
                        
                        for k, v in online_dict.items():
                            x1y1x2y2_list = v[:, :-1]
                            id_list = v[:, -1]
                            for (x1y1x2y2, tr_id) in zip(x1y1x2y2_list, id_list):
                                track_ids.append(
                                    str(k) + "_" + str(tr_id))
                                t_bboxes.append(x1y1x2y2)
                                t_class_ids.append(k)
                            
                        for idx, track_id in enumerate(track_ids):
                            if track_id not in track_id_dict:   # 리스트에 없을 때 ( = 새로 추가된 것일 때)
                                new_id = len(track_id_dict)
                                track_id_dict[track_id] = new_id


                        # 사라진 것이 있는지부터 확인.

                        if disappeared_product_flag:
                            # 화면에 다시 잡히면, 일시적인 오류였던걸로 간주함.
                            if disappeared_product_id in track_ids:
                                disappeared_product_flag = False
                            else:
                                dp_frame_count -= 1
                                # 60프레임 이후에도 여전히 화면에 안 잡히면, 사라진 것.
                                if dp_frame_count == 0:
                                    disappeared_product_flag = False
                                    product = products_list[disappeared_product_id]
                                    print(f"Frame: {frame_id}")
                                    print(f"{id2cls[product[0]]} disappeared.")
                                    # if is_on_the_shelves2(product[3]):       # position
                                    #     c_tid = product[2]
                                    #     p_cid = product[0]
                                    #     # 구매 목록에서 한 개 뺌.
                                    #     purchase_table[c_tid][p_cid] -= 1
                                    #     print(f"Customer #{c_tid} put {id2cls[p_cid]} back.")
                                    # 상품 목록에서 삭제.
                                    del products_list[disappeared_product_id]
                                    print()
                        else:
                            for product in products_list:
                                if product not in track_ids:
                                    disappeared_product_flag = True
                                    disappeared_product_id = product
                                    dp_frame_count = DPF

                        if disappeared_customer_flag:
                            # 화면에 다시 잡히면, 일시적인 오류였던걸로 간주함.
                            if disappeared_customer_id in track_ids:
                                disappeared_customer_flag = False
                            else:
                                dc_frame_count -= 1
                                # 60프레임 이후에도 여전히 화면에 안 잡히면, 사라진 것.
                                if dc_frame_count == 0:
                                    disappeared_customer_flag = False
                                    print(f"Frame: {frame_id}")
                                    print(f"Customer #{customers_list[disappeared_customer_id][0]} disappeared.")
                                    print()                                        
                                    del customers_list[disappeared_customer_id]

                        else:
                            for customer in customers_list:
                                if customer not in track_ids:
                                    disappeared_customer_flag = True
                                    disappeared_customer_id = customer
                                    dc_frame_count = DCF

                        # 이제, 새로 들어온 것들을 처리한다.
                        for idx, track_id in enumerate(track_ids):

                            # 화면에 잡히는 손님과 상품의 위치를 매 프레임마다 갱신해주어야 한다.
                            # 일시적으로 track이 안 될 때 발생할 수 있는 오류를 방지하기 위함이다.
                            class_id = int(t_class_ids[idx])

                            # Person
                            if class_id == 10:
                                # 기존의 것이든, 새로 들어온 것이든, 정보가 업데이트됨.
                                c_tid = int(track_id_dict[track_id])
                                c_pos = position2(t_bboxes[idx])
                                if track_id not in customers_list:
                                    print(f"Frame: {frame_id}")
                                    print(f"Customer #{c_tid} entered.")
                                    print()
                                customers_list[track_id] = [c_tid, c_pos]

                            elif class_id in range(10):        # 물품 id들 중 하나이면...
                            
                                p_pos = position2(t_bboxes[idx])
                                if is_on_the_shelves2(p_pos):
                                # 기존에 리스트에 있던 상품이면, 위치 정보만 업데이트 해줌.
                                    if track_id in products_list:
                                        products_list[track_id][3] = p_pos
                                    # 아니면, 새로 추가해야 함. owner는 일단 0으로 설정.
                                    else:
                                        new_product_flag = True
                                        np_frame_count = NPF
                                        new_product_id = track_id
                                        new_product_origin = p_pos
                                        p_cid = int(t_class_ids[idx])
                                        p_tid = int(track_id_dict[track_id])
                                        products_list[track_id] = [p_cid, p_tid, 99, p_pos]

                        # 새 상품의 구매자를 결정한다.
                        if new_product_flag:
                            np_frame_count -= 1
                            if np_frame_count == 0:
                                new_product_flag = False
                                new_product_after = products_list[new_product_id][3]
                                owner_tid = products_list[new_product_id][2]
                                if customers_list and not np.all((new_product_after == new_product_origin) == True):
                                    owner_id = purchased_customer(
                                        new_product_origin, new_product_after, customers_list)
                                    owner_tid = track_id_dict[owner_id]
                                    products_list[new_product_id][2] = owner_tid
                                p_cid = products_list[new_product_id][0]
                                purchase_table[owner_tid][p_cid] += 1
                                print(f"Frame: {frame_id}")
                                print(f"Customer #{owner_tid} picked {id2cls[p_cid]}.")
                                print()
                        
                        timer.toc()
                        online_img = plot_tracking_ocsort(img=img_info['raw_img'],
                                                          tracks_dict=online_dict,
                                                          frame_id=frame_id + 1,
                                                          fps=1.0 / timer.average_time,
                                                          id2cls=id2cls)

            else:
                timer.toc()
                online_img = img_info['raw_img']

            if opt.save_result:
                vid_writer.write(online_img)

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            print("Read frame {:d} failed!".format(frame_id))
            break

        ## ----- update frame id
        frame_id += 1

    print_receipt(purchase_table, id2cls)              ####################
    print("{:s} saved.".format(vid_save_path))


def imageflow_demo(predictor, vis_dir, current_time, opt):
    """
    :param predictor:
    :param vis_dir:
    :param current_time:
    :param opt:
    :return:
    """
    if opt.demo == "videos":
        if os.path.isdir(opt.video_dir):
            mp4_path_list = [opt.video_dir + "/" + x for x in os.listdir(opt.video_dir)
                             if x.endswith(".mp4")]
            mp4_path_list.sort()
            if len(mp4_path_list) == 0:
                logger.error("empty mp4 video list.")
                exit(-1)

            for video_path in mp4_path_list:
                if os.path.isfile(video_path):
                    video_name = os.path.split(video_path)[-1][:-4]
                    print("\nStart tracking video {:s} offline...".format(video_name))

                    ## ----- video capture
                    cap = cv2.VideoCapture(video_path)
                    ## -----

                    save_dir = os.path.join(vis_dir, video_name)
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)
                    current_time = time.localtime()
                    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                    save_path = os.path.join(save_dir, current_time + ".mp4")

                    ## ---------- Get tracking results
                    track_video(predictor, cap, save_path, opt)
                    ## ----------

                    print("{:s} tracking offline done.".format(video_name))

    elif opt.demo == "video":
        opt.path = os.path.abspath(opt.path)

        if os.path.isfile(opt.path):
            video_name = opt.path.split("/")[-1][:-4]
            print("Start tracking video {:s} offline...".format(video_name))

            if not os.path.isfile(opt.path):
                logger.error("invalid path: {:s}, exit now!".format(opt.path))
                exit(-1)

            ## ----- video capture
            cap = cv2.VideoCapture(opt.path)
            ## -----

            save_dir = os.path.join(vis_dir, video_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            current_time = time.localtime()
            current_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_path = os.path.join(save_dir, current_time + ".mp4")

            ## ---------- Get tracking results
            track_video(predictor, cap, save_path, opt)
            ## ----------

            print("{:s} tracking done offline.".format(video_name))
        else:
            logger.error("invalid video path: {:s}, exit now!".format(opt.path))
            exit(-1)

    elif opt.demo == "camera":
        if os.path.isfile(opt.path):
            cap = cv2.VideoCapture(opt.camid)
            video_name = opt.path.split("/")[-1][:-4]
            save_dir = os.path.join(vis_dir, video_name)
            save_path = os.path.join(save_dir, "camera.mp4")


def run(exp, opt):
    """
    :param exp:
    :param opt:
    :return:
    """
    if not opt.experiment_name:
        opt.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, opt.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if opt.save_result:
        vis_dir = os.path.join(file_name, "track_vis")
        os.makedirs(vis_dir, exist_ok=True)

    if opt.trt:
        opt.device = "gpu"

    logger.info("Args: {}".format(opt))
    if opt.conf is not None:
        exp.test_conf = opt.conf
    if opt.nms is not None:
        exp.nms_thresh = opt.nms
    if opt.tsize is not None:
        exp.test_size = (opt.tsize, opt.tsize)

    ## ---------- whether to do ReID
    if hasattr(exp, "reid"):
        exp.reid = opt.reid

    ## ----- Define the network
    net = exp.get_model()
    if not opt.debug:
        logger.info("Model Summary: {}".format(get_model_info(net, exp.test_size)))
    if opt.device == "gpu":
        net.cuda()
    net.eval()
    ## -----

    if not opt.trt:
        if opt.ckpt is None:
            ckpt_file_path = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file_path = opt.ckpt
        ckpt_file_path = os.path.abspath(ckpt_file_path)

        logger.info("Loading checkpoint...")
        ckpt = torch.load(ckpt_file_path, map_location="cpu")

        # load the model state dict
        net.load_state_dict(ckpt["model"])
        logger.info("Checkpoint {:s} loaded done.".format(ckpt_file_path))

    if opt.fuse:
        logger.info("\tFusing model...")
        net = fuse_model(net)

    if opt.fp16:
        net = net.half()  # to FP16

    if opt.trt:
        assert not opt.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(trt_file), \
            "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        net.head.decode_in_inference = False
        decoder = net.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    ## ---------- Define the predictor
    predictor = Predictor(net, exp, trt_file, decoder, opt.device, opt.fp16, opt.reid)
    ## ----------

    current_time = time.localtime()
    if opt.demo == "image":
        image_demo(predictor, vis_dir, opt.path, current_time, opt.save_result)
    elif opt.demo == "video" or opt.demo == "videos" or opt.demo == "webcam":
        imageflow_demo(predictor, vis_dir, current_time, opt)


if __name__ == "__main__":
    opt = make_parser().parse_args()
    exp = get_exp(opt.exp_file, opt.name)

    class_names = opt.class_names.split(",")
    opt.class_names = class_names
    exp.class_names = class_names
    exp.n_classes = len(exp.class_names)
    print("Number of classes: ", exp.n_classes)

    ## ----- run the tracking
    run(exp, opt)
