import numpy as np
from collections import namedtuple
import mediapipe_utils as mpu
import cv2
from pathlib import Path
from FPS import FPS, now
import argparse
import os
from openvino import Core
import asyncio
import websockets
import json
import base64
from collections import deque
from datetime import datetime

# 设置Qt环境变量以解决Docker容器中的Qt平台插件问题
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_X11_NO_MITSHM'] = '1'
os.environ['QT_DEBUG_PLUGINS'] = '0'

class DynamicGestureProcessor:
    """动态手势处理器"""
    
    def __init__(self, window_size=10, min_confidence=0.7):
        """
        初始化动态手势处理器
        
        Args:
            window_size: 滑动窗口大小（帧数）
            min_confidence: 最小置信度阈值
        """
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.gesture_history = deque(maxlen=window_size)  # 存储手势历史
        self.last_gesture = None
        self.gesture_count = 0
        
        # 定义动态手势模式
        self.dynamic_patterns = {
            "CLOSE_GESTURE": {
                "pattern": ["FIVE", "FIST"],
                "description": "从张开到握拳"
            },
            "OPEN_GESTURE": {
                "pattern": ["FIST", "FIVE"],
                "description": "从握拳到张开"
            },
            "PEACE_WAVE": {
                "pattern": ["PEACE", "FIVE", "PEACE"],
                "description": "和平手势挥手"
            },
            "THUMBS_UP_DOWN": {
                "pattern": ["OK", "FIST", "OK"],
                "description": "拇指上下"
            },
            "FINGER_COUNT_UP": {
                "pattern": ["FIST", "ONE", "TWO", "FIVE"],
                "description": "逐指张开"
            }
        }
    
    async def process_frame(self, hand_info):
        """
        处理单帧手势信息
        
        Args:
            hand_info: 包含手势信息的字典
        """
        if not hand_info or not hand_info.get('gesture'):
            return
        
        current_gesture = hand_info['gesture']
        confidence = hand_info.get('score', 0)
        
        # 只处理置信度足够高的手势
        if confidence < self.min_confidence:
            return
        
        # 添加到历史记录
        self.gesture_history.append({
            'gesture': current_gesture,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
        
        # 检查动态手势模式
        await self._check_dynamic_patterns()
    
    async def _check_dynamic_patterns(self):
        """检查动态手势模式"""
        if len(self.gesture_history) < 2:
            return
        
        # 获取最近的手势序列
        recent_gestures = [frame['gesture'] for frame in self.gesture_history]
        
        # 检查每个预定义模式
        for pattern_name, pattern_info in self.dynamic_patterns.items():
            if self._matches_pattern(recent_gestures, pattern_info['pattern']):
                await self._trigger_dynamic_gesture(pattern_name, pattern_info['description'])
                break
    
    def _matches_pattern(self, gesture_sequence, pattern):
        """
        检查手势序列是否匹配模式
        
        Args:
            gesture_sequence: 实际手势序列
            pattern: 要匹配的模式
            
        Returns:
            bool: 是否匹配
        """
        if len(gesture_sequence) < len(pattern):
            return False
        
        # 检查最近的序列是否匹配模式
        recent_sequence = gesture_sequence[-len(pattern):]
        
        # 允许模式匹配有一定的容错性
        matches = 0
        for i, expected_gesture in enumerate(pattern):
            if i < len(recent_sequence) and recent_sequence[i] == expected_gesture:
                matches += 1
        
        # 至少80%匹配才认为是有效模式
        return matches >= len(pattern) * 0.8
    
    async def _trigger_dynamic_gesture(self, pattern_name, description):
        """触发动态手势事件"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] 动态手势检测: {pattern_name} - {description}")
        
        # 这里可以添加更多处理逻辑，比如：
        # - 发送WebSocket消息给客户端
        # - 触发特定的动作
        # - 记录到日志文件等
        
        # 清空历史记录，避免重复触发
        self.gesture_history.clear()
    
    def get_gesture_statistics(self):
        """获取手势统计信息"""
        if not self.gesture_history:
            return {}
        
        gesture_counts = {}
        for frame in self.gesture_history:
            gesture = frame['gesture']
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        return {
            'total_frames': len(self.gesture_history),
            'gesture_distribution': gesture_counts,
            'window_size': self.window_size
        }

# 全局动态手势处理器实例（将在main函数中初始化）
dynamic_processor = None

class HandTracker:
    def __init__(self, input_src=None,
                pd_xml="models/palm_detection_FP32.xml", 
                pd_device="GPU",
                pd_score_thresh=0.5, pd_nms_thresh=0.3,
                use_lm=True,
                lm_xml="models/hand_landmark_FP32.xml",
                lm_device="GPU",
                lm_score_threshold=0.5,
                use_gesture=False,
                crop=False,
                no_gui=False,
                show_pd_box=None, show_pd_kps=None, show_rot_rect=None,
                show_landmarks=None, show_handedness=None, show_scores=None,
                show_gesture_display=None, show_original_video=None):
        
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.use_lm = use_lm
        self.lm_score_threshold = lm_score_threshold
        self.use_gesture = use_gesture
        self.crop = crop
        self.no_gui = no_gui
        
        # 跟踪摄像头索引
        self.current_camera_index = None
        
        # 显示原始视频窗口的标志
        self.show_original_video = True
        
        if input_src.endswith('.jpg') or input_src.endswith('.png') :
            self.image_mode = True
            self.img = cv2.imread(input_src)
        else:
            self.image_mode = False
            if input_src.isdigit():
                input_src = int(input_src)
                self.current_camera_index = input_src
            else:
                self.current_camera_index = 0  # 默认摄像头
            self.cap = cv2.VideoCapture(input_src)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
        # Create SSD anchors 
        # https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt
        anchor_options = mpu.SSDAnchorOptions(num_layers=4, 
                                min_scale=0.1484375,
                                max_scale=0.75,
                                input_size_height=128,
                                input_size_width=128,
                                anchor_offset_x=0.5,
                                anchor_offset_y=0.5,
                                strides=[8, 16, 16, 16],
                                aspect_ratios= [1.0],
                                reduce_boxes_in_lowest_layer=False,
                                interpolated_scale_aspect_ratio=1.0,
                                fixed_anchor_size=True)
        self.anchors = mpu.generate_anchors(anchor_options)
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

        # Load Openvino models
        self.load_models(pd_xml, pd_device, lm_xml, lm_device)

        # Rendering flags - 使用传入参数或默认值
        if self.use_lm:
            self.show_pd_box = show_pd_box if show_pd_box is not None else False
            self.show_pd_kps = show_pd_kps if show_pd_kps is not None else False
            self.show_rot_rect = show_rot_rect if show_rot_rect is not None else False
            self.show_handedness = show_handedness if show_handedness is not None else False
            self.show_landmarks = show_landmarks if show_landmarks is not None else True
            self.show_scores = show_scores if show_scores is not None else False
            self.show_gesture = show_gesture_display if show_gesture_display is not None else self.use_gesture
        else:
            self.show_pd_box = show_pd_box if show_pd_box is not None else True
            self.show_pd_kps = show_pd_kps if show_pd_kps is not None else False
            self.show_rot_rect = show_rot_rect if show_rot_rect is not None else False
            self.show_scores = show_scores if show_scores is not None else False
        
        # 原始视频窗口显示控制
        self.show_original_video = show_original_video if show_original_video is not None else True
        

    def load_models(self, pd_xml, pd_device, lm_xml, lm_device):

        print("Loading OpenVINO Runtime")
        self.core = Core()
        print("Device info:")
        print(f"{' '*8}{pd_device}")
        try:
            print(f"{' '*8}OpenVINO Runtime version: {self.core.get_property('RUNTIME_VERSION', pd_device)}")
        except:
            print(f"{' '*8}OpenVINO Runtime loaded successfully")

        # Palm detection model
        print("Palm Detection model - Reading model file:\n\t{}".format(pd_xml))
        self.pd_model = self.core.read_model(pd_xml)
        # Input tensor: input - shape: [1, 3, 128, 128]
        # Output tensor: classificators - shape: [1, 896, 1] : scores
        # Output tensor: regressors - shape: [1, 896, 18] : bboxes
        self.pd_input_tensor = self.pd_model.input(0)
        print(f"Input tensor: {list(self.pd_input_tensor.names)[0] if self.pd_input_tensor.names else 'unnamed'} - shape: {self.pd_input_tensor.shape}")
        _,_,self.pd_h,self.pd_w = self.pd_input_tensor.shape
        for output in self.pd_model.outputs:
            output_name = list(output.names)[0] if output.names else 'unnamed'
            print(f"Output tensor: {output_name} - shape: {output.shape}")
            if "classificators" in output_name:
                self.pd_scores = output_name
            elif "regressors" in output_name:
                self.pd_bboxes = output_name
        print("Loading palm detection model into the device")
        self.pd_compiled_model = self.core.compile_model(self.pd_model, pd_device)
        self.pd_infer_time_cumul = 0
        self.pd_infer_nb = 0

        self.infer_nb = 0
        self.infer_time_cumul = 0

        # Landmarks model
        if self.use_lm:
            if lm_device != pd_device:
                print("Device info:")
                print(f"{' '*8}{lm_device}")
                try:
                    print(f"{' '*8}OpenVINO Runtime version: {self.core.get_property('RUNTIME_VERSION', lm_device)}")
                except:
                    print(f"{' '*8}OpenVINO Runtime loaded successfully")

            print("Landmark model - Reading model file:\n\t{}".format(lm_xml))
            self.lm_model = self.core.read_model(lm_xml)
            # Input tensor: input_1 - shape: [1, 3, 224, 224]
            # Output tensor: Identity_1 - shape: [1, 1]
            # Output tensor: Identity_2 - shape: [1, 1]
            # Output tensor: Identity_dense/BiasAdd/Add - shape: [1, 63]
            self.lm_input_tensor = self.lm_model.input(0)
            print(f"Input tensor: {list(self.lm_input_tensor.names)[0] if self.lm_input_tensor.names else 'unnamed'} - shape: {self.lm_input_tensor.shape}")
            _,_,self.lm_h,self.lm_w = self.lm_input_tensor.shape
            # Batch reshaping if lm_2 is True
            for output in self.lm_model.outputs:
                output_name = list(output.names)[0] if output.names else 'unnamed'
                print(f"Output tensor: {output_name} - shape: {output.shape}")
                if "Identity_1" in output_name:
                    self.lm_score = output_name
                elif "Identity_2" in output_name:
                    self.lm_handedness = output_name
                elif "Identity_dense" in output_name:
                    self.lm_landmarks = output_name
            print("Loading landmark model to the device")
            self.lm_compiled_model = self.core.compile_model(self.lm_model, lm_device)
            self.lm_infer_time_cumul = 0
            self.lm_infer_nb = 0
            self.lm_hand_nb = 0

    
    def pd_postprocess(self, inference):
        scores = np.squeeze(inference[self.pd_scores])  # 896
        bboxes = inference[self.pd_bboxes][0] # 896x18
        # Decode bboxes
        self.regions = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors)
        # Non maximum suppression
        self.regions = mpu.non_max_suppression(self.regions, self.pd_nms_thresh)
        if self.use_lm:
            mpu.detections_to_rect(self.regions)
            mpu.rect_transformation(self.regions, self.frame_size, self.frame_size)

    def pd_render(self, frame):
        for r in self.regions:
            if self.show_pd_box:
                box = (np.array(r.pd_box) * self.frame_size).astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)
            if self.show_pd_kps:
                for i,kp in enumerate(r.pd_kps):
                    x = int(kp[0] * self.frame_size)
                    y = int(kp[1] * self.frame_size)
                    cv2.circle(frame, (x, y), 6, (0,0,255), -1)
                    cv2.putText(frame, str(i), (x, y+12), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            if self.show_scores:
                cv2.putText(frame, f"Palm score: {r.pd_score:.2f}", 
                        (int(r.pd_box[0] * self.frame_size+10), int((r.pd_box[1]+r.pd_box[3])*self.frame_size+60)), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)

    def recognize_gesture(self, r):           

        # Finger states
        # state: -1=unknown, 0=close, 1=open
        d_3_5 = mpu.distance(r.landmarks[3], r.landmarks[5])
        d_2_3 = mpu.distance(r.landmarks[2], r.landmarks[3])
        angle0 = mpu.angle(r.landmarks[0], r.landmarks[1], r.landmarks[2])
        angle1 = mpu.angle(r.landmarks[1], r.landmarks[2], r.landmarks[3])
        angle2 = mpu.angle(r.landmarks[2], r.landmarks[3], r.landmarks[4])
        r.thumb_angle = angle0+angle1+angle2
        if angle0+angle1+angle2 > 460 and d_3_5 / d_2_3 > 1.2: 
            r.thumb_state = 1
        else:
            r.thumb_state = 0

        if r.landmarks[8][1] < r.landmarks[7][1] < r.landmarks[6][1]:
            r.index_state = 1
        elif r.landmarks[6][1] < r.landmarks[8][1]:
            r.index_state = 0
        else:
            r.index_state = -1

        if r.landmarks[12][1] < r.landmarks[11][1] < r.landmarks[10][1]:
            r.middle_state = 1
        elif r.landmarks[10][1] < r.landmarks[12][1]:
            r.middle_state = 0
        else:
            r.middle_state = -1

        if r.landmarks[16][1] < r.landmarks[15][1] < r.landmarks[14][1]:
            r.ring_state = 1
        elif r.landmarks[14][1] < r.landmarks[16][1]:
            r.ring_state = 0
        else:
            r.ring_state = -1

        if r.landmarks[20][1] < r.landmarks[19][1] < r.landmarks[18][1]:
            r.little_state = 1
        elif r.landmarks[18][1] < r.landmarks[20][1]:
            r.little_state = 0
        else:
            r.little_state = -1

        # Gesture
        if r.thumb_state == 1 and r.index_state == 1 and r.middle_state == 1 and r.ring_state == 1 and r.little_state == 1:
            r.gesture = "FIVE"
        elif r.thumb_state == 0 and r.index_state == 0 and r.middle_state == 0 and r.ring_state == 0 and r.little_state == 0:
            r.gesture = "FIST"
        elif r.thumb_state == 1 and r.index_state == 0 and r.middle_state == 0 and r.ring_state == 0 and r.little_state == 0:
            r.gesture = "OK" 
        elif r.thumb_state == 0 and r.index_state == 1 and r.middle_state == 1 and r.ring_state == 0 and r.little_state == 0:
            r.gesture = "PEACE"
        elif r.thumb_state == 0 and r.index_state == 1 and r.middle_state == 0 and r.ring_state == 0 and r.little_state == 0:
            r.gesture = "ONE"
        elif r.thumb_state == 1 and r.index_state == 1 and r.middle_state == 0 and r.ring_state == 0 and r.little_state == 0:
            r.gesture = "TWO"
        elif r.thumb_state == 1 and r.index_state == 1 and r.middle_state == 1 and r.ring_state == 0 and r.little_state == 0:
            r.gesture = "THREE"
        elif r.thumb_state == 0 and r.index_state == 1 and r.middle_state == 1 and r.ring_state == 1 and r.little_state == 1:
            r.gesture = "FOUR"
        else:
            r.gesture = None
            
    def lm_postprocess(self, region, inference):
        region.lm_score = np.squeeze(inference[self.lm_score])    
        region.handedness = np.squeeze(inference[self.lm_handedness])
        lm_raw = np.squeeze(inference[self.lm_landmarks])
        
        lm = []
        for i in range(int(len(lm_raw)/3)):
            # x,y,z -> x/w,y/h,z/w (here h=w)
            lm.append(lm_raw[3*i:3*(i+1)]/self.lm_w)
        region.landmarks = lm
        if self.use_gesture: self.recognize_gesture(region)


    
    def lm_render(self, frame, region):
        if region.lm_score > self.lm_score_threshold:
            if self.show_rot_rect:
                cv2.polylines(frame, [np.array(region.rect_points)], True, (0,255,255), 2, cv2.LINE_AA)
            if self.show_landmarks:
                src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
                dst = np.array([ (x, y) for x,y in region.rect_points[1:]], dtype=np.float32) # region.rect_points[0] is left bottom point !
                mat = cv2.getAffineTransform(src, dst)
                lm_xy = np.expand_dims(np.array([(l[0], l[1]) for l in region.landmarks]), axis=0)
                lm_xy = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int32)
                list_connections = [[0, 1, 2, 3, 4], 
                                    [0, 5, 6, 7, 8], 
                                    [5, 9, 10, 11, 12],
                                    [9, 13, 14 , 15, 16],
                                    [13, 17],
                                    [0, 17, 18, 19, 20]]
                lines = [np.array([lm_xy[point] for point in line]) for line in list_connections]
                cv2.polylines(frame, lines, False, (255, 0, 0), 2, cv2.LINE_AA)
                if self.use_gesture:
                    # color depending on finger state (1=open, 0=close, -1=unknown)
                    color = { 1: (0,255,0), 0: (0,0,255), -1:(0,255,255)}
                    radius = 6
                    cv2.circle(frame, (lm_xy[0][0], lm_xy[0][1]), radius, color[-1], -1)
                    for i in range(1,5):
                        cv2.circle(frame, (lm_xy[i][0], lm_xy[i][1]), radius, color[region.thumb_state], -1)
                    for i in range(5,9):
                        cv2.circle(frame, (lm_xy[i][0], lm_xy[i][1]), radius, color[region.index_state], -1)
                    for i in range(9,13):
                        cv2.circle(frame, (lm_xy[i][0], lm_xy[i][1]), radius, color[region.middle_state], -1)
                    for i in range(13,17):
                        cv2.circle(frame, (lm_xy[i][0], lm_xy[i][1]), radius, color[region.ring_state], -1)
                    for i in range(17,21):
                        cv2.circle(frame, (lm_xy[i][0], lm_xy[i][1]), radius, color[region.little_state], -1)
                else:
                    for x,y in lm_xy:
                        cv2.circle(frame, (x, y), 6, (0,128,255), -1)
            if self.show_handedness:
                cv2.putText(frame, f"RIGHT {region.handedness:.2f}" if region.handedness > 0.5 else f"LEFT {1-region.handedness:.2f}", 
                        (int(region.pd_box[0] * self.frame_size+10), int((region.pd_box[1]+region.pd_box[3])*self.frame_size+20)), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0) if region.handedness > 0.5 else (0,0,255), 2)
            if self.show_scores:
                cv2.putText(frame, f"Landmark score: {region.lm_score:.2f}", 
                        (int(region.pd_box[0] * self.frame_size+10), int((region.pd_box[1]+region.pd_box[3])*self.frame_size+90)), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
            if self.use_gesture and self.show_gesture:
                cv2.putText(frame, region.gesture, (int(region.pd_box[0]*self.frame_size+10), int(region.pd_box[1]*self.frame_size-50)), 
                        cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)

        

    def run(self):
        """同步版本的运行方法，用于直接运行"""
        self.fps = FPS(mean_nb_frames=20)

        nb_pd_inferences = 0
        nb_lm_inferences = 0
        glob_pd_rtrip_time = 0
        glob_lm_rtrip_time = 0
        while True:
            self.fps.update()
            if self.image_mode:
                vid_frame = self.img
            else:
                ok, vid_frame = self.cap.read()
                if not ok:
                    break
            h, w = vid_frame.shape[:2]
            if self.crop:
                # Cropping the long side to get a square shape
                self.frame_size = min(h, w)
                dx = (w - self.frame_size) // 2
                dy = (h - self.frame_size) // 2
                video_frame = vid_frame[dy:dy+self.frame_size, dx:dx+self.frame_size]
            else:
                # Padding on the small side to get a square shape
                self.frame_size = max(h, w)
                pad_h = int((self.frame_size - h)/2)
                pad_w = int((self.frame_size - w)/2)
                video_frame = cv2.copyMakeBorder(vid_frame, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT)

            # Resize image to NN square input shape
            frame_nn = cv2.resize(video_frame, (self.pd_w, self.pd_h), interpolation=cv2.INTER_AREA)
            # Transpose hxwx3 -> 1x3xhxw
            frame_nn = np.transpose(frame_nn, (2,0,1))[None,]

            annotated_frame = video_frame.copy()

            # Get palm detection
            pd_rtrip_time = now()
            input_name = list(self.pd_input_tensor.names)[0] if self.pd_input_tensor.names else 'input'
            inference = self.pd_compiled_model({input_name: frame_nn})
            glob_pd_rtrip_time += now() - pd_rtrip_time
            self.pd_postprocess(inference)
            self.pd_render(annotated_frame)
            nb_pd_inferences += 1

            # Hand landmarks
            if self.use_lm:
                for i,r in enumerate(self.regions):
                    frame_nn = mpu.warp_rect_img(r.rect_points, video_frame, self.lm_w, self.lm_h)
                    # Transpose hxwx3 -> 1x3xhxw
                    frame_nn = np.transpose(frame_nn, (2,0,1))[None,]
                    # Get hand landmarks
                    lm_rtrip_time = now()
                    lm_input_name = list(self.lm_input_tensor.names)[0] if self.lm_input_tensor.names else 'input_1'
                    inference = self.lm_compiled_model({lm_input_name: frame_nn})
                    glob_lm_rtrip_time += now() - lm_rtrip_time
                    nb_lm_inferences += 1
                    self.lm_postprocess(r, inference)
                    self.lm_render(annotated_frame, r)

            if not self.crop:
                annotated_frame = annotated_frame[pad_h:pad_h+h, pad_w:pad_w+w]

            self.fps.display(annotated_frame, orig=(50,50),color=(240,180,100))
            
            # 检查是否在无头环境中运行
            if not self.no_gui:
                try:
                    # 显示处理后的视频（带检测结果）
                    cv2.imshow("Hand Tracking Result", annotated_frame)
                    
                    # 显示原始视频输入（如果启用）
                    if self.show_original_video:
                        original_frame = vid_frame.copy()
                        
                        # 在原始视频上添加详细信息
                        y_offset = 30
                        cv2.putText(original_frame, f"Original Input - Camera: {self.current_camera_index if self.current_camera_index is not None else 'Unknown'}", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_offset += 30
                        
                        cv2.putText(original_frame, f"Resolution: {w}x{h}", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_offset += 30
                        
                        cv2.putText(original_frame, f"Mode: {'Image' if self.image_mode else 'Video'}", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_offset += 30
                        
                        # 显示FPS信息
                        fps_text = f"FPS: {self.fps.fps:.1f}"
                        cv2.putText(original_frame, fps_text, 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_offset += 30
                        
                        # 显示检测统计
                        if hasattr(self, 'regions'):
                            cv2.putText(original_frame, f"Hands Detected: {len(self.regions)}", 
                                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            y_offset += 30
                        
                        # 显示处理信息
                        if self.crop:
                            cv2.putText(original_frame, "Processing: Cropped", 
                                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        else:
                            cv2.putText(original_frame, "Processing: Padded", 
                                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        cv2.imshow("Original Video Input", original_frame)
                    else:
                        # 如果原始视频窗口被关闭，确保它被销毁
                        try:
                            cv2.destroyWindow("Original Video Input")
                        except:
                            pass
                    key = cv2.waitKey(1) 
                    if key == ord('q') or key == 27:
                        break
                    elif key == 32:
                        # Pause on space bar
                        cv2.waitKey(0)
                    elif key == ord('1'):
                        self.show_pd_box = not self.show_pd_box
                    elif key == ord('2'):
                        self.show_pd_kps = not self.show_pd_kps
                    elif key == ord('3'):
                        self.show_rot_rect = not self.show_rot_rect
                    elif key == ord('4'):
                        self.show_landmarks = not self.show_landmarks
                    elif key == ord('5'):
                        self.show_handedness = not self.show_handedness
                    elif key == ord('6'):
                        self.show_scores = not self.show_scores
                    elif key == ord('7'):
                        self.show_gesture = not self.show_gesture
                    elif key == ord('o'):
                        # 切换原始视频窗口显示
                        self.show_original_video = not self.show_original_video
                        print(f"原始视频窗口: {'开启' if self.show_original_video else '关闭'}")
                    elif key == ord('h'):
                        # 显示帮助信息
                        self.show_help()
                except cv2.error as e:
                    if "Qt platform plugin" in str(e) or "xcb" in str(e) or "not implemented" in str(e) or "GTK" in str(e):
                        print("GUI display not available, running in headless mode...")
                        print("Press Ctrl+C to stop the program")
                        self.no_gui = True
                    else:
                        raise e
            else:
                # 无头模式运行
                import time
                time.sleep(0.1)  # 避免 CPU 占用过高

        # Print some stats
        print(f"# palm detection inferences : {nb_pd_inferences}")
        print(f"# hand landmark inferences  : {nb_lm_inferences}")
        print(f"Palm detection round trip   : {glob_pd_rtrip_time/nb_pd_inferences*1000:.1f} ms")
        print(f"Hand landmark round trip    : {glob_lm_rtrip_time/nb_lm_inferences*1000:.1f} ms")

    async def process_frame_async(self, frame_data=None):
        """异步处理单帧图像，用于WebSocket服务器"""
        try:
            if frame_data is not None:
                # 从WebSocket接收的帧数据
                nparr = np.frombuffer(frame_data, np.uint8)
                vid_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if vid_frame is None:
                    return None, None
            else:
                # 从摄像头读取
                if self.image_mode:
                    vid_frame = self.img
                else:
                    ok, vid_frame = self.cap.read()
                    if not ok:
                        return None, None

            h, w = vid_frame.shape[:2]
            if self.crop:
                # Cropping the long side to get a square shape
                self.frame_size = min(h, w)
                dx = (w - self.frame_size) // 2
                dy = (h - self.frame_size) // 2
                video_frame = vid_frame[dy:dy+self.frame_size, dx:dx+self.frame_size]
            else:
                # Padding on the small side to get a square shape
                self.frame_size = max(h, w)
                pad_h = int((self.frame_size - h)/2)
                pad_w = int((self.frame_size - w)/2)
                video_frame = cv2.copyMakeBorder(vid_frame, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT)

            # Resize image to NN square input shape
            frame_nn = cv2.resize(video_frame, (self.pd_w, self.pd_h), interpolation=cv2.INTER_AREA)
            # Transpose hxwx3 -> 1x3xhxw
            frame_nn = np.transpose(frame_nn, (2,0,1))[None,]

            annotated_frame = video_frame.copy()

            # Get palm detection
            input_name = list(self.pd_input_tensor.names)[0] if self.pd_input_tensor.names else 'input'
            inference = self.pd_compiled_model({input_name: frame_nn})
            self.pd_postprocess(inference)
            self.pd_render(annotated_frame)

            # Hand landmarks
            if self.use_lm:
                for i,r in enumerate(self.regions):
                    frame_nn = mpu.warp_rect_img(r.rect_points, video_frame, self.lm_w, self.lm_h)
                    # Transpose hxwx3 -> 1x3xhxw
                    frame_nn = np.transpose(frame_nn, (2,0,1))[None,]
                    # Get hand landmarks
                    lm_input_name = list(self.lm_input_tensor.names)[0] if self.lm_input_tensor.names else 'input_1'
                    inference = self.lm_compiled_model({lm_input_name: frame_nn})
                    self.lm_postprocess(r, inference)
                    self.lm_render(annotated_frame, r)

            if not self.crop:
                annotated_frame = annotated_frame[pad_h:pad_h+h, pad_w:pad_w+w]

            # 准备返回数据
            hand_data = []
            if hasattr(self, 'regions'):
                for region in self.regions:
                    # 转换 landmarks 为可序列化的格式
                    landmarks = getattr(region, 'landmarks', [])
                    if landmarks:
                        # 将 NumPy 数组转换为 Python 列表
                        landmarks_serializable = []
                        for lm in landmarks:
                            if hasattr(lm, 'tolist'):
                                landmarks_serializable.append(lm.tolist())
                            else:
                                landmarks_serializable.append(list(lm))
                    else:
                        landmarks_serializable = []
                    
                    hand_info = {
                        'gesture': getattr(region, 'gesture', None),
                        'handedness': float(getattr(region, 'handedness', 0)),  # 确保是 Python float
                        'landmarks': landmarks_serializable,
                        'score': float(getattr(region, 'lm_score', 0))  # 确保是 Python float
                    }
                    hand_data.append(hand_info)
                    
                    # 处理动态手势（如果启用）
                    if dynamic_processor is not None:
                        await dynamic_processor.process_frame(hand_info)

            # 编码处理后的帧
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            encoded_frame = base64.b64encode(buffer).decode('utf-8')

            return {
                'frame': encoded_frame,
                'hands': hand_data,
                'fps': float(getattr(self, 'fps', {}).fps if hasattr(self, 'fps') else 0)
            }, annotated_frame

        except Exception as e:
            print(f"处理帧时出错: {e}")
            return None, None

    def show_help(self):
        """显示帮助信息"""
        help_text = """
=== 手势跟踪程序控制键 ===
q 或 ESC    - 退出程序
空格        - 暂停/继续
1          - 切换手掌检测框显示
2          - 切换手掌关键点显示
3          - 切换旋转矩形显示
4          - 切换手部关键点显示
5          - 切换左右手显示
6          - 切换分数显示
7          - 切换手势识别显示
o          - 切换原始视频窗口显示
h          - 显示此帮助信息

=== 窗口说明 ===
Hand Tracking Result - 显示处理后的视频（带检测结果）
Original Video Input - 显示原始视频输入（带详细信息）
        """
        print(help_text)

# 全局HandTracker实例
ht = None

async def handtracker_websocket_handler(websocket):
    """手部跟踪WebSocket处理器 - 接收客户端视频流，返回手势识别结果"""
    global ht
    print(f"客户端连接: {websocket.remote_address}")
    
    try:
        # 发送初始配置信息
        config = {
            'message': 'HandTracker WebSocket Server Ready',
            'gesture_support': ht.use_gesture,
            'landmark_support': ht.use_lm
        }
        await websocket.send(json.dumps(config))
        
        # 等待客户端发送视频元数据
        meta_msg = await websocket.recv()
        meta = json.loads(meta_msg)
        width, height, fps = meta["width"], meta["height"], meta["fps"]
        print(f"接收到视频参数: {width}x{height}, {fps}fps")
        
        # 初始化FPS计算器
        ht.fps = FPS(mean_nb_frames=20)
        
        # 处理视频流
        async for message in websocket:
            try:
                # 更新FPS
                ht.fps.update()
                
                # 解码视频帧
                if isinstance(message, (bytes, bytearray)):
                    nparr = np.frombuffer(message, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # 进行手部跟踪和手势识别
                        result, annotated_frame = await ht.process_frame_async(message)
                        
                        if result is not None:
                            # 更新FPS信息
                            result['fps'] = float(ht.fps.fps)
                            
                            # 发送手势识别结果到客户端
                            await websocket.send(json.dumps(result))
                        else:
                            # 发送空结果
                            empty_result = {
                                'frame': '',
                                'hands': [],
                                'fps': float(ht.fps.fps)
                            }
                            await websocket.send(json.dumps(empty_result))
                    else:
                        print("无法解码视频帧")
                else:
                    print("收到非二进制消息，忽略:", message)
                    
            except websockets.exceptions.ConnectionClosed:
                print("客户端断开连接")
                break
            except Exception as e:
                print(f"处理视频帧时出错: {e}")
                # 发送错误信息
                error_result = {
                    'frame': '',
                    'hands': [],
                    'fps': float(ht.fps.fps if hasattr(ht, 'fps') else 0),
                    'error': str(e)
                }
                try:
                    await websocket.send(json.dumps(error_result))
                except:
                    break
                
    except websockets.exceptions.ConnectionClosed:
        print("客户端断开连接")
    except Exception as e:
        print(f"WebSocket连接错误: {e}")
    finally:
        print("WebSocket连接已关闭")

async def main():
    global ht, dynamic_processor
    
    # 初始化动态手势处理器
    if args.dynamic_gestures:
        dynamic_processor = DynamicGestureProcessor(
            window_size=args.gesture_window_size,
            min_confidence=args.gesture_confidence
        )
        print(f"动态手势处理已启用 - 窗口大小: {args.gesture_window_size}, 置信度阈值: {args.gesture_confidence}")
    else:
        dynamic_processor = None
        print("动态手势处理已禁用")
    
    # 处理显示控制参数
    show_pd_box = args.show_pd_box if not args.hide_pd_box else False
    show_pd_kps = args.show_pd_kps if not args.hide_pd_kps else False
    show_rot_rect = args.show_rot_rect if not args.hide_rot_rect else False
    show_landmarks = args.show_landmarks if not args.hide_landmarks else False
    show_handedness = args.show_handedness if not args.hide_handedness else False
    show_scores = args.show_scores if not args.hide_scores else False
    show_gesture_display = args.show_gesture_display if not args.hide_gesture_display else False
    show_original_video = args.show_original_video if not args.hide_original_video else False
    
    # 创建HandTracker实例
    ht = HandTracker(input_src=args.input, 
                    pd_device=args.pd_device, 
                    use_lm= not args.no_lm, 
                    lm_device=args.lm_device,
                    use_gesture=args.gesture,
                    crop=args.crop,
                    no_gui=True,  # WebSocket模式下强制无头模式
                    show_pd_box=show_pd_box,
                    show_pd_kps=show_pd_kps,
                    show_rot_rect=show_rot_rect,
                    show_landmarks=show_landmarks,
                    show_handedness=show_handedness,
                    show_scores=show_scores,
                    show_gesture_display=show_gesture_display,
                    show_original_video=show_original_video)
    
    # 启动WebSocket服务器
    print("启动手部跟踪WebSocket服务器...")
    print("服务器地址: ws://0.0.0.0:8765")
    print("功能: 接收客户端视频流，进行手势识别，返回识别结果")
    print("\n=== 当前显示设置 ===")
    print(f"手掌检测框: {'开启' if ht.show_pd_box else '关闭'}")
    print(f"手掌关键点: {'开启' if ht.show_pd_kps else '关闭'}")
    print(f"旋转矩形: {'开启' if ht.show_rot_rect else '关闭'}")
    print(f"手部关键点: {'开启' if ht.show_landmarks else '关闭'}")
    print(f"左右手显示: {'开启' if ht.show_handedness else '关闭'}")
    print(f"分数显示: {'开启' if ht.show_scores else '关闭'}")
    print(f"手势识别: {'开启' if ht.show_gesture else '关闭'}")
    print(f"原始视频窗口: {'开启' if ht.show_original_video else '关闭'}")
    print("=" * 30)
    
    # 启动WebSocket服务器
    async with websockets.serve(handtracker_websocket_handler, "0.0.0.0", 8765):
        print("手部跟踪WebSocket服务器已启动，等待连接...")
        await asyncio.Future()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='0', 
                        help="Path to video or image file to use as input (default=%(default)s)")
    parser.add_argument('-g', '--gesture', action="store_true", 
                        help="enable gesture recognition")
    parser.add_argument("--pd_m", default="models/palm_detection_FP32.xml", type=str,
                        help="Path to an .xml file for palm detection model (default=%(default)s)")
    parser.add_argument("--pd_device", default='GPU', type=str,
                        help="Target device for the palm detection model (default=%(default)s)")  
    parser.add_argument('--no_lm', action="store_true", 
                        help="only the palm detection model is run, not the hand landmark model")
    parser.add_argument("--lm_m", default="models/hand_landmark_FP32.xml", type=str,
                        help="Path to an .xml file for landmark model (default=%(default)s)")
    parser.add_argument("--lm_device", default='GPU', type=str,
                        help="Target device for the landmark regression model (default=%(default)s)")
    parser.add_argument('-c', '--crop', action="store_true", 
                        help="center crop frames to a square shape before feeding palm detection model")
    parser.add_argument('--no_gui', action="store_true", 
                        help="run in headless mode without GUI display")
    
    # 显示控制参数
    parser.add_argument('--show_pd_box', action="store_true", 
                        help="show palm detection boxes")
    parser.add_argument('--show_pd_kps', action="store_true", 
                        help="show palm detection keypoints")
    parser.add_argument('--show_rot_rect', action="store_true", 
                        help="show rotation rectangles")
    parser.add_argument('--show_landmarks', action="store_true", 
                        help="show hand landmarks")
    parser.add_argument('--show_handedness', action="store_true", 
                        help="show left/right hand indication")
    parser.add_argument('--show_scores', action="store_true", 
                        help="show detection and landmark scores")
    parser.add_argument('--show_gesture_display', action="store_true", 
                        help="show gesture recognition results")
    parser.add_argument('--show_original_video', action="store_true", 
                        help="show original video window")
    
    # 关闭显示的参数
    parser.add_argument('--hide_pd_box', action="store_true", 
                        help="hide palm detection boxes")
    parser.add_argument('--hide_pd_kps', action="store_true", 
                        help="hide palm detection keypoints")
    parser.add_argument('--hide_rot_rect', action="store_true", 
                        help="hide rotation rectangles")
    parser.add_argument('--hide_landmarks', action="store_true", 
                        help="hide hand landmarks")
    parser.add_argument('--hide_handedness', action="store_true", 
                        help="hide left/right hand indication")
    parser.add_argument('--hide_scores', action="store_true", 
                        help="hide detection and landmark scores")
    parser.add_argument('--hide_gesture_display', action="store_true", 
                        help="hide gesture recognition results")
    parser.add_argument('--hide_original_video', action="store_true", 
                        help="hide original video window")
    
    # 动态手势处理参数
    parser.add_argument('--dynamic_gestures', action="store_true", 
                        help="enable dynamic gesture recognition")
    parser.add_argument('--gesture_window_size', type=int, default=8,
                        help="sliding window size for dynamic gesture detection (default=%(default)s)")
    parser.add_argument('--gesture_confidence', type=float, default=0.6,
                        help="minimum confidence threshold for dynamic gestures (default=%(default)s)")

    args = parser.parse_args()
    asyncio.run(main())
