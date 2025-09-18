#!/usr/bin/env python3
"""
摄像头检测脚本
用于检测系统中可用的摄像头设备
"""

import cv2
import sys

def test_cameras(max_cameras=10):
    """测试可用的摄像头"""
    available_cameras = []
    
    print("正在检测可用的摄像头...")
    print("-" * 50)
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # 尝试读取一帧来确认摄像头真的可用
            ret, frame = cap.read()
            if ret:
                # 获取摄像头信息
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps
                })
                
                print(f"摄像头 {i}: {width}x{height} @ {fps:.1f}fps - 可用")
            else:
                print(f"摄像头 {i}: 无法读取帧")
        else:
            print(f"摄像头 {i}: 不可用")
        
        cap.release()
    
    print("-" * 50)
    print(f"找到 {len(available_cameras)} 个可用摄像头")
    
    if available_cameras:
        print("\n使用方法:")
        for cam in available_cameras:
            print(f"  python OV-HandTracker.py -i {cam['index']}")
    
    return available_cameras

if __name__ == "__main__":
    test_cameras()
