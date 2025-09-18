#!/usr/bin/env python3
"""
摄像头分辨率检测脚本
检测摄像头的真实分辨率和可用分辨率
"""

import cv2
import numpy as np

def test_camera_resolutions(camera_index=0):
    """测试摄像头的分辨率和设置"""
    print(f"检测摄像头 {camera_index} 的分辨率信息...")
    print("=" * 60)
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_index}")
        return
    
    # 获取当前设置的分辨率
    current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    current_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"当前设置分辨率: {current_width}x{current_height}")
    print(f"当前FPS: {current_fps:.1f}")
    print()
    
    # 测试不同的分辨率设置
    test_resolutions = [
        (640, 480),   # VGA
        (800, 600),   # SVGA
        (1024, 768),  # XGA
        (1280, 720),  # HD 720p
        (1280, 960),  # SXGA
        (1600, 1200), # UXGA
        (1920, 1080), # Full HD 1080p
        (2560, 1440), # 2K
        (3840, 2160), # 4K
    ]
    
    print("测试不同分辨率设置:")
    print("-" * 40)
    
    working_resolutions = []
    
    for width, height in test_resolutions:
        # 设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 读取实际设置的分辨率
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 尝试读取一帧
        ret, frame = cap.read()
        
        if ret and frame is not None:
            actual_frame_height, actual_frame_width = frame.shape[:2]
            status = "✓ 支持"
            working_resolutions.append((actual_width, actual_height, actual_fps))
        else:
            status = "✗ 不支持"
        
        print(f"{width:4d}x{height:4d} -> {actual_width:4d}x{actual_height:4d} @ {actual_fps:5.1f}fps {status}")
    
    print()
    print("摄像头支持的常用分辨率:")
    print("-" * 40)
    
    # 测试一些常见的分辨率
    common_resolutions = [
        (320, 240),   # QVGA
        (640, 480),   # VGA
        (800, 600),   # SVGA
        (1024, 768),  # XGA
        (1280, 720),  # HD
        (1280, 1024), # SXGA
        (1600, 1200), # UXGA
        (1920, 1080), # Full HD
    ]
    
    for width, height in common_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✓ {actual_width}x{actual_height}")
        else:
            print(f"✗ {width}x{height}")
    
    print()
    print("摄像头属性信息:")
    print("-" * 40)
    
    # 获取更多摄像头属性
    properties = [
        (cv2.CAP_PROP_FRAME_WIDTH, "FRAME_WIDTH"),
        (cv2.CAP_PROP_FRAME_HEIGHT, "FRAME_HEIGHT"),
        (cv2.CAP_PROP_FPS, "FPS"),
        (cv2.CAP_PROP_FOURCC, "FOURCC"),
        (cv2.CAP_PROP_FRAME_COUNT, "FRAME_COUNT"),
        (cv2.CAP_PROP_BRIGHTNESS, "BRIGHTNESS"),
        (cv2.CAP_PROP_CONTRAST, "CONTRAST"),
        (cv2.CAP_PROP_SATURATION, "SATURATION"),
        (cv2.CAP_PROP_HUE, "HUE"),
        (cv2.CAP_PROP_GAIN, "GAIN"),
        (cv2.CAP_PROP_EXPOSURE, "EXPOSURE"),
        (cv2.CAP_PROP_AUTO_EXPOSURE, "AUTO_EXPOSURE"),
    ]
    
    for prop_id, prop_name in properties:
        try:
            value = cap.get(prop_id)
            print(f"{prop_name:15s}: {value}")
        except:
            print(f"{prop_name:15s}: 不支持")
    
    # 恢复原始设置
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, current_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, current_height)
    
    cap.release()
    
    print()
    print("=" * 60)
    print("总结:")
    print(f"- 摄像头默认分辨率: {current_width}x{current_height}")
    print(f"- 支持的分辨率数量: {len(working_resolutions)}")
    print("- 建议使用支持的最高分辨率以获得最佳质量")
    
    return working_resolutions

def test_backend_support():
    """测试不同后端的支持情况"""
    print("\n测试OpenCV后端支持:")
    print("-" * 40)
    
    backends = [
        (cv2.CAP_ANY, "CAP_ANY"),
        (cv2.CAP_V4L2, "CAP_V4L2 (Linux)"),
        (cv2.CAP_DSHOW, "CAP_DSHOW (Windows)"),
        (cv2.CAP_MSMF, "CAP_MSMF (Windows)"),
        (cv2.CAP_FFMPEG, "CAP_FFMPEG"),
        (cv2.CAP_GSTREAMER, "CAP_GSTREAMER"),
    ]
    
    for backend_id, backend_name in backends:
        try:
            cap = cv2.VideoCapture(0, backend_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"✓ {backend_name}: {width}x{height}")
                else:
                    print(f"✗ {backend_name}: 无法读取帧")
            else:
                print(f"✗ {backend_name}: 无法打开")
            cap.release()
        except Exception as e:
            print(f"✗ {backend_name}: 错误 - {e}")

if __name__ == "__main__":
    import sys
    
    camera_index = 0
    if len(sys.argv) > 1:
        camera_index = int(sys.argv[1])
    
    print("摄像头分辨率检测工具")
    print("=" * 60)
    
    # 检测摄像头分辨率
    working_resolutions = test_camera_resolutions(camera_index)
    
    # 测试后端支持
    test_backend_support()
    
    print("\n使用建议:")
    print("-" * 40)
    print("1. 如果摄像头支持更高分辨率，可以修改程序设置")
    print("2. 在OV-HandTracker.py中添加分辨率设置代码")
    print("3. 使用cap.set()方法设置所需的分辨率")
    print("4. 注意：更高分辨率可能影响处理速度")
