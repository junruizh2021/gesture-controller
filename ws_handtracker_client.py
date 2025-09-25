#!/usr/bin/env python3
"""
WebSocket客户端 - 发送视频到服务器并接收手部跟踪结果
"""

import asyncio
import websockets
import json
import cv2
import base64
import numpy as np

async def send_video_and_receive_results():
    """发送视频到服务器并接收手部跟踪结果"""
    uri = "ws://localhost:8765"
    
    try:
        async with websockets.connect(uri, max_size=None) as websocket:
            print("已连接到服务器，准备发送视频并接收结果")
            
            # 打开摄像头
            cap = cv2.VideoCapture(4)
            if not cap.isOpened():
                print("无法打开摄像头")
                return
            
            # 获取摄像头参数
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #width = 640
            #height = 480
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 60
            
            # 发送视频元数据
            metadata = {
                "width": width,
                "height": height,
                "fps": fps,
                "mode": "handtracking"  # 指定为手部跟踪模式
            }
            
            await websocket.send(json.dumps(metadata))
            print(f"发送视频元数据: {width}x{height}, {fps}fps")
            
            frame_count = 0
            
            # 创建任务来处理接收和发送
            async def send_frames():
                nonlocal frame_count
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 编码帧
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_bytes = buffer.tobytes()
                    
                    # 发送帧数据
                    await websocket.send(frame_bytes)
                    frame_count += 1
                    
                    # 控制发送频率
                    await asyncio.sleep(1.0/fps)
            
            async def receive_results():
                while True:
                    try:
                        # 接收服务器返回的结果
                        data = await websocket.recv()
                        result = json.loads(data)
                        
                        # 解码处理后的图像
                        if 'frame' in result:
                            frame_data = base64.b64decode(result['frame'])
                            nparr = np.frombuffer(frame_data, np.uint8)
                            processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if processed_frame is not None:
                                # 显示处理后的图像
                                cv2.imshow("Hand Tracking Result", processed_frame)
                                
                                # 显示手部信息
                                if 'hands' in result and result['hands']:
                                    print(f"帧 {result.get('frame_id', 0)}: 检测到 {len(result['hands'])} 只手")
                                    for i, hand in enumerate(result['hands']):
                                        print(f"  手 {i+1}: 手势={hand.get('gesture', 'None')}, "
                                              f"左右手={hand.get('handedness', 'Unknown')}, "
                                              f"置信度={hand.get('score', 0):.2f}")
                                
                                # 显示FPS
                                if 'fps' in result:
                                    print(f"服务器FPS: {result['fps']:.1f}")
                                
                                # 按 'q' 退出
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                            else:
                                print("无法解码处理后的图像")
                                
                    except websockets.exceptions.ConnectionClosed:
                        print("服务器连接已关闭")
                        break
                    except Exception as e:
                        print(f"接收数据时出错: {e}")
                        break
            
            # 同时运行发送和接收任务
            try:
                await asyncio.gather(
                    send_frames(),
                    receive_results()
                )
            except KeyboardInterrupt:
                print("用户中断")
            except Exception as e:
                print(f"运行出错: {e}")
                    
    except Exception as e:
        print(f"连接服务器失败: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("WebSocket客户端 - 发送视频并接收手部跟踪结果")
    print("按 'q' 键退出")
    asyncio.run(send_video_and_receive_results())
