import asyncio
import websockets
import cv2
import numpy as np
import json

async def handler(websocket):
    print("客户端已连接")

    # 第一次接收 metadata（字符串）
    meta_msg = await websocket.recv()
    meta = json.loads(meta_msg)
    width, height, fps = meta["width"], meta["height"], meta["fps"]
    print(f"接收到视频参数: {width}x{height}, {fps}fps")

    output_file = "received_video.mp4"
    # 使用 mp4v 编码，保存为 mp4 文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    try:
        async for message in websocket:
            # 后续消息是二进制帧
            if isinstance(message, (bytes, bytearray)):
                nparr = np.frombuffer(message, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    out.write(frame)
            else:
                print("收到非二进制消息，忽略:", message)
    except websockets.ConnectionClosed:
        print("客户端断开连接")
    finally:
        out.release()
        print(f"视频已保存到 {output_file}")

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("服务器已启动，等待连接...")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())