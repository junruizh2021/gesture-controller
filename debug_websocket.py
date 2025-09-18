#!/usr/bin/env python3
"""
调试WebSocket服务器
"""

import asyncio
import websockets
import json

async def debug_handler(websocket, path):
    """调试WebSocket处理器"""
    print(f"客户端连接: {websocket.remote_address}, 路径: {path}")
    print(f"处理器参数: websocket={type(websocket)}, path={type(path)}")
    
    try:
        # 发送欢迎消息
        welcome = {
            'message': '调试服务器已连接',
            'status': 'ready',
            'path': path
        }
        await websocket.send(json.dumps(welcome))
        
        # 等待客户端消息
        async for message in websocket:
            print(f"收到消息: {type(message)} - {message[:100] if len(str(message)) > 100 else message}")
            
            # 回显消息
            response = {
                'echo': str(message)[:100],
                'timestamp': asyncio.get_event_loop().time()
            }
            await websocket.send(json.dumps(response))
            
    except websockets.exceptions.ConnectionClosed:
        print("客户端断开连接")
    except Exception as e:
        print(f"处理连接时出错: {e}")
    finally:
        print("连接已关闭")

async def main():
    print("启动调试WebSocket服务器...")
    print("服务器地址: ws://0.0.0.0:8765")
    
    # 测试不同的调用方式
    try:
        async with websockets.serve(debug_handler, "0.0.0.0", 8765):
            print("WebSocket服务器已启动，等待连接...")
            await asyncio.Future()
    except Exception as e:
        print(f"启动服务器失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())

