# 手势识别与云台控制

## 视觉模型

- Palm Detection model: `palm_detection.tflite` (mediapipe tag 0.8.0 04/11/2020)
- Hand Landmarks models: `hand_landmark_080.tflite` (mediapipe tag 0.8.0 04/11/2020)

## 云台型号

- 华鑫京 U50H

## 服务端架构示意图

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   WebSocket     │    │   任务队列       │    │   舵机控制      │
│   服务进程      │───▶│   (multiprocessing.Queue) │◀───│   进程        │
│                 │    │                 │    │                 │
│ - 手势识别      │    │ - 任务生产       │    │ - 舵机控制       │
│ - WebSocket通信 │    │ - 任务消费       │    │ - 硬件接口      │
│ - 图像处理      │    │ - 进程间通信     │    │ - 状态监控       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   主服务器       │
                    │                 │
                    │ - 进程管理       │
                    │ - 生命周期控制   │
                    │ - 错误处理       │
                    └─────────────────┘
```

## 核心模块

- 任务队列 (`task_queue.py`): 提供进程间通信的任务队列系统

- 舵机控制进程 (`servo_process.py`): 独立运行舵机控制逻辑

- 主服务器 (`main_server.py`): 协调各个组件的启动和停止

## 使用方法

### 启动主服务器（包含WebSocket服务和舵机控制进程）
```bash
python main_server.py --dynamic_gestures --enable_servo
```

### 启动python客户端

#### 使用UI展示手势识别结果
```bash
python ws_handtracker_client.py --enable-UI
```
#### 不使用UI展示
```bash
python ws_handtracker_client.py
```

## 手势定义

### 1. 竖起食指 (✊-->☝️): 舵机抬头并左右旋转
### 2. 挥手 (👋): 舵机上下点头