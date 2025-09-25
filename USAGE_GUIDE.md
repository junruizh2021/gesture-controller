# 使用指南

## 概述

本指南介绍如何使用修改后的解耦架构系统，该系统将WebSocket服务与舵机控制分离，使用任务队列进行进程间通信。

## 快速开始

### 1. 启动完整系统（推荐）

```bash
# 启动主服务器（包含WebSocket服务和舵机控制进程）
python main_server.py --dynamic_gestures --enable_servo

# 启动主服务器（禁用舵机控制）
python main_server.py --dynamic_gestures --no-servo
```

### 2. 直接启动WebSocket服务（原有方式）

```bash
# 使用原有的命令启动WebSocket服务
python ws-handtracker-server.py -g --dynamic_gestures --show_landmarks --show_scores --right_hand_only
```

## 参数说明

### 主服务器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | 0.0.0.0 | WebSocket服务器主机地址 |
| `--port` | 8765 | WebSocket服务器端口 |
| `--dynamic_gestures` | False | 启用动态手势识别 |
| `--enable_servo` | False | 启用舵机控制 |
| `--no-servo` | False | 禁用舵机控制 |
| `--servo_process_id` | servo_process | 舵机进程ID |
| `--log_level` | INFO | 日志级别 (DEBUG/INFO/WARNING/ERROR) |
| `--right_hand_only` | False | 只处理右手 |
| `--enable_wave_detection` | False | 启用挥手检测 |

### 默认配置

主服务器已经将你常用的参数设置为默认值：

- **手势识别** (`-g`): 默认启用
- **动态手势** (`--dynamic_gestures`): 默认启用
- **显示手部关键点** (`--show_landmarks`): 默认启用
- **显示分数** (`--show_scores`): 默认启用
- **只处理右手** (`--right_hand_only`): 默认启用

## 使用示例

### 1. 基本使用

```bash
# 启动系统，使用默认配置
python main_server.py --dynamic_gestures --enable_servo
```

### 2. 自定义配置

```bash
# 自定义主机和端口
python main_server.py \
    --host 192.168.1.100 \
    --port 8080 \
    --dynamic_gestures \
    --enable_servo \
    --log_level DEBUG
```

### 3. 禁用舵机控制

```bash
# 只启动WebSocket服务，不启动舵机控制
python main_server.py --dynamic_gestures --no-servo
```

### 4. 启用挥手检测

```bash
# 启用挥手检测功能
python main_server.py \
    --dynamic_gestures \
    --enable_servo \
    --enable_wave_detection
```

## 系统架构

### 解耦架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   WebSocket     │    │   任务队列      │    │   舵机控制      │
│   服务进程      │───▶│   (multiprocessing.Queue) │◀───│   进程        │
│                 │    │                 │    │                 │
│ - 手势识别      │    │ - 任务生产      │    │ - 舵机控制      │
│ - WebSocket通信 │    │ - 任务消费      │    │ - 硬件接口      │
│ - 图像处理      │    │ - 进程间通信    │    │ - 状态监控      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   主服务器      │
                    │                 │
                    │ - 进程管理      │
                    │ - 生命周期控制  │
                    │ - 错误处理      │
                    └─────────────────┘
```

### 组件说明

1. **主服务器** (`main_server.py`)
   - 协调各个组件的启动和停止
   - 管理进程生命周期
   - 处理错误和恢复

2. **WebSocket服务** (`ws-handtracker-server.py`)
   - 处理手势识别
   - WebSocket通信
   - 图像处理

3. **舵机控制进程** (`servo_process.py`)
   - 独立运行舵机控制逻辑
   - 通过任务队列接收指令
   - 硬件接口管理

4. **任务队列** (`task_queue.py`)
   - 进程间通信
   - 任务优先级管理
   - 错误重试机制

## 测试

### 1. 测试解耦系统

```bash
# 运行完整测试
python test_decoupled_system.py
```

### 2. 测试主服务器

```bash
# 测试主服务器（不启动舵机）
python test_main_server.py
```

### 3. 测试任务队列

```bash
# 直接运行任务队列测试
python task_queue.py
```

## 故障排除

### 1. 常见问题

**Q: 舵机进程启动失败**
```bash
# 检查串口权限
sudo chmod 666 /dev/ttyUSB0

# 检查硬件连接
ls -la /dev/ttyUSB*
```

**Q: WebSocket连接失败**
```bash
# 检查端口是否被占用
netstat -tlnp | grep 8765

# 检查防火墙设置
sudo ufw status
```

**Q: 任务队列阻塞**
```bash
# 检查进程状态
ps aux | grep python

# 查看日志
tail -f main_server.log
```

### 2. 调试模式

```bash
# 启用调试日志
python main_server.py --dynamic_gestures --enable_servo --log_level DEBUG
```

### 3. 日志文件

- `main_server.log` - 主服务器日志
- `servo_process_*.log` - 舵机进程日志
- `test_decoupled_system.log` - 测试日志

## 性能优化

### 1. 系统资源

- **CPU**: 建议4核以上
- **内存**: 建议8GB以上
- **GPU**: 建议使用GPU加速（默认配置）

### 2. 网络配置

- **带宽**: 建议100Mbps以上
- **延迟**: 建议<50ms
- **并发**: 支持多客户端连接

### 3. 硬件要求

- **摄像头**: 支持USB摄像头
- **舵机**: 支持串口通信的舵机
- **串口**: 确保串口设备可用

## 扩展功能

### 1. 添加新的手势模式

在 `ws-handtracker-server.py` 中的 `DynamicGestureProcessor` 类中添加：

```python
self.dynamic_patterns = {
    "YOUR_GESTURE": {
        "pattern": ["GESTURE1", "GESTURE2"],
        "description": "你的手势描述"
    },
    # ... 其他模式
}
```

### 2. 添加新的任务类型

在 `task_queue.py` 中的 `TaskType` 枚举中添加：

```python
class TaskType(Enum):
    YOUR_TASK = "your_task"
    # ... 其他类型
```

### 3. 自定义舵机控制

在 `servo_process.py` 中修改 `_servo_control_callback` 方法：

```python
def _servo_control_callback(self, gesture_name: str, description: str) -> bool:
    # 添加你的舵机控制逻辑
    pass
```

## 总结

1. **独立运行** - WebSocket服务和舵机控制可以独立运行
2. **高可用性** - 单个组件故障不影响其他组件
3. **易于扩展** - 可以轻松添加新的功能和控制逻辑
4. **性能优化** - 异步处理，提高系统响应速度
5. **易于维护** - 清晰的模块分离和接口定义