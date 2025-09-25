# 解耦架构说明文档

## 概述

本文档描述了如何将WebSocket服务与舵机控制解耦，使用多进程和任务队列实现系统分离。

## 架构设计

### 系统组件

1. **WebSocket服务进程** - 处理手势识别和WebSocket通信
2. **舵机控制进程** - 独立运行舵机控制逻辑
3. **任务队列** - 进程间通信的桥梁
4. **主服务器** - 协调各个组件的启动和停止

### 架构图

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

## 核心模块

### 1. 任务队列模块 (`task_queue.py`)

**功能**: 提供进程间通信的任务队列系统

**主要类**:
- `Task` - 任务数据结构
- `TaskQueue` - 任务队列管理器
- `TaskProducer` - 任务生产者（WebSocket服务使用）
- `TaskConsumer` - 任务消费者（舵机控制进程使用）

**关键特性**:
- 支持任务优先级
- 支持任务重试机制
- 线程安全
- 支持多种任务类型

### 2. 舵机控制进程 (`servo_process.py`)

**功能**: 独立运行舵机控制逻辑

**主要类**:
- `ServoProcess` - 舵机控制进程类
- `ServoProcessManager` - 进程管理器

**关键特性**:
- 独立进程运行
- 自动重连机制
- 错误处理和恢复
- 进程状态监控

### 3. 主服务器 (`main_server.py`)

**功能**: 协调各个组件的启动和停止

**主要类**:
- `MainServer` - 主服务器类

**关键特性**:
- 组件生命周期管理
- 信号处理
- 错误恢复
- 状态监控

## 使用方法

### 1. 启动完整系统

```bash
# 启动主服务器（包含WebSocket服务和舵机控制进程）
python main_server.py --dynamic_gestures --enable_servo

# 启动主服务器（禁用舵机控制）
python main_server.py --dynamic_gestures --no-servo

# 自定义参数
python main_server.py \
    --host 0.0.0.0 \
    --port 8765 \
    --dynamic_gestures \
    --enable_servo \
    --servo_process_id servo_main \
    --log_level INFO
```

### 2. 单独启动舵机控制进程

```bash
# 直接运行舵机控制进程
python servo_process.py

# 在代码中创建舵机进程
python -c "
from servo_process import get_process_manager
manager = get_process_manager()
manager.start_servo_process('test_servo')
"
```

### 3. 测试解耦系统

```bash
# 运行完整测试
python test_decoupled_system.py

# 测试特定功能
python -c "
from test_decoupled_system import TestDecoupledSystem
tester = TestDecoupledSystem()
tester.setup_logging()
tester.test_task_queue()
"
```

## 配置参数

### 主服务器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | 0.0.0.0 | WebSocket服务器主机 |
| `--port` | 8765 | WebSocket服务器端口 |
| `--dynamic_gestures` | False | 启用动态手势识别 |
| `--enable_servo` | False | 启用舵机控制 |
| `--no-servo` | False | 禁用舵机控制 |
| `--servo_process_id` | servo_process | 舵机进程ID |
| `--log_level` | INFO | 日志级别 |

### 任务队列参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `maxsize` | 1000 | 队列最大大小 |
| `priority` | 0 | 任务优先级（数字越小优先级越高） |
| `max_retries` | 3 | 最大重试次数 |

## 任务类型

### 1. SERVO_CONTROL
舵机控制任务

```python
{
    "task_type": "SERVO_CONTROL",
    "data": {
        "gesture_name": "FIVE",
        "description": "张开手掌",
        "action": "execute_gesture"
    }
}
```

### 2. GESTURE_DETECTED
手势检测任务

```python
{
    "task_type": "GESTURE_DETECTED",
    "data": {
        "gesture": "WAVE",
        "confidence": 0.9,
        "landmarks": [...]
    }
}
```

### 3. SYSTEM_STATUS
系统状态任务

```python
{
    "task_type": "SYSTEM_STATUS",
    "data": {
        "status": "running",
        "timestamp": 1234567890
    }
}
```

### 4. SHUTDOWN
关闭任务

```python
{
    "task_type": "SHUTDOWN",
    "data": {
        "reason": "system_shutdown"
    }
}
```

## 错误处理

### 1. 进程崩溃恢复
- 主服务器监控舵机进程状态
- 自动重启崩溃的进程
- 记录错误日志

### 2. 任务队列错误
- 队列满时的处理策略
- 任务发送失败的重试机制
- 死锁检测和恢复

### 3. 舵机控制错误
- 串口连接丢失检测
- 舵机响应超时处理
- 硬件故障恢复

## 性能优化

### 1. 任务队列优化
- 使用优先级队列
- 批量处理任务
- 内存使用优化

### 2. 进程通信优化
- 减少序列化开销
- 使用共享内存（可选）
- 异步通信

### 3. 舵机控制优化
- 命令缓存
- 批量执行
- 状态缓存

## 监控和调试

### 1. 日志系统
- 分级日志记录
- 进程标识
- 时间戳记录

### 2. 状态监控
- 进程状态查询
- 队列状态监控
- 性能指标收集

### 3. 调试工具
- 任务队列浏览器
- 进程状态查看器
- 性能分析工具

## 扩展性

### 1. 添加新的任务类型
1. 在 `TaskType` 枚举中添加新类型
2. 在 `TaskProducer` 中添加创建方法
3. 在 `TaskConsumer` 中添加处理逻辑

### 2. 添加新的控制进程
1. 继承 `TaskConsumer` 类
2. 实现特定的处理逻辑
3. 在 `MainServer` 中注册

### 3. 支持分布式部署
1. 使用消息队列（如Redis、RabbitMQ）
2. 实现服务发现
3. 添加负载均衡

## 故障排除

### 1. 常见问题

**Q: 舵机进程启动失败**
A: 检查串口权限和硬件连接

**Q: 任务队列阻塞**
A: 检查消费者进程状态和任务处理逻辑

**Q: WebSocket连接断开**
A: 检查网络连接和服务器状态

### 2. 调试步骤

1. 检查日志文件
2. 验证进程状态
3. 测试任务队列
4. 检查硬件连接

### 3. 性能问题

1. 监控CPU和内存使用
2. 分析任务处理时间
3. 优化队列大小
4. 调整进程数量

## 总结

这个解耦架构提供了以下优势：

1. **高可用性** - 组件独立运行，单个组件故障不影响其他组件
2. **可扩展性** - 易于添加新的控制进程和任务类型
3. **可维护性** - 清晰的模块分离和接口定义
4. **性能优化** - 异步处理和任务队列优化
5. **错误恢复** - 完善的错误处理和恢复机制

通过这种架构，WebSocket服务可以专注于手势识别和通信，而舵机控制可以独立运行，两者通过任务队列进行松耦合的通信。
