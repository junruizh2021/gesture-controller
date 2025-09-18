# 动态手势识别功能使用指南

## 功能概述

新增的动态手势识别功能可以检测连续的手势变化，识别特定的手势序列模式。

## 支持动态手势模式

### 1. CLOSE_GESTURE - 闭合手势
- **模式**: FIVE → FIST
- **描述**: 从张开到握拳
- **触发条件**: 检测到从"五"手势变为"拳头"手势

### 2. OPEN_GESTURE - 张开手势  
- **模式**: FIST → FIVE
- **描述**: 从握拳到张开
- **触发条件**: 检测到从"拳头"手势变为"五"手势

### 3. PEACE_WAVE - 和平手势挥手
- **模式**: PEACE → FIVE → PEACE
- **描述**: 和平手势挥手
- **触发条件**: 检测到"和平"→"五"→"和平"的手势序列

### 4. THUMBS_UP_DOWN - 拇指上下
- **模式**: OK → FIST → OK
- **描述**: 拇指上下
- **触发条件**: 检测到"OK"→"拳头"→"OK"的手势序列

### 5. FINGER_COUNT_UP - 逐指张开
- **模式**: FIST → ONE → TWO → FIVE
- **描述**: 逐指张开
- **触发条件**: 检测到从"拳头"逐指张开到"五"的手势序列

## 使用方法

### 1. 启用动态手势识别
```bash
python ws-handtracker-server.py --dynamic_gestures
```

### 2. 自定义参数
```bash
python ws-handtracker-server.py --dynamic_gestures --gesture_window_size 10 --gesture_confidence 0.7
```

### 3. 参数说明
- `--dynamic_gestures`: 启用动态手势识别
- `--gesture_window_size`: 滑动窗口大小（默认8帧）
- `--gesture_confidence`: 最小置信度阈值（默认0.6）

### 4. 组合使用
```bash
# 启用动态手势 + 显示所有功能
python ws-handtracker-server.py --dynamic_gestures --show_gesture_display --show_landmarks --show_scores
```

## 输出示例

当检测到动态手势时，控制台会输出：
```
[2024-01-15 14:30:25] 动态手势检测: CLOSE_GESTURE - 从张开到握拳
[2024-01-15 14:30:28] 动态手势检测: OPEN_GESTURE - 从握拳到张开
[2024-01-15 14:30:32] 动态手势检测: PEACE_WAVE - 和平手势挥手
```

## 技术原理

1. **滑动窗口**: 使用固定大小的窗口存储最近的手势历史
2. **模式匹配**: 检查手势序列是否匹配预定义的模式
3. **容错机制**: 允许80%的匹配度，提高识别鲁棒性
4. **置信度过滤**: 只处理置信度足够高的手势
5. **防重复触发**: 检测到模式后清空历史记录

## 自定义模式

可以通过修改 `DynamicGestureProcessor` 类中的 `dynamic_patterns` 字典来添加新的手势模式：

```python
self.dynamic_patterns = {
    "CUSTOM_GESTURE": {
        "pattern": ["GESTURE1", "GESTURE2", "GESTURE3"],
        "description": "自定义手势描述"
    }
}
```

## 注意事项

1. **性能影响**: 动态手势处理会增加一定的计算开销
2. **延迟**: 需要等待足够的手势历史才能检测模式
3. **准确性**: 窗口大小和置信度阈值会影响检测准确性
4. **环境要求**: 需要稳定的手势识别环境，避免误触发
