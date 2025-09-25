#!/usr/bin/env python3
"""
舵机控制模块
用于控制云台舵机的转动
"""
import time
import serial
import logging
import asyncio
from typing import Optional, Dict, Any

try:
    import fashionstar_uart_sdk as uservo
except ImportError:
    print("警告: 未找到fashionstar_uart_sdk，请安装: pip install fashionstar-uart-sdk")
    uservo = None

class ServoController:
    """舵机控制器类"""
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200, servo_id: int = 0):
        """
        初始化舵机控制器
        
        Args:
            port: 串口设备路径
            baudrate: 波特率
            servo_id: 舵机ID
        """
        self.port = port
        self.baudrate = baudrate
        self.servo_id = int(servo_id)  # 确保servo_id是整数类型
        self.uart = None
        self.control = None
        self.is_connected = False
        
        # 舵机配置参数
        self.current_angle = 0.0
        self.target_angle = 0.0
        self.max_angle = 180.0
        self.min_angle = -180.0
        
        # 运动参数
        self.default_velocity = 100.0  # 默认转速 (度/秒)
        self.default_interval = 1000   # 默认运动时间 (毫秒)
        
        # 初始化连接
        self._initialize_connection()
    
    def _initialize_connection(self):
        """初始化串口连接"""
        try:
            if uservo is None:
                logging.error("fashionstar_uart_sdk 未安装，无法初始化舵机控制器")
                return False
            
            # 创建串口连接
            self.uart = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=1,
                bytesize=8,
                timeout=0
            )
            
            # 创建舵机管理器
            self.control = uservo.UartServoManager(self.uart)
            
            # 测试连接
            if self._test_connection():
                self.is_connected = True
                logging.info(f"舵机控制器初始化成功 - 端口: {self.port}, 舵机ID: {self.servo_id}")
                return True
            else:
                logging.error("舵机连接测试失败")
                return False
                
        except Exception as e:
            logging.error(f"舵机控制器初始化失败: {e}")
            return False
    
    def _test_connection(self) -> bool:
        """测试舵机连接"""
        try:
            if self.control is None:
                return False
            
            # 尝试ping舵机
            is_online = self.control.ping(self.servo_id)
            if is_online:
                # 获取当前角度
                self.current_angle = self.control.query_servo_angle(self.servo_id)
                self.target_angle = self.current_angle
                logging.info(f"舵机 {self.servo_id} 连接成功，当前角度: {self.current_angle:.1f}°")
                return True
            else:
                logging.warning(f"舵机 {self.servo_id} 不在线")
                return False
                
        except Exception as e:
            logging.error(f"舵机连接测试异常: {e}")
            return False

# 全局舵机控制器实例
servo_controller = None

def get_servo_controller() -> Optional[ServoController]:
    """获取全局舵机控制器实例"""
    return servo_controller

if __name__ == "__main__":
    # 测试舵机控制器
    logging.basicConfig(level=logging.INFO)
    
    # 初始化舵机控制器
    controller = initialize_servo_controller()
    
    if controller.is_connected:
        print("舵机控制器测试")
        print(f"当前角度: {controller.get_current_angle():.1f}°")
        
        # 测试角度设置
        controller.set_angle(30.0, velocity=100.0, interval=2000)
        time.sleep(3)
        
        controller.set_angle(-30.0, velocity=100.0, interval=2000)
        time.sleep(3)
        
        controller.set_angle(0.0, velocity=100.0, interval=2000)
        
        # 显示状态
        status = controller.get_status()
        print("舵机状态:", status)
        
        controller.close()
    else:
        print("舵机控制器初始化失败")
