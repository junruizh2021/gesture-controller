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
    
    
    def set_angle(self, angle: float, velocity: Optional[float] = None, 
                  interval: Optional[int] = 800, wait: bool = True) -> bool:
        """
        设置舵机角度 - 改进版本
        
        Args:
            angle: 目标角度 (-180° 到 180°)
            velocity: 转速 (度/秒)，如果为None则使用默认值
            interval: 运动时间 (毫秒)，如果为None则使用默认值
            wait: 是否等待运动完成
            
        Returns:
            bool: 是否成功设置角度
        """
        if not self.is_connected or self.control is None:
            logging.error("舵机未连接，无法设置角度")
            return False
        
        try:
            # 角度限制
            angle = max(self.min_angle, min(self.max_angle, angle))
            
            # 使用默认参数
            if velocity is None:
                velocity = self.default_velocity
            if interval is None:
                interval = self.default_interval
            
            # 调试日志：打印参数类型和值
            logging.debug(f"舵机控制参数 - servo_id: {self.servo_id} (类型: {type(self.servo_id)}), "
                         f"angle: {angle} (类型: {type(angle)}), "
                         f"velocity: {velocity} (类型: {type(velocity)}), "
                         f"interval: {interval} (类型: {type(interval)})")
            
            # 使用舵机SDK的异步模式
            self.control.begin_async()
            
            # 设置舵机角度
            self.control.set_servo_angle(
                servo_id=int(self.servo_id),  # 确保servo_id是整数
                angle=float(angle),           # 确保angle是浮点数
                velocity=float(velocity),     # 确保velocity是浮点数
                interval=int(interval),       # 确保interval是整数
                is_mturn=True,
                t_acc=int(interval/2),        # 确保t_acc是整数
                t_dec=int(interval/2)         # 确保t_dec是整数
            )
            
            # 根据wait参数决定是否等待
            if wait:
                # 等待运动完成，使用interval时间估算
                self.control.end_async(0)  # 0表示等待执行完成
                # wait_time = interval / 1000.0  # 转换为秒
                # time.sleep(wait_time + 0.1)  # 额外等待0.1秒确保完成
                
                # 更新当前角度
                self.current_angle = self.control.query_servo_angle(self.servo_id)
                logging.info(f"舵机运动完成，当前角度: {self.current_angle:.1f}°")
            else:
                # 异步执行，立即返回
                self.control.end_async(1)  # 1表示不等待，立即返回
                logging.info(f"舵机角度设置完成，目标角度: {angle:.1f}° (异步执行)")
            
            # 更新目标角度
            self.target_angle = angle
            
            logging.info(f"设置舵机 {self.servo_id} 角度: {angle:.1f}° (转速: {velocity}°/s, 时间: {interval}ms)")
            
            return True
            
        except Exception as e:
            logging.error(f"设置舵机角度失败: {e}")
            return False
    
    def move_relative(self, delta_angle: float, velocity: Optional[float] = None, 
                     interval: Optional[int] = None, wait: bool = True) -> bool:
        """
        相对角度移动
        
        Args:
            delta_angle: 相对角度变化
            velocity: 转速 (度/秒)
            interval: 运动时间 (毫秒)
            wait: 是否等待运动完成
            
        Returns:
            bool: 是否成功移动
        """
        target_angle = self.current_angle + delta_angle
        return self.set_angle(target_angle, velocity, interval, wait)
    
    def get_current_angle(self) -> float:
        """获取当前角度"""
        if not self.is_connected or self.control is None:
            return self.current_angle
        
        try:
            self.current_angle = self.control.query_servo_angle(self.servo_id)
            return self.current_angle
        except Exception as e:
            logging.error(f"获取舵机角度失败: {e}")
            return self.current_angle
    
    def is_moving(self) -> bool:
        """检查舵机是否正在运动"""
        if not self.is_connected or self.control is None:
            return False
        
        try:
            # 通过比较目标角度和当前角度来判断是否在运动
            current_angle = self.control.query_servo_angle(self.servo_id)
            angle_diff = abs(self.target_angle - current_angle)
            return angle_diff > 1.0  # 角度差大于1度认为在运动
        except Exception as e:
            logging.error(f"检查舵机运动状态失败: {e}")
            return False
    
    def set_damping(self, power: int = 0):
        """设置舵机为阻尼模式"""
        if not self.is_connected or self.control is None:
            logging.error("舵机未连接，无法设置阻尼模式")
            return False
        
        try:
            self.control.set_damping(self.servo_id, power)
            logging.info(f"设置舵机 {self.servo_id} 为阻尼模式，功率: {power}mW")
            return True
        except Exception as e:
            logging.error(f"设置舵机阻尼模式失败: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取舵机状态信息"""
        status = {
            'connected': self.is_connected,
            'servo_id': self.servo_id,
            'current_angle': self.current_angle,
            'target_angle': self.target_angle,
            'is_moving': self.is_moving(),
            'port': self.port,
            'baudrate': self.baudrate
        }
        
        if self.is_connected and self.control is not None:
            try:
                # 获取详细状态信息
                status.update({
                    'voltage': self.control.query_voltage(self.servo_id),
                    'current': self.control.query_current(self.servo_id),
                    'power': self.control.query_power(self.servo_id),
                    'temperature': self.control.query_temperature(self.servo_id),
                    'status': self.control.query_status(self.servo_id)
                })
            except Exception as e:
                logging.error(f"获取舵机详细状态失败: {e}")
        
        return status
    
    def close(self):
        """关闭舵机控制器"""
        if self.uart is not None:
            try:
                self.uart.close()
                logging.info("舵机控制器已关闭")
            except Exception as e:
                logging.error(f"关闭舵机控制器失败: {e}")
        
        self.is_connected = False
        self.control = None
        self.uart = None

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
