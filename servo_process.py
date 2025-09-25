#!/usr/bin/env python3
"""
舵机控制进程
独立运行舵机控制逻辑，通过任务队列接收指令
"""

import multiprocessing as mp
import logging
import time
import signal
import sys
import os
from typing import Optional, Dict, Any
import serial
import fashionstar_uart_sdk as uservo

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from task_queue import TaskQueue, TaskConsumer, get_task_queue
from servo_controller import get_servo_controller

# 舵机ID常量
SERVO_ID0 = 0
SERVO_ID1 = 1

class ServoProcess:
    """舵机控制进程类"""
    
    def __init__(self, process_id: str = "servo_process"):
        self.process_id = process_id
        self.task_queue = get_task_queue()
        self.consumer = TaskConsumer(self.task_queue, process_id)
        self.servo_controller = None
        self.running = False
        self.logger = None
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """设置日志"""
        log_format = f'%(asctime)s - {self.process_id} - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'servo_process_{self.process_id}.log')
            ]
        )
        self.logger = logging.getLogger(self.process_id)
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info(f"收到信号 {signum}，开始关闭进程...")
        self.stop()
        sys.exit(0)
    
    def _initialize_servo_controller(self) -> bool:
        """初始化舵机控制器"""
        try:
            uart = serial.Serial(port='/dev/ttyUSB0', baudrate=115200,parity=serial.PARITY_NONE, stopbits=1,bytesize=8,timeout=0)
            self.servo_controller = uservo.UartServoManager(uart)
            if (self.servo_controller.ping(SERVO_ID0) == False):
                self.logger.error("舵机0未连接")
                return False
            
            self.logger.info("舵机控制器初始化成功 - 舵机0和舵机1")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化舵机控制器失败: {e}")
            return False
    
    def _servo_control_callback(self, gesture_name: str, description: str) -> bool:
        """
        舵机控制回调函数（同步执行）
        
        Args:
            gesture_name: 手势名称
            description: 手势描述
            
        Returns:
            bool: 是否执行成功
        """
        try:
            if self.servo_controller is None:
                self.logger.error("舵机控制器未初始化")
                return False
            
            self.logger.info(f"执行舵机控制: {gesture_name} - {description}")
            
            # 根据手势类型执行不同的控制序列
            if gesture_name == "CLOSE_GESTURE" and "从张开到握拳" in description:
                return self._execute_close_gesture_sequence()
            elif gesture_name == "ONE_FINGER_GESTURE" and "比1手势" in description:
                return self._execute_one_finger_gesture_sequence()
            elif gesture_name == "TWO_FINGER_GESTURE" and "比2手势" in description:
                return self._execute_one_finger_gesture_sequence()
            elif gesture_name == "LIKE_GESTURE" and "点赞手势" in description:
                return self._execute_like_gesture_sequence()
            elif gesture_name == "WAVE_GESTURE" and "挥手动作" in description:
                return self._execute_wave_gesture_sequence()
            else:
                self.logger.info(f"未定义的手势控制: {gesture_name} - {description}")
                return True  # 对于未定义的手势，返回成功但不执行动作
            
        except Exception as e:
            self.logger.error(f"舵机控制回调异常: {e}")
            return False
    
    def _execute_close_gesture_sequence(self) -> bool:
        """执行(张开🖐️->握拳✊)手势序列（使用舵机0 - 水平方向）"""
        try:
            self.logger.info("执行张开到握拳手势序列 - 舵机0（水平方向）")
            target_angle = 45.0
            
            self.servo_controller.set_servo_angle( servo_id = 0, angle = target_angle, interval = 500, t_acc=250, t_dec=250,is_mturn=True)
            time.sleep(0.5)
            self.servo_controller.set_servo_angle( servo_id = 0, angle = -target_angle, interval = 1000, t_acc=500, t_dec=500,is_mturn=True)
            time.sleep(1)
            self.servo_controller.set_servo_angle( servo_id = 0, angle = 0, interval = 500, t_acc=250, t_dec=250,is_mturn=True)
            time.sleep(0.5)
            self.logger.info("张开到握拳手势序列执行成功")
            return True
            
        except Exception as e:
            self.logger.error(f"张开到握拳手势序列执行异常: {e}")
            return False
    
    def _execute_one_finger_gesture_sequence(self) -> bool:
        """执行比1手势(握拳✊->比1☝️)序列（使用舵机1 - 垂直方向）"""
        try:
            self.logger.info("执行比1手势序列 - 舵机1（垂直方向）")
            target_angle = 30.0
            # 快速点头动作
            self.servo_controller.set_servo_angle( servo_id = 1, angle = target_angle, interval = 500, t_acc=250, t_dec=250,is_mturn=True)
            time.sleep(0.5)
            self.servo_controller.set_servo_angle( servo_id = 1, angle = 0, interval = 500, t_acc=250, t_dec=250,is_mturn=True)
            time.sleep(0.5)
            
            self.logger.info("比1手势序列执行成功")
            return True
            
        except Exception as e:
            self.logger.error(f"比1手势序列执行异常: {e}")
            return False
    def _execute_like_gesture_sequence(self) -> bool:
        """执行点赞手势序列（使用舵机1 - 垂直方向）"""
        try:
            self.logger.info("执行点赞手势序列 - 舵机1（垂直方向）")
            target_angle = 30.0
            # 快速点头动作
            self.servo_controller.set_servo_angle( servo_id = 1, angle = target_angle, interval = 500, t_acc=250, t_dec=250,is_mturn=True)
            time.sleep(0.5)
            self.servo_controller.set_servo_angle( servo_id = 1, angle = 0, interval = 500, t_acc=250, t_dec=250,is_mturn=True)
            time.sleep(0.5)
            
            self.logger.info("点赞手势序列执行成功")
            return True
            
        except Exception as e:
            self.logger.error(f"点赞手势序列执行异常: {e}")
            return False
    
    def _execute_wave_gesture_sequence(self) -> bool:
        """执行挥手手势序列（使用舵机0 - 云台）"""
        try:
            self.logger.info("执行挥手手势序列 - 舵机0（云台）")
            
            # 左右摆动动作
            for _ in range(3):
                if not self.servo_controller_0.set_angle(20.0, wait=True):
                    return False
                time.sleep(0.1)
                
                if not self.servo_controller_0.set_angle(-20.0, wait=True):
                    return False
                time.sleep(0.1)
            
            # 回到中心
            if not self.servo_controller_0.set_angle(0.0, wait=True):
                return False
            
            self.logger.info("挥手手势序列执行成功")
            return True
            
        except Exception as e:
            self.logger.error(f"挥手手势序列执行异常: {e}")
            return False
    
    def start(self):
        """启动舵机控制进程"""
        self._setup_logging()
        self.logger.info(f"启动舵机控制进程: {self.process_id}")
        
        # 初始化舵机控制器
        if not self._initialize_servo_controller():
            self.logger.error("舵机控制器初始化失败，进程退出")
            return False
        
        # 设置舵机控制回调
        self.consumer.set_servo_control_callback(self._servo_control_callback)
        
        # 启动任务消费者
        self.consumer.start()
        self.running = True
        
        self.logger.info("舵机控制进程已启动，等待任务...")
        
        try:
            # 主循环
            while self.running:
                time.sleep(0.1)
                
                # 检查舵机连接状态
                if (self.servo_controller and not self.servo_controller.ping(0)) or \
                   (self.servo_controller and not self.servo_controller.ping(1)):
                    self.logger.warning("舵机连接丢失，尝试重新连接...")
                    if not self._initialize_servo_controller():
                        self.logger.error("重新连接舵机失败")
                        time.sleep(5)  # 等待5秒后重试
                
        except KeyboardInterrupt:
            self.logger.info("收到键盘中断信号")
        except Exception as e:
            self.logger.error(f"舵机控制进程异常: {e}")
        finally:
            self.stop()
        
        return True
    
    def stop(self):
        """停止舵机控制进程"""
        if not self.running:
            return
        
        self.logger.info("正在停止舵机控制进程...")
        self.running = False
        
        # 停止任务消费者
        self.consumer.stop()
        
        # 断开舵机连接
        self._disconnect_servo_controller(self.servo_controller, "舵机0")
        
        self.logger.info("舵机控制进程已停止")
    
    def _disconnect_servo_controller(self, servo_controller, controller_name):
        """断开舵机控制器连接"""
        if servo_controller:
            try:
                # 检查是否有disconnect方法
                if hasattr(servo_controller, 'disconnect'):
                    servo_controller.disconnect()
                    self.logger.info(f"{controller_name}连接已断开")
                else:
                    # 如果没有disconnect方法，尝试关闭串口连接
                    if hasattr(servo_controller, 'uart') and servo_controller.uart:
                        servo_controller.uart.close()
                        self.logger.info(f"{controller_name}串口连接已关闭")
                    else:
                        self.logger.info(f"{controller_name}控制器已停止")
            except Exception as e:
                self.logger.error(f"断开{controller_name}连接时发生异常: {e}")

def run_servo_process(process_id: str = "servo_process"):
    """
    运行舵机控制进程的入口函数
    
    Args:
        process_id: 进程ID
    """
    servo_process = ServoProcess(process_id)
    return servo_process.start()

def create_servo_process(process_id: str = "servo_process") -> mp.Process:
    """
    创建舵机控制进程
    
    Args:
        process_id: 进程ID
        
    Returns:
        mp.Process: 舵机控制进程
    """
    process = mp.Process(
        target=run_servo_process,
        args=(process_id,),
        name=f"ServoProcess-{process_id}"
    )
    return process

class ServoProcessManager:
    """舵机进程管理器"""
    
    def __init__(self):
        self.processes: Dict[str, mp.Process] = {}
        self.logger = logging.getLogger(__name__)
    
    def start_servo_process(self, process_id: str = "servo_process") -> bool:
        """
        启动舵机控制进程
        
        Args:
            process_id: 进程ID
            
        Returns:
            bool: 是否启动成功
        """
        # 检查进程是否已存在且仍在运行
        if process_id in self.processes:
            existing_process = self.processes[process_id]
            if existing_process.is_alive():
                self.logger.warning(f"舵机进程 {process_id} 已在运行")
                return False
            else:
                # 进程已死亡，清理记录
                self.logger.info(f"清理已死亡的进程记录: {process_id}")
                del self.processes[process_id]
        
        try:
            process = create_servo_process(process_id)
            process.start()
            self.processes[process_id] = process
            
            # 等待进程启动
            time.sleep(1)
            
            if process.is_alive():
                self.logger.info(f"舵机进程 {process_id} 启动成功 (PID: {process.pid})")
                return True
            else:
                self.logger.error(f"舵机进程 {process_id} 启动失败")
                return False
                
        except Exception as e:
            self.logger.error(f"启动舵机进程失败: {e}")
            return False
    
    def stop_servo_process(self, process_id: str) -> bool:
        """
        停止舵机控制进程
        
        Args:
            process_id: 进程ID
            
        Returns:
            bool: 是否停止成功
        """
        if process_id not in self.processes:
            self.logger.warning(f"舵机进程 {process_id} 不存在")
            return False
        
        try:
            process = self.processes[process_id]
            
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)  # 等待最多10秒
                
                if process.is_alive():
                    self.logger.warning(f"舵机进程 {process_id} 未正常退出，强制终止")
                    process.kill()
                    process.join()
            
            del self.processes[process_id]
            self.logger.info(f"舵机进程 {process_id} 已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"停止舵机进程失败: {e}")
            return False
    
    def stop_all_processes(self):
        """停止所有舵机进程"""
        for process_id in list(self.processes.keys()):
            self.stop_servo_process(process_id)
    
    def get_process_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有进程状态并清理死亡进程"""
        status = {}
        dead_processes = []
        
        for process_id, process in self.processes.items():
            is_alive = process.is_alive()
            status[process_id] = {
                'pid': process.pid if is_alive else None,
                'is_alive': is_alive,
                'exitcode': process.exitcode
            }
            
            # 记录已死亡的进程
            if not is_alive:
                dead_processes.append(process_id)
        
        # 清理已死亡的进程记录
        for process_id in dead_processes:
            self.logger.info(f"清理已死亡的进程记录: {process_id}")
            del self.processes[process_id]
        
        return status

# 全局进程管理器实例
_global_process_manager = None

def get_process_manager() -> ServoProcessManager:
    """获取全局进程管理器实例"""
    global _global_process_manager
    if _global_process_manager is None:
        _global_process_manager = ServoProcessManager()
    return _global_process_manager

if __name__ == "__main__":
    # 直接运行舵机控制进程
    logging.basicConfig(level=logging.INFO)
    
    print("启动舵机控制进程...")
    servo_process = ServoProcess("standalone_servo")
    
    try:
        servo_process.start()
    except KeyboardInterrupt:
        print("收到中断信号，正在退出...")
    finally:
        servo_process.stop()
        print("舵机控制进程已退出")
