#!/bin/bash

chmod 777 /dev/ttyUSB0

# init action of servo
python /home/${USERNAME}/WorkSpace/action_opt.py

nohup python /home/${USERNAME}/WorkSpace/main_server.py --dynamic_gestures --enable_servo > ./handtracker-sever.log 2>&1 &

# tail 命令在后台运行
tail -f handtracker-sever.log & 
TAIL_PID=$!

# wait 命令会阻塞在这里，等待 tail 进程结束
wait $TAIL_PID