#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Not enough arguments passed to script"
    echo "usage : sh create_container_on_ivi.sh <docker image name:tag> <container name>"
    exit 1
fi

http_proxy=''
https_proxy=''
no_proxy='127.0.0.0/8,localhost,10.0.0.0/8,192.168.0.0/16,192.168.14.0/24,*.intel.com'

container_docker_image_tag=$1
container_name=$2
container_hostname=${container_name}

docker_device="/dev/dri/renderD128"
runder_devices_list=($(ls /sys/class/drm/ | grep render))
for device in "${runder_devices_list[@]}"; do
  # TODO: can only select intel gpu from all gpus, but can not diff iGPU and dGPU
  if [ "$(cat /sys/class/drm/${device}/device/uevent | grep i915)_tail" != "_tail" ]; then
    docker_device="/dev/dri/${device}"
  fi
done
echo "docker_device = ${docker_device}"

mem_total=$(expr $(cat /proc/meminfo | grep MemTotal | awk '{print $2}') / 1024 / 1024)
memory_size=${mem_total}g
echo "memory_size = ${memory_size}"
# --entrypoint /bin/bash
# -v /home/anna/WorkSpace/:/home/anna/WorkSpace/ \
docker create -ti  --network=host \
    -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy \
    --memory=$memory_size \
    --name ${container_name} --hostname ${container_hostname} \
    --add-host=${container_hostname}:127.0.0.1 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --device-cgroup-rule='a *:* rmw' \
    -v /sys:/sys:rw \
    --device ${docker_device} \
    --device /dev/accel \
    --device /dev/ttyUSB0 \
    --device /dev/snd --device /dev/tty0 --device /dev/tty1 --device /dev/tty2 --device /dev/tty3 \
    --cap-add=NET_ADMIN --cap-add=SYS_ADMIN ${container_docker_image_tag}

docker start ${container_name}
