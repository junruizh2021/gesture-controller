#!/bin/bash
set -e
#!/bin/bash
USERNAME='gesture-controller'
export http_proxy=http://proxy.cd.intel.com:911
export https_proxy=http://proxy.cd.intel.com:911
export no_proxy=127.0.0.0/8,localhost,10.0.0.0/8,192.168.0.0/16,192.168.14.0/24,*.intel.com
CURRENT_DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd);

if [ ! -d "${CURRENT_DIR}"/temp ]; then
  mkdir "${CURRENT_DIR}"/temp
  mkdir "${CURRENT_DIR}"/temp/WorkSpace
  mkdir "${CURRENT_DIR}"/temp/WorkSpace/scripts
  mkdir "${CURRENT_DIR}"/temp/AI-models
fi

# copy model files into image
cp -r "${REPO_DIR}"/models/* "${CURRENT_DIR}"/temp/AI-models/

# copy python files into image
cd "${CURRENT_DIR}"/temp/WorkSpace

cp "${REPO_DIR}"/*.py ./

# copy shell files into image and set executable permission
cp "${REPO_DIR}"/docker/*.sh ./scripts/

cd "${CURRENT_DIR}"
docker build -t gesture-controller:1.1.0 \
       --network=host \
       --build-arg USERNAME=${USERNAME} \
       --build-arg HTTP_PROXY=${http_proxy} \
       --build-arg HTTPS_PROXY=${https_proxy} \
       --build-arg NO_PROXY=${no_proxy} .

if [ $? -eq 0 ]; then
  echo "build image success, rm temp"
  rm -rf "${CURRENT_DIR}"/temp
fi