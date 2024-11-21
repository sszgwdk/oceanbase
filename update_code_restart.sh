#!/bin/bash

# 重新编译 + 切换二进制文件
bash build.sh release --init --make -j16

# 切换二进制文件
./tools/deploy/obd.sh stop -n obcluster
cp build_release/src/observer/observer /data/obcluster/bin/observer
echo "success cp observer to /data/obcluster/bin/observer"
sleep 3

./tools/deploy/obd.sh start -n obcluster

echo "success restart observer"
echo "start obclient"
sleep 3
./deps/3rd/u01/obclient/bin/obclient -h 127.0.0.1 -P2881 -uroot