当我们修改代码，重新编译后，需要修改测试集群中的二进制文件observer，并重启测试集群，通常可以这么做。

假设我们修改代码后已经编译完成observer。
bash build.sh debug --make


1. 停止测试集群

./tools/deploy/obd.sh stop -n single
2. 替换二进制文件

cp build_debug/src/observer/observer /tmp/obtest/bin/observer
3. 启动测试集群

./tools/deploy/obd.sh start -n single

注意：

- 以上命令执行假设都在 oceanbase 源码目录；

- 假设部署的集群目录在 /tmp/obtest，如果有需要请自行调整。