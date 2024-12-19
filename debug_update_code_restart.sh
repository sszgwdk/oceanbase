#!/bin/bash
# 重新编译 + 切换二进制文件
# 切换二进制文件
./tools/deploy/obd.sh stop -n obcluster
bash build.sh debug --make -j16
# cp build_release/src/observer/observer /data/obcluster/bin/observer
cp build_debug/src/observer/observer /data/obcluster/bin/observer
echo "success cp observer to /data/obcluster/bin/observer"
sleep 3

./tools/deploy/obd.sh start -n obcluster

echo "success restart observer"
echo "start obclient"
sleep 3

## 在 ann_benchmark 测试之前，需要重建测试租户（没改索引结构可以不用重建）
# ./deps/3rd/u01/obclient/bin/obclient -h 127.0.0.1 -P2881 -uroot
# obclient> drop tenant perf force;
# obclient> create tenant perf replica_num = 1,primary_zone='zone1', resource_pool_list=('pool_2') set ob_tcp_invited_nodes='%';

# ## 测试时导入数据并构建索引
# cd ../ann-benchmarks
# python run.py --algorithm oceanbase --local --force --dataset sift-128-euclidean --runs 1
# ## 测试时跳过导入数据及构建索引
# python run.py --algorithm oceanbase --local --force --dataset sift-128-euclidean --runs 1 --skip_fit
# # 计算召回率及 QPS
# python plot.py --dataset sift-128-euclidean --recompute
# # 重启 oceanbase 集群
# ./tools/deploy/obd.sh cluster restart obcluster

# # 运行混合标量查询场景SQL，hybrid_ann.py 位于 ann_benchmarks/algorithms/oceanbase/
# python ann_benchmarks/algorithms/oceanbase/hybrid_ann.py
