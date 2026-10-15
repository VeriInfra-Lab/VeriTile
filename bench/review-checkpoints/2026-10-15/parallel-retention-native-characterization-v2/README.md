# parallel_retention_attention 原核实验交接 v2

根代理独立重跑 **42 项工具测试、24 个离线编译配置**，参考文件逐字相同。实际 GPU 尝试为 **NOT_RUN、0 次 launch**。本包不新增公开定理或完成任务数。

原接受证明覆盖完整 forward O 和独立 backward DK/DV helper。DQ、backward shell、host NK/NV 求和及跨 program 组合尚未证明；shell 的采集另标为额外观察。工具保存完整 Q/K/V/dO 和 O/DQ/DK/DV partial 分配的原始前后位、全部 helper 调用、origin=8、guard 和非写入位置，不做 host 求和。

实际图中的贡献 dot 即使关闭融合也使用旧累加器；本批 fp32 PTX 使用标量 FMA，没有 MMA/WGMMA。exp2 使用带 FTZ 的近似指令，log2 是内联 libdevice 图。未指定 padding 的 load 仍保留 undef，不能擅自补零。这些差异已列成 [7 项具体义务](candidate/numeric-obligations.json)，数值比较保持 NOT_ESTABLISHED。

初版验收器的两个缺口已修复：请求的架构必须完整保留；GPU 环境完整标记须由实际执行及匹配的 UUID/driver 查询重新计算。旧漏检复现、修订 diff 和拒错测试均保留于 [候选包](candidate/README.zh-CN.md)；见[根审](root-review.json)。

从本目录在 SM80/SM90 GPU 上运行，输出路径须不存在：

```sh
python3 -B candidate/verify.py
/opt/conda/bin/python3 -B candidate/src/run.py run --reference candidate/reference-v1 --output /tmp/retention-native-capture
python3 -B candidate/src/verify_capture.py /tmp/retention-native-capture
```

默认采集五个完整用例、三个入口及两种融合配置；helper 的每次调用另行计数。成功采集只是实验记录，不能据此宣布数值证明或硬件普遍正确。
