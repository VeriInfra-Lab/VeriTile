# logsumexp_fwd：完整归约图、内部差异与实验交接

根代理独立复编 **3 个普通 Lean 模块、15 条具名公理检查**，复核 48 个原核编译配置，并通过 26 项工具测试、8 项额外 CPU 运行分支测试和 6 个新的离线编译。实际 GPU 为 **NOT_RUN，0 次 launch**。任务继续归入后端依赖，不增加完成数或公开规格数。

完整旧 fp32 row-block 范围保留：两 HAS_SCALE 分支、任意 Real scale、D/正 B、双 program id、初态、尾部与全 mask 块及别名。新的归约图明确规定逐步浮点加减乘、maxnum 和两个有效归约树；exp、log 和任意 Real 标量转换仍未闭合。Python wrapper 最后的 torch.logsumexp 和输出转换仍在旧范围之外。

两个内部差异已经证明：原行前缀可达 Bloom 的指数碰撞输入；另一个有限用例中，独立乘减得到 +0，而编译图的融合 FMA 得到 0xa8800000（−2^-46）。这都不等于最终 LSE 输出反例。实际 logf 展开含 fma.rn.ftz.f32 和特殊值分支，不能用理想 Real.log 替代。

[完整实验合同](candidate/experiment-contract.json)逐项列出 exp、标量转换、融合前缀、归约树、log 和整核输出的待绑定关系。已经证伪的普遍等式不再当成可通过实验确认的假设。[详细范围](candidate/README.md)、[根独立审查](root-review.json)、[48 个数据流复核](root-dataflow-review.json)。

从本目录运行，输出目录须不存在：

```sh
python3 -B candidate/verify.py
/opt/conda/bin/python3 -B candidate/capture.py run --output /tmp/lse-native-off --device cuda:0 --num-warps 4 --fusion 0
/opt/conda/bin/python3 -B candidate/capture.py run --output /tmp/lse-native-on --device cuda:0 --num-warps 4 --fusion 1
```

工具采集 10 个完整原核用例及一个独立 exp 诊断，保存全部输入输出 raw words、guard 和 frame。当前没有完整数值 oracle，因此状态只能是 CHARACTERIZATION；physical ABI、GPU 指令与真实归约顺序也未验收。
