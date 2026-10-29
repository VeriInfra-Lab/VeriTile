# int4_matmul：可直接运行的 GPU 位值采集包

原 kernel、固定完整两块 packed 输入及三份软件参考均已绑定；根代理独立重跑 **26 项测试、4 个离线编译配置**，全部通过。GPU 状态为 **NOT_RUN**，实际 launch 为 0。此包不改变任务证明分类。

在有 SM80/SM90 GPU 的环境中，以本目录为当前目录运行：

```bash
python -B candidate/src/run.py run \
  --reference candidate/reference-v3 \
  --output /tmp/int4-native-capture \
  --device cuda:0 --compare-profile ascending_fma
```

输出目录必须是新目录。Python 环境需要与已绑定 compiler profile 一致；工具记录实际版本、源码/编译器哈希、GPU UUID/driver、TTIR/TTGIR/LLVM/PTX/cubin、完整五个分配的前后 raw words。固定原 autotune 成员为 64×128×32、SPLIT_K=1、warps=4、stages=4；分别采集 fusion=false/true。

三图的 active half 参考分别为原 source `0x3c00`、正序 carry-FMA `0x0000`、逆序 carry-FMA `0x4000`。工具检查完整 C 分配和输入保持性；只对显式选择的图报告固定用例 MATCH/MISMATCH。整体始终是 CHARACTERIZATION，不能把碰巧命中某个参考写成全任务硬件 PASS。

该用例用于辨别计算图行为，不会验证所有形状、NaN/Inf、FTZ、alias 或 MMA 内部精度。原 source 与 carry 图的普遍等价已有反例，不再作为等待实验“通过”的假设。

[工具详细说明](candidate/README.md)、[根独立执行记录](root-replay/commands.json)、[对应完整软件证明](../int4-source-dotacc-software-closure/README.md)。候选中的早期 reference/offline/v2 目录是保留历史；当前交付使用 src 与 reference-v3。重新生成参考的历史绝对路径不影响这里的自包含运行命令。
