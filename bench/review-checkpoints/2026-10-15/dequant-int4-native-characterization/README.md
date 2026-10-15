# matmul_dequantize_int4：GPU 位值采集与软件参考

根代理独立复验了 **45 项工具测试、8 个离线编译配置**；4 个普通 Lean 模块和 25 条具名公理检查已有独立复编。新生成参考与交付参考的 71 个文件逐字相同。实际 GPU 尝试为 **NOT_RUN，0 次 launch**；本实验包不增加任务完成数或公开规格数。

用例采用原 17 项 autotune 配置之一：64×128×32，M=64、N=128、K=64，完整两轮 K，分别执行 NO_GROUPS=false/true。每次采集全部 8192 个有效输出、两侧 guard 以及全部输入分配的前后原始字，使用非零指针 origin=8。

| NO_GROUPS | 原源码及非融合软件图 | 融合 half FMA 软件图 |
|---|---|---|
| false | 0x5401 | 0x5400 |
| true | 0x5402 | 0x5401 |

此输入区分 half 标量融合；原源码的分块 dot 后加 carry 与非融合 live-carry 图在该输入同值，不能据此区分这两个图，也未证明 native MMA 可按软件顺序串行解释。完整父证明保持原参数、特殊值与地址条件；固定有限实验语料不缩小父合同。

从本目录运行，输出目录必须是新目录：

```sh
python3 -B candidate/verify.py
/opt/conda/bin/python3 -B candidate/src/run.py run \
  --reference candidate/final-validation/reference-final \
  --output /tmp/dequant-int4-native-capture \
  --profile-fusion-off compiled_separate_ascending_fma \
  --profile-fusion-on compiled_fused_ascending_fma
python3 -B candidate/src/verify_capture.py /tmp/dequant-int4-native-capture
```

需要 SM80 或 SM90 GPU 与 Triton 环境。工具记录原 kernel、编译选项、TTIR/PTX/cubin、实际指针、设备和驱动信息。只使用运行前选定的软件图判定固定输入 MATCH/MISMATCH；即使命中，整体状态仍为 CHARACTERIZATION，不是全任务数值 PASS。

[详细用法与边界](candidate/README.zh-CN.md)、[根实际执行结果](root-replay/terminal-results.json)、[独立见证复编](root-proof-replay/receipt.json)、[完整软件合同](../dequant-int4-source-compiled-software-closure/README.md)、[根审查](root-review.json)。candidate 内旧 reference-v1/v2、offline-v1 等为保留历史；当前入口是 final-validation。
