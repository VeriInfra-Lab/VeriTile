本包提供 `matmul_dequantize_int4` 的合法大 tile 软件位值见证与 GPU 特征实验工具。4 个 Lean 模块、25 项具名公理检查已通过，并经 root 独立复编；45 项工具测试通过；8 种离线编译成功。当前机器 CUDA 不可用，实际 GPU 尝试是 **NOT_RUN，退出码 2，0 次 launch**。没有 GPU 数值通过结论，也不改变任务分类。

原源码 SHA 为 `163c65f3c3c3c82cd0df16dd558662a8c9161d7b2af5ce9dca672384ffd140ed`。工具只移除原 autotune 装饰器以选定其已有配置，逐 AST 核对函数体；没有更改 kernel。编译用显式运行时参数签名，不利用 JIT 将值为 1 的步长偷偷常量化。

具体配置是原 17 项 autotune 配置中的 `BM=64, BN=128, BK=32, GROUP_M=8, warps=4, stages=4`；`M=64, N=128, K=64`，完整执行两轮 K。两个用例分别为 groupsize=32 / NO_GROUPS=false、groupsize=64 / NO_GROUPS=true。A 为 half 1，B 每个打包字为 `0x33333333`，zero 每个打包字为 `0x22222222`；第一组 scale 为 `0x3c01`，第二组为 `0x3c00`。全部 8192 个输出均活跃，未把小语义 tile 当成实际 GPU 配置。

| NO_GROUPS | 原源码分块 dot 后加 carry | 未融合 half 算术、live-carry 顺序 FMA | 融合 half FMA、live-carry 顺序 FMA |
|---|---|---|---|
| false | 全部 `0x5401` | 全部 `0x5401` | 全部 `0x5400` |
| true | 全部 `0x5402` | 全部 `0x5402` | 全部 `0x5401` |

每个内存分配都具有 8 个元素的非零指针 origin。C 完整分配为 8208 个原始 16 位字，其中两侧各 8 个 guard 初值 `0x7e55`。所有 A/B/S/Z 完整分配的原始字（包括 NO_GROUPS 下未读取的第二组容量）及 C 均逐字采集 before/after，并保存 JSON 和小端二进制；GPU 分支检查实际 data_ptr 差值、类型、分配范围和各分配互不重叠。软件证明的 physicalView 前缀是定义为不变的初态值；CUDA 的字节布局、指针 origin、实际内存不越界仍由实验采集检验，没有把这个定义解释成已证 byte ABI。

这个用例区分半精度标量融合。它**不能区分**原源码 dot-from-zero 后另加 carry 与未融合 live-carry 图，因为两者在本用例同值。软件图使用明确的顺序 FMA 计划，不证明 MMA/WGMMA 可由该顺序实现。`compiler-observations.json` 保存实际 TTIR/PTX 行号和文本：8 个图都有 live-carry `tt.dot`；fusion-on 有 half FMA，且 zero 乘法仍独立；SM80/90 分别落到 MMA/WGMMA。文本观察不证明其位值等价。

运行前必须显式选择 fusion-off 与 fusion-on 各自的比较图。所有支持图的匹配信息会同时显示，但仅预选图决定固定用例的 MATCH/MISMATCH，不能从别的命中图挑一个宣布通过。即使预选图命中，报告仍为 CHARACTERIZATION，`numeric_acceptance=false`、`universal_native_equality_proved=false`。退出码 0 只表示编译/采集完成；1 表示预选图不匹配或编译/采集/绑定错误；2 表示未能运行 GPU。Lean 合同、有限 CPU oracle、GPU 实测结果分别保留。

直接复核（在本包根目录）：

```bash
python -B verify.py
python -B src/verify_capture.py final-validation/offline-final
python -B src/verify_capture.py final-validation/gpu-attempt-final
DEQUANT_TEST_REFERENCE="$PWD/final-validation/reference-final" DEQUANT_TEST_OFFLINE="$PWD/final-validation/offline-final" DEQUANT_TEST_NO_GPU="$PWD/final-validation/gpu-attempt-final" python -B src/test_runner.py
```

在有 SM80 或 SM90 GPU 的机器运行（输出目录必须不存在；不修改冻结结果）：

```bash
python -B src/run.py run \
  --reference final-validation/reference-final \
  --output /tmp/dequant-int4-new-gpu-capture \
  --profile-fusion-off compiled_separate_ascending_fma \
  --profile-fusion-on compiled_fused_ascending_fma
python -B src/verify_capture.py /tmp/dequant-int4-new-gpu-capture
```

离线重编同样使用新目录，可传 `--arch 80 --arch 90`：

```bash
python -B src/run.py compile \
  --reference final-validation/reference-final \
  --output /tmp/dequant-int4-new-offline-capture --arch 80 --arch 90 \
  --profile-fusion-off compiled_separate_ascending_fma \
  --profile-fusion-on compiled_fused_ascending_fma
```

`final-validation/validation-receipt.json` 保存真实子进程命令、退出码、日志及工具输入哈希；其中工具测试包含明确的 CPU runtime test double，模拟记录只存在于临时目录，未作为 GPU 结果封存。`proof/receipt.json`、`proof/last-build.json`、`proof/replay.py` 给出 4 模块拓扑、直接依赖/对象/日志/公理绑定与独立复编入口；root 的独立重放收据位于 `final-validation/reference-final/root-replay-receipt.json`。两份泛化软件证明父包、Q 八配置历史捕获的 receipt pin 均与参考一起携带；其历史外部依赖哈希保留在原证明收据，不伪称搬迁后仍是本机对象。

运行环境记录实际 Triton 包、Python executable、ptxas/libdevice 的前后哈希；有 GPU 时另记录设备 UUID、驱动查询、已载入 CUDA 库。没有声称所有宿主共享库/OS 已 hermetic 绑定。CPU oracle 仅处理这个已证明的有限输入语料，明确拒绝 NaN/Inf 的数值解码；C 的 NaN sentinel 仅原样运输。这不缩小两个父包的完整泛化证明域，也不把本实验推广到任意输入。

主工作区、冻结父包和共享对象均未修改。`reference-v1` 是开发期因未复制父 receipt 而中止的输出；`reference-v2`、`offline-v1`、`gpu-attempt-v1`、`tests-v1/v2.log` 是此前开发验证记录。正式入口只使用 `final-validation`。
