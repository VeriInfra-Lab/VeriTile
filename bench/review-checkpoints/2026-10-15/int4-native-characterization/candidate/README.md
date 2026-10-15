# 原 int4_matmul 的 GPU 位值采集工具

本包是数值特征采集工具，不是新的核正确性证明。最终版本为 `src/`、`reference-v3/`、`offline-v3/`、`native-attempt-v3/`。此前 reference/offline/v2 产物保留为开发历史，不能作为当前工具的验收结果。

实际验证：26 个 CPU/来源绑定/拒错测试通过；4 个原核配置离线编译成功（SM80/SM90 × fusion=false/true）。GPU 运行尝试明确为 **NOT_RUN**：CUDA 不可用，0 次 launch、0 份 GPU 输出。所有输出位向量目前均是软件参考，不能写作硬件结果。

## 采集的程序与配置

原 Python SHA 为 `f8b62e30bed71855e5e40603806f6e3f34cc8548dc4be923ef1168c97f92489b`。逐行提取原 `@triton.jit matmul_kernel`，仅不执行 autotune host wrapper；参数与函数 AST、语句原文均保持。编译前验证 Triton 实际 `fn.src` 正是已 SHA 校验的函数体，源码模块从已检查的 bytes 直接 compile/exec，不使用旧 `.pyc`。

固定原列表中的真实 autotune 成员：SPLIT_K=1，BLOCK_M/N/K=64/128/32，GROUP_M=8，warps=4，stages=4；分别记录 fusion=false/true。原列表共有 36 条，不声称覆盖其它成员。通过 `ASTSource` 显式指定 5 个指针和 14 个 i32 runtime 参数，再直接调用已编译的 `CompiledKernel[(1,1,1)]`；不会因为 Python 参数值等于 1 而静默变成额外 constexpr specialization。

测试采用已被 Lean 完整循环证明覆盖的同一 packed 初态：M=N=1、K=64、两个完整 K blocks、group_size=1。全 64×128 tile 执行，其中一个输出 active，其余输出由 mask 关闭；这只是一个 GPU 特征采集用例，不收窄新的全参数 Lean 合同。

五个 buffer 均有 logical origin=8，实际 kernel 指针必须等于 allocation base+8×elementSize。A、packed weights、scale、packed zeros 与 C 使用不同物理分配；所有 buffer 全量前后快照都保存为 raw word JSON 和 little-endian `.bin`。C 的所有 257 个 half words 都比较，只有物理 offset 8 是 active 写地址。任何其它 C 字改动或任何输入分配改动都单独列出，不能仅比较有效输出。

## 软件参考与判定

参考绑定完整 DotAcc 证明包 `e9aac373ac26355faad834a2263cbe80bb06c62b465f834ed72a68f260b76371`，以及 root 已独立编译的 8 模块/31 公理检查。父包 native receipt 另行 SHA 钉住，并核对其 records 等于父 manifest 已绑定的 running-results，补足该历史 receipt 未在父顶层 map 中的归档边界。

独立 Python oracle 从全部 packed 初态读取、解包、half 乘法、逐次 fp32 RNE 运算，重新得到：

| 软件图 | 物理 C[8] 的 half bits |
|---|---:|
| 原 source 的 dot 从零开始、之后加 carry | `0x3c00`（1） |
| ascending K 的 carry-seeded FMA | `0x0000`（0） |
| descending K 的 carry-seeded FMA | `0x4000`（2） |

这些结果均与完整 Lean 程序见证一致。CPU oracle 限于这组有限输入，不是全 IEEE NaN/Inf/负零模型；NaN/Inf 输入会被拒绝，不能拿 CPU 自检声称硬件全域关系。Lean 的全参数软件证明在另一个冻结证明包中。

GPU 成功采集的总状态始终为 `CHARACTERIZATION`，不以命中 0、1、2 中任一个为整体 PASS。`--compare-profile` 显式选择要检验的一个软件图；该固定用例只标 `MATCH_ON_FIXED_CASE` 或 `MISMATCH_ON_FIXED_CASE`。所有三图的比较结果同时保留，但 `numeric_acceptance=false` 和 `universal_native_equality_proved=false` 始终不变。

本工具不证明 MMA/WGMMA 等于某个软件 FMA 计划，也不证明硬件可串行化为某个 K 排列。未知内部累加顺序/精度、FTZ、NaN 行为、其它 dtype/shape、alias 与跨 program 情形仍未绑定。已否定的“原 separate-dot+add 普遍等于 carry 图”不会被重新列为待实验通过的规律。

## 命令

以下均创建全新输出目录；现有目录会被拒绝。

```bash
/opt/conda/bin/python -B src/test_runner.py
/opt/conda/bin/python -B src/run.py compile \
  --reference reference-v3 --output /tmp/int4-offline-recheck \
  --arch 80 --arch 90
/opt/conda/bin/python -B src/run.py run \
  --reference reference-v3 --output /tmp/int4-native-capture \
  --device cuda:0 --compare-profile ascending_fma
```

`run` 从实际设备 capability 选择 SM80/90；可用 `--arch 90` 要求与该设备精确一致。其它架构需另做明确 profile 的编译/审阅，当前工具拒绝。运行时记录 GPU 名称、UUID、CUDA/Torch/Triton 版本、与 UUID 对应的 driver 查询、实际 compiler metadata、完整 TTIR/TTGIR/LLVM/PTX/cubin、真实指针/launch 参数和 raw word 文件。compiler 全包、实际 ptxas/libdevice、Python executable 在编译前后哈希核对；环境绑定缺 UUID 或 driver 对应时另标 `environment_binding_complete=false`，不把数值匹配当作环境绑定。

退出码 0 仅表示离线编译或数据采集完成；指定软件图不匹配退出 1，CUDA 不可用退出 2。任何状态都不代表全任务 GPU 正确性。

若从完整原证明包重新生成参考：

```bash
/opt/conda/bin/python -B src/run.py prepare \
  --proof-package /tmp/veritile-int4-dotacc-closure-20261015 \
  --output /tmp/int4-fresh-reference
```

`prepare` 校验父 receipt 和每个实际引用的 artifact；运行时又将复制的证据逐项反查父 manifest，不能通过同时修改 case 与本地 hash 绕过 Lean 来源。oracle 也从已验证源码 bytes 加载；测试覆盖 timestamp-valid poisoned `.pyc`、载入后源码漂移、伪造 case/hash 和错误来源声明。

`verify.py` 对本包只读复核。当前工具、原 41 registry、工作区证明、任务分类和既有冻结验收包均未改动。
