本 revision 修复独立审查发现的两处验收协议问题，旧 v1 与其证据保持不变：

- compile 的实际 resolved 架构必须逐项等于预选 requested 架构；run 显式请求目标时也必须匹配实际设备目标。删掉预选 SM80 再修改 expected_records 不再能通过。
- 环境绑定标志由已实际执行的 native 记录、设备 UUID、成功的精确 driver 查询及同 UUID/非空 driver/name 行重新计算。离线或 NOT_RUN 不可标 true；数值状态仍独立保持未闭合。

`protocol-fix.diff` 与 `changed-tools.json` 保存最小代码差异和前后 SHA。11 项新测试覆盖删架构、架构顺序漂移、离线/NOT_RUN 假环境声明、正确及错误 UUID、失败/错误查询、空 driver、未执行状态，以及完整 CPU 模拟运行的环境绑定。模拟记录仍不作为 GPU 证据。

`rbe_triton_transform` 的真实原核调查和完整缓冲区采集工具已完成：**16 种离线编译成功、45 项工具测试通过；当前 GPU 尝试为 NOT_RUN，退出码 2，0 次 launch。** 本包没有新增软件浮点位语义定理，没有改变原公开声明、输入域或任务分类。

调查直接提取原 Python 的两个 `@triton.jit` 函数 `get_freq_multi_tokens` 和 `rbe_triton`，逐 AST 核对完整函数体。pow、sin、cos、两次 barrier、两个掩码、两次输出 store 都保留。未改为供应 cos/sin 的核；未把 host-side 随机数据初始化当作源码计算。主工作区和冻结父包未修改。

16 个编译配置为 SM80/SM90 × 四组 block 参数 × fusion off/on。四组分别是原 wrapper 默认 `BM=2,BK=1024`，原 backend 非空见证的 `1,2`，接受参数域内的 `2,64` 和 `4,128`。THETA=10000，实际 helper 的 DIM=128，编译器默认解析为 4 warps、3 stages。这里只覆盖这些具体合法配置；旧形式化参数允许任意 DIM、theta、Nat 尺寸/步长/程序号，不能由 16 次编译概括其全部范围。

实际图的关键发现：

- **pow 没有被完整常量折叠。** 包括默认 BK1024 在内，TTIR 均保留 `__nv_powf`，PTX 仍有 libdevice pow 路径、`rcp.approx.ftz.f32`、`ex2.approx.ftz.f32` 和 FMA。不能以“只有 64 个频率”直接删去此原语。
- **cos/sin 不是此次图里的单条 `cos.approx` / `sin.approx`。** 当前 16 份 PTX 这两条指令计数均为零；实际是 `__nv_cosf` / `__nv_sinf` 的范围规约和多项式代码。
- fusion-on 会把末端 pair 算术合成 FMA；fusion-off 的 pair 保留分开的乘法与加减，但 libdevice 内部的显式 FMA 仍存在。原源码逐步舍入图不能自动当成这份编译图的等价规格。
- 自然数域与真实 i32 图还有独立差异：LLIR 的 token 加法为 `add i32`，随后 `sitofp`。`start=2147483647,row=1` 在旧 Nat 模型求和为 2147483648，而此 i32 求和解码为 -2147483648。未为旧定理添加 int32 界限；这不是等待有限实验“验证为普遍成立”的关系。

`compiler-observations.json` 给出每个真实工件的 SHA、pow/trig TTIR 位置、PTX 原语计数、源行 37/38 对应的 pair 算术及 LLIR 转换线索。调试行号和文本观察不是正式的编译等价证明。`numeric-obligations.json` 拆出 7 项精确缺口：整数图、pow、除法、sin/cos、pair 融合、half 转换/双 mask、完整 grid/内存 ABI。`software-reuse.json` 记录检查过的既有数值库来源：基础有限位算术可复用，但缺失本次 libdevice/SFU/FTZ 图的完整位模型，因此不新增任意回调的“闭合”包装。

采集语料有 10 个完整 launch：原 wrapper 的四个 pos、奇数 K 和 M 尾部、非单位列/行/批步长、2^24 附近位置、同一分配上的原地输入/输出、i32 token 回绕边界及旧单 pair 见证。原 wrapper 的四个用例各捕获全部 8192 个输出，其他用例也覆盖各自完整 grid。特殊输入包含 ±0、subnormal、±Inf、NaN 位型；无 finite/noNaN 前提。虚部地址严格为 `base + 1`，不是 `base + stride_n`。旧 per-program theorem 的任意参数与 alias 域不变；实验语料是具体实例，未被登记成通用软件定理。

每个分配都有 8 个元素的非零 pointer origin。运行时核对实际 data_ptr、dtype、所有分配大小及 alias 映射，完整保存所有输入和输出分配的 before/after 原始 uint16 JSON 和小端二进制，包括 guard、stride holes、masked 尾部。对真实 active writes 独立枚举完整足迹，报告每个 active 输出位型和所有非写入位置的 frame 变化。原地写回按共享分配建模，允许恰当输出位置变化，不伪称输入始终不变。

**数值比较保持 NOT_ESTABLISHED。** 当前尚无完整独立 native 位参考；即使全部 frame 不变，甚至输出看起来合理，工具也只写 `CHARACTERIZATION_NUMERICS_UNBOUND`、`numeric_acceptance=false`。不会拿 libm、Real.sin/cos、供应的角度表或任意“实际输出相等”参数充当证明。完整原核输出采集可用于后续逐图定位和验证，但有限实验不能证明普遍数值等式。

复核冻结结果：

```bash
python -B verify.py
python -B src/verify_capture.py final-validation/offline
python -B src/verify_capture.py final-validation/gpu-attempt
RBE_TEST_REFERENCE="$PWD/final-validation/reference" RBE_TEST_OFFLINE="$PWD/final-validation/offline" RBE_TEST_GPU="$PWD/final-validation/gpu-attempt" python -B src/test_runner.py
```

在 SM80 或 SM90 GPU 上采集全部 10 用例 × 双 fusion；输出目录必须不存在：

```bash
python -B src/run.py run --reference final-validation/reference --output /tmp/rbe-new-native-capture
python -B src/verify_capture.py /tmp/rbe-new-native-capture
```

离线重编原 helper 和完整原核：

```bash
python -B src/run.py compile --reference final-validation/reference --output /tmp/rbe-new-offline-capture --arch 80 --arch 90
```

退出码 0 只表示编译或采集完成，1 表示编译/launch/来源绑定/frame 错误，2 表示未运行 GPU；任何退出码都不能证明数值正确。当前无 GPU 的正式记录是 `final-validation/gpu-attempt`。`final-validation/receipt.json` 绑定真实进程命令、退出码、日志、工具前后哈希。45 项测试中运行分支通过 CPU test double 演练了完整 20 次 launch、所有完整分配/alias 检查及错误分支；临时模拟数据不作为 GPU 收据封存。

工具记录实际 Triton 包、Python executable、ptxas/libdevice 前后 SHA；有 GPU 时还记录设备 UUID、driver 查询和已载入 CUDA 库。没有声称全部 OS/宿主库 hermetic 绑定。复用了冻结 dequant 工具的纯采集/哈希辅助函数，逐函数代码及父 receipt SHA 在 `inputs/source-bindings.json` 中，RBE 不继承 dequant 的数值规格或证明。所有 RBE 旧 Lean/README/Python 文件的原字节 SHA 也在该绑定中。旧版本的全部开发历史和原34测试仍保留在不可修改的父包，路径和receipt见 `parent-bindings.json`；本revision的正式入口只使用新 `final-validation`。
