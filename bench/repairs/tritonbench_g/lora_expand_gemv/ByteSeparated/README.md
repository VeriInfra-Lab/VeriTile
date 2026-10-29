# LoRA expand GEMV：完整 Python 函数体

`Kernels.lean` 直接粘贴 `../lora_expand_gemv_byte_separated.py` 中 `_bgmv_expand_kernel` 的完整函数体。`CandidateKernels.lean` 对应既有 `../lora_expand_gemv_unit_split.py` 中 `_bgmv_expand_unit_split_kernel`，唯一算法改写仍是 `tl.cdiv(N, SPLIT_N)` → `1`。候选 Python 及其宿主检查、launch 参数均保留原文件。

两者均保留 docstring、`lora_index == -1` 时的提前返回、`EVEN_K` 掩码分支、向权重 dtype 的转换、`ADD_INPUTS` 分支、逐行块循环和原始 store。没有 `tl.region`、Lean antiquote 或补写的输出 cast 混入函数体。

调用信息在函数体外：输入和输出为 FP32，权重为 FP16；索引存储允许 signed int32 或 int64，分别有完整源码入口。运行整数采用现有宿主的 int32 ABI，原生宽度与范围约束仍由既有 caller 契约承担。constexpr 和 stride 等值通过参数记录绑定。两种索引 ABI 的逻辑程序相等有单独证明；该等式不代替原生宽度检查。

`Correct.lean` 保留完整 dense GEMV 的独立实数公式、sentinel 行保持和可选的旧 Y 加法。`IO.lean` 保留原有完整内存映射、输入域、调度与输出观察。`FPEquiv.lean` 仍比较实际原程序和既有 unit-split 候选：宿主传 `SPLIT_N=N`，实际 launch 的 `N>0`，空 grid 的处理不变；没有增加输入条件或数值改写规则。

旧证明表示中的否定哨兵测试、零字面量及多余同格式输出转换，由 `Proofs/SourceConnection.lean` 接回完整粘贴源码。连接同时证明整个最终状态与访存安全，对所有原参数分支成立；同格式转换依据既有具体图模型的定义，不假设任意浮点代数的 cast 恒等律。`SourceCompatibility.lean` 只保留旧通用接口；权威源码 `Kernels.lean` 不导入证明层。

`jit-correspondence.json` 检查原程序和候选的四份完整函数体与各自调用类型。`bench/tests/test_lora_expand_body.py` 防止删除 return/cast、改变 mask/循环、遗漏加回输入或抹去索引宽度；既有 `../check_unit_split_source.py` 核对完整 Python 候选仅有预定改写。构建、公理检查和环境限制以对应审阅凭据为准；源码对应检查本身不构成数值或 GPU 一致性证明。
