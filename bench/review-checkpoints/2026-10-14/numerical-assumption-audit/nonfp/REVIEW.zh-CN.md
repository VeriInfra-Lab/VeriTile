非 admitted_fp 公开契约审阅已完成。数据来自最新验收包的实际 629 条声明，选取 mathematical / abstract_rounding 的全部 474 条，覆盖 178 个任务；逐条检查声明输入前提，另扫描 1,266 个本任务目录内的 Prop/Rules/Assumptions 定义候选，追踪数值结构与相关 IO 包装定义。未改根目录源码，未重跑大型 Lean 构建。

- 429 条 mathematical 声明：未发现新增的数值代数律前提。保持数学模型结论，不自动变为硬件浮点正确性结论。
- 45 条 abstract_rounding 声明：4 条有显式数值律，30 条仅带共享 RoundingModel 结构条件，11 条数值 assumptions 确实为空。
- 4 条显式数值律分别为 3 条 `R.round .fp16 = id`，以及 BGMV 的 1 条实际 native add 交换律。三个全域 fp16 恒等假设在通常二进制16舍入下有直接反例：`2049/2048` 舍入成 `1`。这是宿主 binary16 参考反例，不是 GPU 测试结果；它反驳硬件满足全域前提，并不反驳条件 Lean 定理。
- BGMV 的 72 配置经现有 olean 的轻量 `#eval` 实际抽取出 6 个 dtype/precision profile：显式 fp64/real 12 个、显式 fp32/real 12 个、默认 fp16 28 个、默认 bf16 4 个、默认 typed-fp32 8 个、默认 real 8 个。默认 real 与显式 fp32/real 可由既有 fp32 Rules 条件性给出；其余四个 profile 仍需要精确 native 绑定。没有把 widened-fp32 的 bf16 实验冒充 native bf16 验证。
- RoundingModel 的 `round_idem` 与数学 `.real` 恒等字段是 33 条公开声明共享的结构要求，其中包括上述 3 个全域 fp16 恒等声明。幂等性并不是 33 个独立实验待办。
- 尺寸、布局、地址安全、实际操作数有限、预计算阶段输入等条件按用户要求独立保留，不计作缺失数值律。`hNext` 等 supplied carry 仍是阶段契约，不能据此声称端到端证明。
- `s.undef` 在 50 个公开声明文本出现；另有 IO 包装定义内条件。零 oracle 是输入/语义模型选择，不是代数律；不能据此直接声称任意 undefined padding 都支持。这里没有声称对全部包装器的零 oracle 条件作穷尽计数。

`audit.json` 是合并入口：每条声明有源文件/行号/哈希、精确声明哈希、显式律 ID、共享模型 ID、正常输入前提、阶段前提、oracle 条件和证据。`experiment-requests.json` 单列 BGMV 六种 profile 的问题；`profiles.json` 给完整 72 配置映射。`inventory.json` 保留全部实际声明，`hypotheses.txt` 保留逐声明输入前提，`predicates.json` 保留定义候选。`validation.json` 复核 474 条唯一声明及其当前源码哈希。

本审阅是源码契约审阅，不等同于独立 elaborated-environment comparator replay。数学算子（包括 exp/log/sqrt/rsqrt/pow）、有限宽量化和硬件实现的对应关系单列在 `native_semantics_boundaries`，不把它们伪装成每个数学定理新缺的实验。
