# 原始循环函数体与规格来源示例

本批代码提交：`478603ecc01546f8d6eb46f85191f2a6d466ae60`。
保持 draft，不合入；其余 tasks/examples 仍继续迁移。

新增迁移两道 `rmsnorm_fused`、`rmsnorm_fused_llama`，保留实际 Python 的两次循环、
显式转换、运行整数参数、epsilon、mask 及存储精度。Llama 的 `eps_arg` 改回原始
`eps`，并同步调用绑定和完整证明。RMSNorm 完整程序保持性与两种公开证明均已重放。

依照用户确认，从独立数学规格新建 8 组原生 Python：LogExp、SoftmaxReciprocal、
FloatDTypeSoftmax、RowWiseSum、RowWiseMax、FloatDTypeAdd、SoftmaxStable、
StableLogSumExp。它们明确标记为 `spec_derived_native_source`，固定数学规格版本，
不冒充恢复的历史原版。每段 Lean quotation 与选定 Python 函数体逐句对应。
AdamUpdateGridLaunch 已找到真正的 benchmark 原版并登记来源，其迁移仍待完成。

关键修正：

- FloatDTypeAdd 的旧 `.round_to` 不是原生 Triton，改为 `.to(tl.float32)`，
  重新证明实数投影及实际程序对的严格加法交换律，没有假定 round/cast 通道相同。
- RowWiseMax 新源明确采用 FP32。候选把归约结果内联到 store，同时保留独立的有
  类型 load；没有宣称保持旧的无类型 load 内联程序。完整执行、输出、frame、
  安全、非空行最大值与新程序对的 FP 等价均已重证。
- SoftmaxStable、StableLogSumExp 的数值域绑定实际 FP32 读取后的值，保留原有
  有限性、非零条件、归约树、内建函数选择和条件 fallback；无新增数值法则。
  无条件 LSE shift 的反例也随实际源程序重证。
- LogExp、SoftmaxReciprocal、FloatDTypeSoftmax 有完整新旧 kernel 相等证明。
  公共前缀移到 Proofs 并连接完整源程序。FP64 `tl.exp` 沿用原内建函数；已核对
  Triton v3.5.1 的官方 language/math.py 接口，不替换为 libdevice。
- 前端保留原生循环与循环状态的 dtype，拒绝被覆写参数的错误特化。FP32 字面
  调用绑定要求每次使用均已有明确 FP32 计算上下文；其他情况继续用运行寄存器。
- 浮点数乘布尔 mask 保留乘法及 0/1 转换，避免错误改成选择 val/0。标量和向量
  的完整程序回归已证明；原 Lion/Adam 实数分支证明随之重证，数学目标不变。
  尚不能表达精度事件的嵌套 load 会明确报错，不登记为完成。

验证范围与限制：

- 初始联合构建 486 个显式模块、9,261 jobs 全部通过；包含前批回归及本批两个
  任务、八组 examples 目录的全部 140 个 Lean 文件（该目录重放为 3,710 jobs）。
- 独立检出起初没有项目 build 或 `.veritile-cache`，只复用固定版本外部 packages；
  同样 486 个模块、9,261 jobs 从项目源码重新构建通过。486 个模块联合导入通过。
- 共享前端的 KV cache 选定 KernelOnly 入口及旧入口另作本地/独立检出回归，
  具体范围与结果见 `review.json`、`kv-*.log.gz`。
- 60 个对应关系检查通过，其中 55 个是原始函数体模式，5 个仍为 argmax 旧 AST。
  32 项针对性 Python 回归通过。全量 Python 测试运行 277 项，仍有 1 项旧数值
  receipt 对未改动 Structural.lean 的哈希错误；保留之前干净基线的复现日志。
  不把全量 Python suite 记为通过，也不直接重钉旧数值证据的哈希。
- 官方 comparator 实际尝试仍因未配置失败，不记通过。没有运行新的 GPU/native
  实验，没有新增数值假设或 axioms。软件模型证明不充当 GPU 编译或硬件符合性实测。

当前 183 个 tasks 中 17 个完成函数体对应，163 个仍待完整对应，1 个为旧 AST，
2 个间接源码入口待接线；19 组 examples 中，1 个对照历史原版、2 个是既有的新配套、
8 个是本批明确标注的规格来源程序，另 8 个尚待迁移。`--require-complete` 应失败。

`source-closure.json` 固定列出的全部仓库内依赖源码，并核对它们均受 Git 跟踪且
与独立检出逐字相同。`review.json` 记录逐题语义检查、具体命令范围和上述限制。
先前记录归档于 `previous-selected-reviews.json.gz`；历史数值验收记录没有被改写。
