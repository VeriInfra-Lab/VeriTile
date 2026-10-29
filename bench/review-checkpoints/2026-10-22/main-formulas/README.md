# 183 题主入口公式、前提与范围检查

后续抽查更正：`mean_reduction` 的旧入口仍暴露扁平内存与调度细节，先前可读性
验收过宽。现已通过逻辑矩阵 IO 和新旧完整判断的双向连接整理，见
[定点修正](../mean-logical-io/README.md)。下述 183 题统计是此前检查的范围；
本次只刷新这一题，不据此宣称所有题都已按新增的逻辑坐标要求复核。

本轮逐题检查唯一清单选定的 183 个 task，包含此前单独修复的
`reversed_cumsum` 和 `softmax_triton2`。168 题调整数学展示、域记号或说明，
15 题保留当前入口；未发现新的实质数学或程序证明缺口。这里统计的是主入口
检查结果，不是新增了 168 道正确性证明。

每题记录见 [audit.json](audit.json)：独立数学目标、必要条件及其来源、实际
程序／grid／阶段范围、到既有规格的证明连接、两个 FP 程序及实际数值依赖、
最终源码哈希与针对性检查证据。主入口共 367 个 reader 文件、449 个主声明；
真实的多阶段、存储分支、全局／维度结果均保留。

主线完成统计保持 **182 已完成、1 带数值延期完成**。既有延期仍只有
`matmul_leakyrelu` 的相应规则实例，两项实验门仍为 `NOT_RUN`；本轮没有新增、
重新运行或宣布通过任何数值实验。已有准入规则和严格模型定理分别保留，
不把空实验规则集解释为任意硬件或任意 opaque 运算上的无条件等式。

## 固定的入口规则

- `Correct.lean` 显示独立 Real 数学公式；简单公式直接写入主声明，复杂多输出
  公式可用同文件相邻的纯数学定义。连接必须有严格证明，不能只给导入的
  `reference` 换名字或作无意义 eta 展开。
- 逻辑尺寸与实现 tile 尺寸分开。完整 `n` 列 softmax 对 `n` 求和，带 mask 的
  `B` 列窗口通过公开的覆盖条件连接；旧窗口定理保留在 `Proofs/`。
- 输入域、cell/reset、整数观察、分离、mask、调度和后端条件必须保留；
  `io ⊨{P} f` 正好表示既有 `ImplementsOn io P f`，不会删除 `P`。
- `FPEquiv.lean` 比较实际 `original ≡[R] candidate`，保留源码精度、转换、
  顺序、规则适用域及原有比较范围。单程序、完整 grid、独立阶段与已组合管线
  分别说明，不互相冒充。
- reader 只保留主声明、纯数学定义和薄程序／IO 实例。短连接可为主证明的局部
  步骤；复用助手在 `Proofs/`，可选 native／host／Tensor 应用在 `Misc/`。
- Python 检查、契约和定理条件有对应。已有 launch 推出的覆盖条件直接证明；
  不增加无必要的输入拒绝条件。`uniform_sampling` 主证明不需要的
  `N ≤ 2^34` 前提已移除，实际 wrapper 的分块及模型宽度边界仍准确保留。

上述规则已写入 [AGENTS.md](../../../../AGENTS.md)、
[主定理约定](../../../MAIN_THEOREM_CONVENTIONS.md) 和
[正确性接口说明](../../../../documents/CorrectnessSurfaces.md)。

## 检查范围与保留限制

各题修改后的 Real 主入口均有针对性 Lean 检查。180 题当前主文件／逐题联合
检查通过；下列 3 题准确记录额外的重放限制，不据此撤回已有适用成果，也不把
这些限制列成 numerical assumption：

| 题目 | 可用证据 | 本次未宣称完成的部分 |
| --- | --- | --- |
| `chunk_gla_fwd` | 当前 Real 主入口；FP 五个完整主声明及公理检查；既有完整证书 | 附加打印与整个 FP 模块末尾的长审计未重放 |
| `int_scaled_matmul` | 当前 Real 主入口；两个当前 FP 主声明的精确历史成功证书、对象及依赖哈希；后续完整源码保持链 | 本轮统一缓存缺 CandidateIO／含旧 Algebra ABI，未完成该代 FP 新重放 |
| `kv_cache_filling` | 清单选定归档版本的完整 Real 文件；精确保留完整 FP 主声明和证明的夹具及公理检查 | 完整 FP 文件的深 `print_fp_assumptions` 在旧 Safety／Algebra ABI 处失败；未称完整文件重放成功 |

`int_scaled_matmul` 的确切证书复核见
[独立证据审计](../../../../.veritile-cache/headline-formulas-20261022/queue12/int-scaled-certificate-audit.json)。
旧的可导入 FP 缓存并不包含当前两个声明名；它没有被当作当前声明的成功证据。

多个额外跨题总导入遇到旧证明对象重复导出生成方程声明的兼容性问题，保留了
失败日志和逐题成功证据。没有宣布 183 题同进程总导入、全依赖独立源码重放、
GPU 实测或新的硬件一致性验证通过。复用的内部证明、选定归档版本、具体严格
软件模型与实验规则域都在逐题记录中说明。

另有独立审查者检查最终各批语义、入口及证据，7 题接受更深入的独立抽核；
这不等于审查者重新从源码编译了全部内部证明。原角色／IO 审查的旧判定保留，
本轮只刷新变化的 reader 及新增纯数学定义；旧受影响记录保存在
[previous-review-rows.json](previous-review-rows.json)。

## 后续防护

```sh
python bench/check_main_formula_review.py
python bench/check_main_entry_roles.py
python bench/check_io_organization.py
python bench/task_progress.py --check
```

最终结果见 [validation.json](validation.json)。这些脚本检查选定版本、主声明、
角色、源码连接和证据是否与人工逐题记录一致；它们不会自动推断数学正确性，
也不能代替 Lean 证明或数值实验。
