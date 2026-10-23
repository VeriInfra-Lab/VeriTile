# Reference 模块职责整理

本轮检查当前选定版本中的 25 题 Reference 辅助模块。23 题需要职责拆分；`context_attn_nopad` 和 `matmul_leakyrelu_fp8` 的相应模块已具有单一主证明职责，保留现状。其余 158 题复用上一轮检查记录，本轮没有重新声称全库语义验收。

这是一轮源码组织整理，不新增数学证明完成数。整题状态维持 182 `verified`、1 `verified_with_deferred_numerics`（`matmul_leakyrelu`）；没有新增数值规则、实验或 GPU 回放。

| 组 | 题数 | 职责处理 |
| --- | ---: | --- |
| ref04 | 4 | CE 有序公式移入 Proofs；gate 可选 native 移入 Misc；delta 接口适配归 IO |
| ref05 | 4 | norm、log-softmax 和量化转置的接口适配归 IO；可选 native 专属内容归 Misc |
| ref06 | 1 | GEGLU 通道适配归 IO，Real 连接归 Proofs，可选 native 和物理应用归 Misc |
| ref07 | 4 | attention 和 llama 整理可选 native 链；另两题保持 |
| ref09 | 5 | attention、FFN 和 RWKV 的接口、主证明公式及可选 native 分开 |
| ref11 | 7 | 必要有序公式归 Proofs，IO 适配归 IO；复用已有纯数学模块和薄导入兼容文件 |

分拆依据声明的实际用途。主 FPEquiv 所需的 native 有序公式、执行及 frame 证书继续留在主证明链；不能仅因名称含 `native` 就移入 Misc。移动保留原 namespace、公开名称和声明体；kernel、独立数学公式、公开前提、范围、精度、计算顺序、实际 FP 程序对和规则依赖保持。

本轮采用逐题针对性构建和必要联合导入，复用未改依赖的既有成功对象。旧缓存 ABI 或自动生成方程冲突的失败和处理过程保留在 owner 报告中。ref04 的额外跨四题联合导入冲突已在修改前后复现；ref05 的额外跨四题联合导入失败未判定来源。它们没有替代已通过的逐题必要检查。Llama FF 原有 `SourceSafety` fresh 构建限制及只读对象复用继续单列。

`tasks/` 记录本轮实际修改题的声明保持、构建、来源哈希及限制；未改题在三份完整清单中保留旧记录。旧报告、旧清单和旧缓存保持不变。`validation.json` 记录最终 Python 防护检查；这些检查不自动证明数学正确性，不代表完整依赖重编、独立回放或 GPU 实测。

全 183 题导入路径对比另记录 `kv_cache_filling` 的既有缺失源码引用：`SharedTarget` 和 `SharedReadback`。整理前后缺失清单完全相同，未发现本轮新增缺失或已退役模块回退；这不构成本轮对该题完整源码闭包的新回放验收。

独立复核：25 题全部接受，其中 23 题整理、2 题保持。完整汇总见 `.veritile-cache/reference-separation-20261023/evidence/independent/result.json`；发布所用 23 题精确记录见同目录 `release-review.json`。逐题保存完整声明保持、生产源码与构建来源核对证据；独立复核未增加 Lean、GPU 或数值实验回放任务。

首次发布检查另发现 `destindex_copy/Correct.lean` 在本轮开始前已有一个 ASCII 空格漂移，把项目复合记号 `⊨{` 写成了 `⊨ {`，导致 Lean 解析失败。已保存漂移版本，并恢复整文件到上一轮通过检查的精确 SHA；没有改公式、前提、导入或证明。本次恢复单独记录于 `.veritile-cache/reference-separation-20261023/unrelated-drift/restoration.json`，不计作新的证明完成。失败的发布尝试保留，总表在失败时未被写入。

总表中的 `audit`／`report` 容器指针统一指向新一代完整清单；160 道未改题的逐题证据、源码哈希、状态和限制保持原样。该容器指针收尾记录见 `.veritile-cache/reference-separation-20261023/container-pointer-finalization.json`。

最终四项命令行检查全部通过：主公式、主入口职责、IO 组织、总表与渲染一致性。命令、退出码及日志哈希见 `command-validation.json`。
