# KL 散度原始函数体与证明

四段完整原程序/已有整数交换候选的前向、反向函数体直接保留在 Kernels.lean。
原 cast、运行时循环上界、两个 target 分支、全部 reduction、mask 和 store 都保留。
原 Python/wrapper、条件候选、公开规格未改。新增 candidate_kernels.py 只是已有
Lean 整数地址交换候选的 Python 表达，逐 AST 检查限定两处整数乘法交换。

移除旧转写多加的 epsilon FP32 转换，重证实际前向/反向执行、安全、数学结果
及具体 native FP32 lowering；不假设任意 FP cast 恒等。完整原/候选执行和失败、
安全在任意 algebra 下由整数乘法交换连接。保留原分支、空尺寸、别名与 frame 范围。

85 个题级模块、3162 个累计入口及联合导入，在 warm 和独立空项目构建中通过。
18 项假设审计通过，两项 FP assumptions 均为 none。首次诊断打印达到 4M
heartbeat 限制；只提高临时诊断预算后通过，保留两次记录。114 项针对性 Python
通过；359 项全量检查只有已知旧 Structural 数值凭据错误。13 次官方 comparator
因宿主隔离不可用，CPU checker 缺 Torch，均如实记录。未跑 GPU/新数值实验。

累计 tasks 87/183、examples 19/19，仍有 96 个任务待完成。保持草稿，等待人工审阅。
