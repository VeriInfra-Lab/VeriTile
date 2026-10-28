# KV cache filling 原始函数体与证明

原始 helper 和 policy 0/8/4 完整函数体保留全部语句，共7个对应项。既有候选仅交换3处向量 key 地址乘法。前端处理 helper 局部作用域、参数求值、tuple/underscore 赋值和隐式 uint8 运算；源码不增删 cast。公开规格、Python 和输入域不变。

补回此前缺失的物理内存源码与导入，明确初始字节、stride 和调度 ABI；实际执行产生的 store 接到 Tensor 读回及完整 frame。保留修复前失败证据。55个本题入口、117个前端入口、累计4420个入口及联合导入通过，含独立空项目重建。255个任务定理和7个前端证明公理审计通过，公开FP比较无数值假设。

230项针对性Python检查通过；全量475项保留旧Structural数值凭据错误。63次comparator尝试受隔离限制，CPU checker缺少Torch，均未计通过；未运行GPU或新数值实验。

累计tasks106/183、examples19/19、357个对应项，77个任务待修。保持草稿，不合入。
