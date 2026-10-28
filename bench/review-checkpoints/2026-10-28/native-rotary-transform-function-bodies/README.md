# Rotary transform 原始函数体与证明

原始 Python 函数体完整粘贴，保留提前返回、两种布局、packed/fixed metadata、scalar/tensor offset、共轭、全部 cast/mask/store。已有 rk 复用候选仅改 rk_repeat；新增 Python 文件明确记录这份既有候选。两套程序各六个调用配置，共十二个映射。类型和值在调用处提供，未使用的两个参数不影响函数体。

前端支持条件重绑定和显式调用 Boolean specialization，签名元数据即使位于被裁掉的分支也会拒绝；嵌套 signed load 保留 signed 通道。源码没有历史的额外 seqlen 初始化，不宣称与旧 AST 相等。完整执行和 trace safety 连接到独立数学规格、完整 grid 和原/候选等价，未缩小输入域。

42 个本题模块、114 个前端入口、累计4300个入口与联合导入通过，含独立空项目重建。169 项本题审计和39项前端证明（其中7项新增）使用标准公理基，主FP结论无数值假设。212项针对性Python检查通过；457项全量检查保留旧Structural数值凭据错误。50次官方comparator尝试受主机隔离限制，CPU checker缺少Torch，均不计为通过。未运行GPU或新数值实验。

累计tasks104/183、examples19/19，338个选中对应项，79个任务待修。保持草稿，等待人工审阅后再合入。
