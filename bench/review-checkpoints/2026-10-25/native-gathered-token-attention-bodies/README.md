# 两题 gathered token attention 原始函数体与完整证明

token_attn_reduceV 和 token_attn_mistral 的选定 grid-y checked 入口及
其底层共享实现一并迁移。原版和已有 affine 候选 Python JIT 全部未改；
四份19/20条语句函数体独立逐句核对。类型/调用/constexpr 元信息体外；
全部 gather、独立 mask、padding、reduce、输出转换和 return 保留。

两题完整 original/affine 可执行语句表与原有 FP32 程序的 rfl 恒等已证。
全部99根涵盖选定主目录和底层 execution/host/native 助手。实际全状态/
失败及trace安全等价、独立Real公式、公开IO、全grid、物理frame、Tensor
和native范围重放，既有FP32输入/INT32元数据范围、数值端口不变。去掉
仅供内部且始终传FP32的泛型dtype参数，没有限制公开定理或输入域。

2458根warm和联合导入通过；99根完整依赖在项目产物初始为空的独立
checkout重建。无共享前端修改；精确jobs、Git源码闭包见review.json。
82项针对性Python通过；全量327项仅旧Structural凭据哈希错误。实际
比较器缺主机隔离，原版/候选CPU探针缺Torch，失败留存不计通过；探针
side copy的完整原版JIT AST与实际选定源码一致另行核对。无GPU/新数值实验。

累计69/183tasks、19/19examples；222对中217raw、5旧AST。其余114题继续；
草稿不合入，等待用户审阅。
