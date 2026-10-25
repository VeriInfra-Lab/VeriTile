# 两道 reversed cumsum 原始函数体与完整证明

标量、向量的原版和已有候选 Python 均未修改，四份原始函数体独立逐句
核对。所有 cast、tuple、mask、反向循环、转换及 store/carry 顺序保留。
元信息放在体外，原有六种输入乘六种输出 dtype、配置及公开契约不变。

实际 Int ID、feature/address 和循环执行重新证明；向量版的旧数学
不变量通过只用于证明的观察视图连接，源码没有添加 ghost 语句。
独立 Real 后缀公式、有序 FP 执行、完整状态/失败等价、公开 IO、实际
访问安全、全 grid 输出和 frame、既有 Tensor/host/native 范围保留。
没有新增浮点恒等式、输入限制或硬件符合性结论。

完整联合 warm/导入重放覆盖 2278 个根；82 个受影响题级根及完整依赖
在项目对象初始为空的独立 checkout 重建，共享前端没有修改。精确
根、jobs、源码闭包和记录见 review.json。78 项针对性 Python 通过；
全量 323 项仅旧 Structural 凭据哈希错误。官方比较器隔离不可用，
CPU 契约缺 torch，实际失败留存，不计通过；没有 GPU 或新数值实验。

累计 65/183 tasks、19/19 examples；214 对中 209 raw、5 旧 AST。
尚余 118 题继续处理；不合入，等待用户审阅。
