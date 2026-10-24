# 完整 block-pointer reduction 函数体

源码提交、逐题语义核对和完整验证记录见 review.json。FP16/FP32 两种类型的
原始与 affine 候选均逐句粘贴，运行 stride 和唯一 constexpr 块尺寸分开。
block-pointer 的类型通过重绑定/advance 保留，FP16 max 提升与窄存储保留。
旧表示多出的同格式 FP32 cast 被删除；实际任意 primitive 后端的完整图、
执行、安全、全部输出和 frame 重新证明，未添加 cast 恒等前提。
完整算法的 Real 投影连接独立数学 companion；所有维度/stride、M=0/N>0、
原有 IO、数学公式及可选 strict bit/native/Tensor/host 结论均保留并重放。

787 roots / 9,621 jobs 本地与初始空项目的独立干净构建通过，联合导入通过。
Git 源码闭包、全部模块及原始日志见本目录。只复用固定外部 packages。
36 项针对性 Python 测试通过；全量 281 项保留已复现的旧 numerical receipt
错误。官方 comparator 缺 setup；额外 CPU contract 缺 torch，未计为通过。
没有 GPU 实验或新数值假设。旧的数值/来源证据保留历史边界，不虚假 repin。

任务源码当前为 24/183，examples 为 19/19：105 对中 100 对是原始函数体，
5 对旧 AST。156 个未覆盖任务、1 个旧 AST 和 2 个间接入口仍待处理；不合入。
