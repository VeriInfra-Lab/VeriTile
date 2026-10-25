# Dequantization、grouped transpose 与 KL divergence 函数体

三题原始 Python 保留，函数体逐句粘贴，调用 dtype/运行参数/constexpr 在外部。
候选 Python 仅为既有 Lean 候选新写的对照，明确标注来源。
反量化的整个原始函数族与规范原始/候选调用由 rfl 核对。
转置完整执行等价保留真实 INT8 窄化；KL 的全部语句与真实读写端口
由四个完整函数定理核对。完整 Real/FP、安全、输出/frame、所有既有尺寸/
strides/分支、native/host/Tensor 范围均重放，无新数值规则或全程序前提。

1,188 roots / 10,121 jobs 本地与独立初始空项目构建通过，联合导入通过。
源码提交、完整 Git/干净副本闭包、逐题审查及原始日志见 review.json。
45 项 Python 针对性通过；全量 290 项保留既有未改动 Structural.lean
旧数值凭据错误，未虚假 repin。CPU 检查缺 torch；未跑 GPU 实验。
官方固定版本 comparator/exporter/landrun 已借助临时目录内 SHA256 校验的
Go 成功编译。三题均实际尝试，主机 Landlock ABI v4 不满足 v5、
systemd 用户总线不可用；保持隔离检查失败关闭，未记为 comparator 通过。

任务 35/183，examples 19/19；141 对中 136 对为原始函数体、5 对旧 AST。
145 个未覆盖任务、1 个旧 AST 和 2 个间接入口仍待处理。保留草稿不合入。
