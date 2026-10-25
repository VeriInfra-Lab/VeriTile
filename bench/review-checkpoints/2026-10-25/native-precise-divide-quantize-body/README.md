# 精确除法与 quantize_global 原始函数体

原 Python JIT/wrapper 未改；13条语句、全部 INT64 cast、ABI注解、mask、
safe offset、FP32计算、精确除法、llrint和INT8隐式转换逐句核对。元信息体外。

Op.divRN/独立可选回调保留原语身份；不回退普通除法/倒数。缺少实现时
空tile也失败；FP64/未知family须显式转换。输入局部精度、实际type/erase、
flatten/locality/访存安全均有连接。源程序执行、严格Nat/Int地址与整数
modular INT8存储连接重证，独立Real规格、完整配置调度FP等价及所有既有
Tensor/native/host/frame范围保留，无新数值恒等式/条件/实验。

96前端+10旧控制通过；加30题级根的136根/8814jobs通过。2491根warm/
联合导入及完整依赖在项目产物初始为空的独立checkout重建；精确jobs、
Git源码闭包和审计见review.json。83项针对性Python通过；全量328项仅旧
Structural数值凭据错误，前一已发表快照已同样失败，本轮文件又有变更，
不虚假重钉。官方比较器隔离不可用、CPU缺Torch，实际失败留存不计通过。
此前磁盘耗尽的中断日志保留；清理旧临时项目产物后，独立构建从空项目产物重新开始。无GPU或新数值实验。

累计70/183tasks、19/19examples；223对中218raw、5旧AST，其余113题继续。
草稿不合入，等待用户审阅。
