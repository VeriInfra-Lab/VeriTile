# Token attention 原始函数体与证明

原始函数和已有 block-origin 复用候选完整粘贴，保留所有语句、整数 where、signed 0/1 循环、load、mask、sum、scale、store 与末尾 return。类型和值在调用处提供。原始 Python、Correct、FPEquiv、Contract 和 Admission 不变。

前端修复 Python 整数 where 推断及 typed pointer 的隐式 store 转换。实际源公式去掉旧表示凭空添加的输入 cast，显式保留选定有序 FP32 sum 原语、scale 的表示转换及隐式写回转换；不假定任意 cast 恒等。完整执行、frame、安全、公开 Real/FP 与 native/Tensor 全部重证，未缩小输入域。

44 个本题模块、111 个前端入口、累计 4255 个入口与联合导入通过，含独立空项目重建。138 项本题审计及 5 项前端证明使用标准公理基；主 FP 结论无数值假设。203 项针对性 Python 检查通过；448 项全量检查保留旧 Structural 数值凭据错误。48 次官方 comparator 尝试受主机隔离限制；CPU checker 缺少 Torch，均不计为通过。未运行 GPU 或新数值实验。

累计 tasks 103/183、examples 19/19，326 个选中对应项，80 个任务待修。保持草稿，等待人工审阅后再合入。
