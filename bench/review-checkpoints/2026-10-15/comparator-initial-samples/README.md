# 正式 comparator：首批三个检查通过

三个原始官方命令、guest driver 和 host launcher 均实际退出成功。根代理另行核对了原始结果、目标映射、两份独立 judge 的源码前后校验和 kernel 接受日志。

| 检查 | 实际目标 | 属于 872 条任务声明的数量 |
|---|---:|---:|
| canonical library | 118 | 独立集合 |
| VectorAdd 样例 | 9 | 0 |
| FP4 unscaled FPCorrect | 1 | 1 |

完整 365 文件／872 条任务声明尚未通过。后续新增声明不在冻结 F3 的这次检查内；本包没有 GPU 正确性结论。library 已在[独立归档](../comparator-canonical-library/README.md)中记录，这里重复引用同一次运行，不能加成两次覆盖。

查看[覆盖明细](coverage.json)、[根代理检查](root-checks.json)和[完整原始证据](evidence/README.md)。
