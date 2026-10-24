# 浮点值契约修复证据

本目录记录原 41 项仅有数学结论的任务补充浮点值契约的审阅过程。当前结果以[验收总表](../ACCEPTANCE.zh-CN.md)、[最终收据](../acceptance-after-value-repair/receipt.json)和[逐项修复范围](../../../tritonbench_g/fp_value_repairs.md)为准。

- `reviewed_value_registry.json`：41 项的逐声明人工审阅登记。107 条参数化契约与 11 条固定实例分开；固定实例不扩大参数化证明范围。
- `reviewed_input_sha256.json`、`reviewed_legacy_changes.json`、`reviewed_line_mapping.json`：源文件版本、旧数学证明兼容改动和原证据行位置的明确绑定。
- `integrate_value_reviews.py`：核对原 629 条公开命题文本及新增声明的登记，再生成当前审阅。未登记的来源变化会拒绝集成。
- `finalize_reports.py`：核对收据、归档产物和当前输入一致后发布总表；报告会保留实际未通过或未运行的检查。
- `before/`：本轮开始前的报告，作为历史对照，不代表当前完成状态。
- `scalar-reference.json`、`scalar-validation-preflight.json`：836 个软件标量参考案例及运行环境预检。本轮没有 GPU 实验结果。
- `report-tests.log`：报告工具回归检查，日志保留实际计数。
- [构建独立复核](final-build-review/receipt.json)：41 个入口及 239 个支持模块的注册、源码与编译产物哈希。
- [最终数值独立复核](final-numerical-review/receipt.json)：声明、范围、分类和归档依赖一致性。
- [源码审查附件归档](acceptance-source-dependency-closure/receipt.json)：完整证据文件收录、回归测试及搬迁复现结果。
- [清单依赖归档](acceptance-inventory-dependency-closure/receipt.json)：保留清单声明的全部依赖，包括生成的规格索引。

其余子目录保存任务及兼容修复的精确补丁、输入摘要、编译日志和审阅回执。每份回执只证明其记录的输入与范围；最终同一版本的全库结果见最终收据。`attempt1/` 等失败尝试保留作追踪，不作为通过证据。

旧数学模型兼容修复不会获得新的浮点完成标记。23 项 cast/store 模型、3 项已发现全域反例的假设、6 项缺源码，以及硬件原语和编译器对应的实验工作仍独立记录。
