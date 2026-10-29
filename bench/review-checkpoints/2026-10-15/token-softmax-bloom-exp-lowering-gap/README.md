# token_softmax_bloom：已证伪的指数绑定与实验交接

原任务仍依赖后端，**不增加软件完成数或公开规格数**。本包已独立复编 3 个普通 Lean 模块，21 条具名公理检查通过；22 项工具测试、36 个原核 PTX 配置的独立复核、2 个诊断程序离线编译通过。GPU 实际尝试为 **NOT_RUN，0 次执行**。

两个合法浮点输入经编译图中的 log2e 乘法变成同一个 word，而直接正确舍入 exp 应落在两个不同输出 word 的区间。因此不能把该编译路径普遍等同于直接 RN(exp)。原 fp32 正常数和默认 fp16 存储均有完整 32 lane 行初态可达证明。结论只涉及指数内部阶段，未证明最终 softmax 输出不同。

[原始证明与说明](candidate/README.md)、[根语义审查](root-semantic-review.json)、[独立编译](root-replay/last-build.json)、[实验契约](candidate/experiment-contract.json)、[原任务范围](candidate/original-scope.json)。三个诊断模块保留在本包，没有进入主任务注册或替代原全范围合同。

从本目录执行，输出目录须为新目录：

```sh
python3 -B candidate/verify.py
/opt/conda/bin/python3 -B candidate/capture.py compile --output /tmp/bloom-review-offline --arch 80 --arch 90
/opt/conda/bin/python3 -B candidate/capture.py run --output /tmp/bloom-review-capture --device cuda:0
```

采集器保存原核整块输入/输出、非零指针、guard、编译配置和设备信息；诊断程序单独记录缩放与 ex2 word。它没有完整 softmax 数值 oracle，因此采集完成最多记为 CHARACTERIZATION，不能记为任务数值通过。旧验收范围未添加有限数、无 NaN、无 alias 或固定宽度限制；固定输入只用于缺口诊断。

还需确定并绑定设备 ex2.approx 与 div.full 位行为、真实归约树、存储转换及 NaN/Inf/±0/subnormal 行为。已证伪的普遍等式不能靠实验通过来补作假设。软件可达树、PTX 文本检查、实际硬件行为分别记录在[根审查](root-review.json)与[封存回执](receipt.json)。
