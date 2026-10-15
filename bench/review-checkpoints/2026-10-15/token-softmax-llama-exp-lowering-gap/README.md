# token_softmax_llama：来源桥、指数缺口与实验交接

独立复编了 **2 个来源/复用模块、7 条具名公理检查**，22 项工具测试、20 个原核 PTX 配置复核、2 个诊断离线编译通过。原任务仍依赖后端，公开规格数和完成数均不变。GPU 实际状态为 **NOT_RUN，0 次执行**。

原 Llama 与 Bloom 的 kernel 和 host 函数逐字节及 AST 相同；Llama 已有合同也使用同一完整行 reference。新增桥保留全部旧存储格式、长度、地址、特殊值、初态与 frame 范围。[Bloom 已证指数反例](../token-softmax-bloom-exp-lowering-gap/README.md)由此适用于 Llama 的内部指数操作数。父包的 3 个数值模块没有重复计为新证明，且没有推断最终 softmax 输出必然不同。

原核离线配置覆盖 SM80/90、fp32、BLOCK_SIZE=8/16/32/2048/4096、fusion 开关。12 个共有配置的可执行 PTX 文本相同；这不是 native 语义证明。[根独立来源/数据流复核](root-dataflow-review.json)、[编译记录](root-replay/last-build.json)、[实验契约](candidate/experiment-contract.json)、[详细边界](candidate/README.md)。

从本目录执行，输出目录须为新目录：

```sh
python3 -B candidate/verify.py
/opt/conda/bin/python3 -B candidate/capture.py compile --output /tmp/llama-review-offline --arch 80 --arch 90
/opt/conda/bin/python3 -B candidate/capture.py run --output /tmp/llama-review-capture --device cuda:0
```

原核采集保存完整分配及 frame，另一个诊断程序记录缩放/ex2 位值。没有完整 softmax 数值 oracle，运行最多记为 CHARACTERIZATION。`ex2.approx`、`div.full` 的设备位行为、真实归约树、转换与地址对应仍待确定；已证伪的直接 RN(exp) 普遍等式不能列为等待实验通过的假设。来源桥、固定实验和 GPU 正确性分别记录于[根审查](root-semantic-review.json)与[回执](receipt.json)。
