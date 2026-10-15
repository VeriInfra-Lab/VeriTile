# int4 完整循环反例

已安装 3 个模块，并独立复编译通过；原 host 配置下两轮 K 的明确位模型反例，说明 dot-from-zero 后加 accumulator 不能无条件替换为带 accumulator 的 dot。实际 TTIR 绑定已保存，但未证明 GPU 必然返回反例软件图的值。本项维持依赖后端分类。

子代理与 root 的三个新模块对象 SHA 完全一致，原基线源码和配置逐项 SHA 不变。所有命名定理的公理检查通过，未运行 GPU。

详见 [范围与剩余关系](candidate/README.md)、[定理清单](candidate/theorems.json)、[root 实际复编译](root-replay/receipt.json)、[安装清单](installation.json)、[验收回执](receipt.json)。
