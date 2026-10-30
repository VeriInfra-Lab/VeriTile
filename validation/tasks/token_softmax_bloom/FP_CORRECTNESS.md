# 逐算子浮点值合同

`FPCorrect.lean` 的 `token_softmax_typed_correctness` 从初始 logits 和序列元数据出发，执行完整原始行核：加载/补位、转 fp32、maximum、减法、exp、sum、除法、目标格式转换、masked store。独立值公式在 `token_softmax_bloom/Proofs/TypedContract.lean`，包含每次转换和每个 primitive 的精度；没有使用实数 exp 或实数求和替代浮点运算，也没有预先给定最终 softmax 答案。

存储格式参数覆盖全部 FloatDType；本任务 Python 测试的格式是 fp16。宽度正数保证最大值有输入，输出 token stride 非零保证读回无覆盖。其余地址、程序号、序列长度、输入值均参数化。frame 保证所有未写单元保持原值，允许输入输出别名并固定从初态读取。全源擦除等式仅证明地址/控制结构对应，不冒充数值等价。

这是 `backend_value_contract`：显式的 fp32 max/exp/sum primitive 仍由后端解释。还没有闭合的 GPU 位级正确性证明。部署到具体硬件前，需绑定实际 exp 指令、归约树、浮点舍入/FTZ、NaN/无穷/正负零行为、fp32 计算与目标格式转换，以及编译器融合选项。不能把这些回调设为实数精确运算后宣布完成；也不能把减最大值这一稳定化操作推断为已验证的额外代数规律。

原来的数学模型声明保留为历史/数学证据，新增合同通过独立公开入口纳入清单。普通输入条件、主机分配及 launch 组合维持原有边界。
