当前验收缺口中的通用工具链要求应收敛，但真实源码缺口仍然保留。

用户标准是覆盖原程序所有声明阶段、dtype 和分支；明确登记的 numerical assumptions 可以延期。`check_original_program_scope.py` 并没有另设通用 Python、Torch、编译器、GPU 正确性门槛。它检查 exact scope metadata、公开执行契约、数值登记和 root 的实质审查。`current_source_audit.py:369` 仍明确写着 native_functional_correctness=not_established。

19 条 ReLU/vector 当前 structural_gaps 已在 `calibration.json` 逐条保留原文并拆分：真实 source-to-model、typed host/grid、地址域、dtype 路线和 source-visible 错误继续阻塞；通用 JIT/cache、LLVM/backend、allocator/MMU、debug/sanitizer、native exception 字符串和 import-time harness 不再混作额外原程序缺口。完整证据与逐行哈希在 `evidence.json`。

有限验收定义：固定原源码和 production entrypoints；有限、基于实际来源的 dtype/configuration 域；全部成功路线的执行与输出/frame；被拒绝路线的真实首个拒绝阶段；可信 Python/Torch/launch primitives 下的有限 host 参数与结果连接；实际内存/整数地址/访问冲突语义；逐 statement/case/port 登记未验证数值对应；最后绑定 schema2 与独立实质审查。测试 helper 经现有 checker 的 JIT/调用关系检查后标注 test_harness，不证明整个 import/RNG 系统。

ReLU 旧证据已重新核对：原源码哈希相符；17 个常规输入中13个前端生成、4个 fp8 拒绝。现有12个成功 typed source profile 已覆盖整数/bool 与 fp16/bf16/fp32，真正缺 fp64。不能要求4个被拒绝 fp8 成功执行，也不应因缺浮点 byte-grid 扩展而否认已有 logical execution。fp64 下一步需保留64-bit load、ordered comparison、f64 select、f32 store narrowing，以及所有pid的预门控计算和仅pid0写回。

Vector addition 的189个成功 dtype 路线不需要统一成单一解释器，但必须用一个有限、源码对应的 host dispatch 连接。289个常规 pair 的实际 dtype admission 划分为189成功、64签名拒绝、36加法拒绝；所谓28个“promotion成功但缺grid”的fnuz组合就在64个签名拒绝内。现有 protocol tables 不可直接冒充执行契约；旧 host logicalLaunch 仍是 .real 路线，需要typed连接。9个fp64整数/bool输出路线的 OutputDefined 也不可被隐去。

量化新 root-review 的“Native physical memory/runtime/launch refinement”同样应拆分：通用运行时实现移到信任边界；abs/max/unsqueeze/reciprocal 到 scale 的连接、实际分配/断言/返回、完整grid、transpose grouped pid和其他实际admitted dtype仍是有限结构缺口。llrintf与浮点/窄化对应可精确登记延期；已实现的整数寄存器与int8 store不能被无穷硬件要求重新变成未实现。

最接近闭合的候选是 matrix_transpose、relu_triton_kernel、vector_addition；具体动作见 JSON。matrix_transpose 固定fp16且已有全host按位契约，主要是明确合法arange配置和源primitive连接。按父agent当前优先级，下一项具体实现应是ReLU剩余fp64。ReLU在N>1024时留下尾部的已知源码错误必须保留：全源码行为刻画不等于全tensor数学ReLU正确。

本包建议保留历史冻结记录，新增边界校准补充，再将当前 structural_gaps 改成带源位置和有限闭合动作的真实缺口。没有仓库修改、Lean/前端/GPU新执行或验收增量。父agent审查此包后再决定后续实现。
