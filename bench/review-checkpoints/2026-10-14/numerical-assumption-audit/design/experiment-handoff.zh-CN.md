可复用现有实验入口，不必为178个kernel逐个新建whole-kernel接受原子。现有原子实验保存严格的身份、配置与CPU复验材料。

主算术关系（ADD-COMMUTE、MUL-COMMUTE、ADD-ASSOC 等，先以实际证明用量为准）：

```bash
python scripts/check_numerics.py check
python scripts/check_numerics.py run --rules ADD-COMMUTE,MUL-COMMUTE --formats fp32 --output Logs/assumptions-main
python scripts/check_numerics.py import Logs/assumptions-main --output Logs/assumptions-main-import.json
```

补充关系（例如 DIV-MUL-RCP）：

```bash
python scripts/check_numerics_supplement.py check
python scripts/check_numerics_supplement.py run --rules DIV-MUL-RCP --formats fp32 --output Logs/assumptions-supplement
python scripts/check_numerics_supplement.py import Logs/assumptions-supplement --output Logs/assumptions-supplement-import.json
```

这些命令交给用户在准备好的 NVIDIA GPU 环境执行；本次只读分析未执行 GPU。不要从上面的示例命令推断所有 kernel 恰好只用了三种关系，最终列表必须连接实际 proof atom 用量。

既有 report 只是既有精度、实现和采样合同下的结果，先复核来源和绑定，不能把“缺前提”一律记为“从未实验”。TypedScalarArithmetic 的 MUL-COMMUTE 使用 :typed-fp32 新绑定，其 validatedTyped 是现有 fp32 结果与新 Lean 片段的映射义务；默认重复旧 real-channel 实验并不能自动完成 typed-channel 绑定。新的 source/probe 应以实际 typed fp32 operands/output 为比较对象，包含 original source 编译后的定位证据。

bgmv_expand_slice 的 NativeAdditionCommutes 隐含条件在 Proofs/NativeFP.lean::requirementsFor 中按真正 reached addition 的 precision/dtype 选择。现有 fp32 real-channel ADD-COMMUTE 可供fp32分支的桥接；fp16/bf16 native addition 不能借用 bf16-input/fp32-compute profile 顶替。先提取实际 source add 的 dtype、default precision、casts、reachable branch，再复用或新增对应局部原子；不需要把 dot/reduction 和 whole BGMV 写成新准入假设。

全域 hfp16 不能通过有限实验验证：

```text
∀ x : ℝ, round_fp16(x)=x
```

标准 binary16 有限可表示集合，实数集合无限，该恒等式不成立。可用确定的 FP32 输入 x=1+2^-11（1.00048828125，binary16 在1处的半ULP）：ties-to-even 下 fp16(x)=1，而 identity(x)=1.00048828125。只采样已量化fp16再cast回fp16会全部通过，但只说明grid idempotence，不能证明上式。类似 x=1+2^-12 不在中点，也可避免依赖tie规则。

要移除该三条公开前提，有三种不同工作，不能混称“测通”：

- 将目标限制为所有实际cast输入均为fp16可表示值，并给出这些中间值的可表示性前提/证据。输入本身fp16并不保证乘积、softmax概率仍可表示。
- 保留每一步round，重写精确round-aware参考公式并证明。
- 为cast/round相关局部替换定义有误差的实验接受契约，再在该契约下推导；统计通过只得到该协议、输入域下的接受，不能把原精确等式改为无条件真。

建议新增数值清单和实验receipt分离。数值清单每行至少：

- obligation_id；kind=atomic_admission/native_scalar_law/global_identity/shared_model/native_binding。
- Lean来源：file、line、声明全名、准确 premise/formula、相关结构字段与def展开链、源文件SHA256。
- affected_public_declarations 和 task list；proof-used atom 与仅在 Rules 容器出现的 atom 分开。
- 左右表达式与准确operation graph；input/compute/accumulator/output formats；Lean carrier channel；intrinsic API；cast位置；default precision。
- guards（finite/nonzero等）、实际rewrite-site表达式、masked/eager lanes是否被检查。
- 已有report instance key/来源SHA256/decision、可复用原因或不匹配原因；不得只有规则名和PASS。
- experiment_action=reuse_and_bind / rerun_exact_profile / add_probe / refuted_requires_contract_change / model_boundary_only。
- experiment_status=NOT_RUN/EXISTING_SCOPED/REJECT/INCONCLUSIVE/ACCEPT；strict_universal_proof=false 除非有真正证明/有限穷举域证据。
- full_source_scope、未覆盖阶段、正常输入前提均独立字段。

receipt 尽量沿用 scripts/numerical_registry.py 的DIMENSIONS与 check_numerics importer：

- reference/candidate graph+lowering hashes；numerics semantics/version/per-node formats/rounding/NaN/subnormal/intrinsics。
- layout shapes/strides/reduction/scan/dot（不适用显式null）；backend device/capability/driver/compiler/version/options/launch。
- probe distribution/roles/joint_distribution/seed/quantization/special_values；protocol thresholds/version/decision_policy。
- manifest、每侧PTX/TTIR、record.json、observations.npz 与所有hash。
- valid/skipped样本、errors、bias/vars统计、阈值、最终decision；SMOKE_ONLY永不接受。
- 独立CPU replay output/hash/exit；单独 provenance-not-proof 字段承认GPU/编译器/oracle仍是信任边界。
- 对exact counterexample保存输入原始bits/有理值、双方bits/值、rounding模式、必需cast位置和counterexample status。不要用统计accept覆盖counterexample。

完整GPU bundle要返回，包括observations和PTX；只交summary PASS不足以重新计算gate。当前 importer做的是统计观察重算而不是原始GPU输出重放，不能写成“所有GPU执行独立复现”。
