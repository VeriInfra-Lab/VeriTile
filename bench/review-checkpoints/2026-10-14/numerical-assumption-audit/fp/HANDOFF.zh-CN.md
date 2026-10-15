FP 数值假设审计交付

155 个 admitted_fp 公开声明中，92 个实际不使用数值 atom；63 个声明（45 个任务）使用 4 种精确绑定。

| 绑定 | 声明数 | 已有证据 | 仍需明确的内容 |
|---|---:|---|---|
| fp32_add_commute_real | 34 | PASS/PASS，4096 replicates | published_row_bound_under_explicit_scoped_model_premise |
| fp32_mul_commute_real | 19 | PASS/PASS，4096 replicates | published_row_bound_under_explicit_scoped_model_premise |
| fp32_div_mul_rcp_real | 10 | PASS/PASS，4096 replicates | published_row_bound_under_explicit_scoped_model_premise |
| fp32_mul_commute_typed | 2 | PASS/PASS，4096 replicates | explicit_typed_binding_validation_pending |

主证据是已完成八库验收中的 154 条 proof-directed 输出、seeded_dropout 的新 Lean 查询以及四个固定 entry key 的新 Lean 求值。3379 个 Lean 输入哈希与完成验收快照完全一致。

附加：115 条后半区 proof-walk 查询 exit 0；113 条直接输出全部 instance_key。两 context 的 typed key 是闭合字符串拼接，原 whnf-only 字符串读取器输出 null，完整 key 已由另一次 #eval 与精确源码定义确认。未把此显示限制计为数值规则未知，也未声称该 printer 已直接解析两个 key。重复全155查询被停止，exit 143，保留日志，不计PASS。

两份实际使用报告的 exporter load/render 只读检查通过，Lean 导出完全一致；历史源哈希对应的 5 个旧源码从 git 恢复。普通 DIV-MUL-RCP 的当前/原始 rnd+elementwise specialization AST 完全相同。主报告 DIV-RCP 使用 tl.div_rn，fp32 WARN_NOT_ACCEPTED，未用于这些声明。

Typed MUL 的离线编译只确认实际 fp32 操作和源映射，不执行GPU、不重放样本、不会生成 EvidenceValidated。统计报告也不证明普遍IEEE相等或整个kernel统计门可组合。12 个 test_fp_assumptions 测试通过。

交付文件：audit.json 全信息；rows.json 155声明映射；obligations.json 4条实验任务（公式、dtype、precision、domain、已有报告、源码引用）；verification-receipt.json 命令与终态。
