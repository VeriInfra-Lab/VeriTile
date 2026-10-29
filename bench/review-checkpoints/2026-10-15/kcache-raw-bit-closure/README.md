# kcache：五格式原始位存储证明

新增证明覆盖原先接受的 `n_tokens=1`、参数化 `splitX` 完整复制程序：fp16、bf16、fp32、f8e4、f8e5 五种明确位宽，整个最终内存等于独立初始内存散射结果的编码。保留全部原参数、错误标签默认读、源／目标／元数据别名、零长度和任意初始寄存器、undef。

NaN 载荷和正负零不会被数值投影合并；非空重叠复制见证核对了读取初态快照。Canonical 是表示编码像的性质，没有作为额外输入限制。历史任意 R／64 位载荷分支另行保留，显式 fp64 load 仍不受解释器支持。

根代理在新目录独立编译 **4/4 模块成功，29 条具名公理检查通过**；另有 [110 项独立语义审查](independent-semantic-review/receipt.json)。仅新增 **1 条公开 specification**，其余为普通定理或支持定义。源与编译证据见 [候选说明](candidate/README.zh-CN.md)、[复编记录](root-replay/build-records.json)和[公理审查](axiom-review.json)。

这是带类型、按元素寻址的字存储软件结论。GPU 字节地址、整数索引对应、跨 program 并发、编译器和硬件执行尚未建立；没有新增数值舍入假设。多 token 分支不在原接受范围内。

此包冻结证明与复核，统计变更由后续 metadata 包单独记录。旧公开源均未改；候选说明中“必须追加四条显式 Lake 登记”已在[整合说明](integration-notes.json)纠正，现有递归注册应覆盖这些模块。
