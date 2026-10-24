# mean_reduction 逻辑矩阵主入口

本次只整理 `mean_reduction`。旧入口虽然展示了正确的均值公式，仍暴露完整
调度方法名、单输入通道编号、扁平地址与界限证明，可读性验收过宽。

新入口直接写为：

```lean
specification mean_grid_real (M N : Nat) (tiles : List Nat)
    (schedule : tiles.Perm (List.range (gridSize M))) :
    Spec.Real (meanIO M N tiles ⊨ fun xs row =>
      (∑ col : Fin N, xs row col) / (N : ℝ)) := by
  exact MeanIO.logical_implements M N tiles schedule
```

`IO.lean` 的 `matrixInputs` 和 `rowOutputs` 是双射，覆盖全部原输入及输出。
`meanIO` 直接引用原 `realIO` 和 `realSchedule`，没有复制内存描述或定义新执行器。
共享的 `LogicalScheduledKernelIO` 只在同一个完整调度判断上转换逻辑坐标。

`Proofs/RealIO.lean` 的 `matrixInputs_apply` 证明逻辑矩阵坐标对应原行主序地址；
`logical_mean_iff` 双向连接新旧完整判断，再复用原 `real_implements`。检查夹具
还验证带所有尺寸、调度参数及覆盖条件的完整量化命题等价。实际执行、访问安全、
全部行输出和写入范围外的原始内存 frame 均保留。空 grid 和形式语义中的 `N=0`
范围继续保留；Python 既有非空归约轴条件不变。

共享接口、题级 IO、连接证明、Real 与 FP 主文件的针对性构建，以及新旧命题
保持性／联合导入通过。只复用既有适用的内部程序证明，不宣称全依赖独立重放。
两个实际 FP 程序、计算顺序及空实验规则集不变；无新增数值实验或 GPU 实测。
整题完成数和既有数值延期数不变。

证据及源码哈希见 [review.json](review.json)；后续防护结果见
[validation.json](validation.json)。此前逐题证据与冻结构建产物保持不变，当前
清单和三个审查 manifest 只刷新本题；本次不声称重新按逻辑坐标标准验收全部 183 题。
