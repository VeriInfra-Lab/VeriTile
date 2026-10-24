# 直接粘贴 Python 函数体

`Kernels.lean` 的函数体入口保留原始 Python 的变量、语句、cast、分支、mask
和运算顺序。调用信息写在外层；不在函数体中添加 `tl.region`、`tl.param`
或 Lean antiquote。当前入口接收函数体，不接收 decorator 和 Python 函数头。

例如，以下函数体可以直接放进 quotation：

```lean
def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("n",.i32)]
  constexpr := ["BLOCK"]

def source (X Y : RegionName) (n BLOCK : Nat) : ComputeKernel :=
  with_triton_call signature call_args [n := n] in
  triton {
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < n
    x = tl.load(X + offsets, mask=active, other=0)
    tl.store(Y + offsets, x, mask=active)
  }
```

`n` 仍是 Python 的运行参数。`call_args` 给出本次验证调用的实际整数值，
不会把它改成 `tl.constexpr`。已有数学索引投影使用 `Nat` 或 `Int`；`.i32`
和 `.i64` 记录源 ABI，原有 native 范围、地址和宿主连接证明仍须成立。
这个入口本身不提供固定宽度算术不溢出的证明。

FP32/FP64 运行标量保留为有精度的寄存器，由题级 IO 调用绑定。例如
`l2_norm_triton1` 只特化整数 `N`，epsilon 仍由实际调用提供。所有标量均以
寄存器提供时可以使用 `with_triton_signature signature in triton { ... }`。
目前按调用特化的整数参数不能在函数体中重新赋值或被循环变量遮蔽；不支持的
情况会报错。

在每次使用都已有明确 FP32 计算精度的上下文中，也可用 `call_args` 绑定
具有 `ℝ` 载体的 FP32 调用值。前端先检查精度，再发布程序；FP64、FP16 或
无法确认精度的使用会被拒绝。这不会把运行标量改成 constexpr；通用的运行
标量接口仍使用寄存器。

调用签名可用 `readPorts`、`writePorts` 明确公开端口，例如原始参数指针在
load 前执行了 `X += offset`。这只是接口元数据，不是访存安全或成功执行的
证明；编译器能识别的实际访问必须出现在声明中。没有指定时按签名中的指针
顺序收集访问，并去重。

原始注释、docstring、跨行调用、调用末尾逗号和 Python 缩进保存在 quotation
中。前端只适配执行语法；不支持的表达式或控制流会报错。隐式 store 转换由
外部存储类型决定，显式 `.to(...)` 留在原源码中。

普通 `range` 循环保留原始函数体。已知类型的循环状态必须在每条更新中保持
类型；无法验证的控制流会被拒绝。循环界限只有在确认来自不可变的本次调用值
或 constexpr 参数时才会特化，函数体内被覆盖的参数继续按局部值执行。
当前任意表达式内部嵌套 load 的精度事件还不能完整表达，因此前端会报错，
不能把这种源码登记为迁移完成。浮点数乘布尔 mask 会保留乘法及 0/1 转换。

每题的 `jit-correspondence.json` 指向真实选定 Python 和实际 Lean 定义。
检查器比较完整 Python AST，并核对外部签名与调用绑定：

```bash
python3 -m bench.jit_correspondence path/to/jit-correspondence.json
python3 -m bench.check_body_corpus --check
python3 -m bench.check_body_corpus --check --require-complete
```

AST 相同只是源码检查。迁移还须重放公开 Real/FP 证明，并保留完整 IO、执行、
输出、安全和 frame。旧翻译正确时可证明完整程序相等；发现旧翻译多了转换等
实际差异时，应修正执行公式并重新证明。不能用不成立的保持性断言复用旧结果。

全量清单中的 pending 项仍是未完成。Lean 最先编写、没有历史 Python 的示例
也单独记录来源；新增 native 配套不能充当独立的历史原版证据。

用户确认无可追溯源码时，可以依据独立数学规格新建 Python：登记为
`spec_derived_native_source`，注明新来源，并固定数学规格文件的 SHA256。
检查通过后状态为 `spec_derived_body_copy_checked`，与历史 Python 对照状态
分别统计。语法对照仍须配合实际程序的 Real、FP、执行和内存证明重新验证。
