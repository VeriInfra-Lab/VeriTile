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
直接整数参数赋值按原始语句顺序绑定，后续语句读取局部寄存器。尚不支持的
分支赋值、tuple／augmented 参数改写和循环变量遮蔽会报错。

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
不可变 constexpr 的 `Nat`、`Int`、`ℝ`、`Bool` 值从函数体外提供；constexpr
浮点数按 Python 弱标量处理，由周围的 tensor 运算决定精度，不强制改成 FP32。
constexpr 名称不能同时充当循环变量。`tl.zeros((B,), tl.float32)` 和
`tl.full((), 0.0, tl.float32)` 等原生 tuple 形状可直接粘贴；`(B)` 不是 tuple，
不会被当作 `[B]` 接受。
`range(N)` 的不可变 Nat 界限使用已有的 counted-loop 语义。原生整数
`.to(tl.float32)`／`.to(tl.float64)` 保留目标精度及 `fromNat`／`fromInt`
转换事件；整数直接转换到其他浮点格式尚不支持，会明确报错。原有机器整数
范围和索引投影的证明仍由对应调用契约提供。
当前任意表达式内部嵌套 load 的精度事件还不能完整表达，因此前端会报错，
不能把这种源码登记为迁移完成。浮点数乘布尔 mask 会保留乘法及 0/1 转换。

纯 helper 的原始函数体也可以单独粘贴，通过外部调用签名登记：

```lean
def reluFunction : SourceFunction :=
  triton_function relu_forward(x) in triton {
    return tl.where(x > 0, x, 0)
  }

def reluSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  functions := [reluFunction]
```

主函数体保留 `y = relu_forward(x)`。helper 参数类型由调用的实际值决定；
不会把整个 helper 结果作为假设或外部原语。支持已检查的顺序赋值和末尾
return，以及已登记 helper 的调用；双结果、三结果的直接赋值分别检查数量。
运行实参按源码顺序求值一次，helper 局部变量使用独立名称，返回值先完整求值
再写入调用者的目标。递归、自由名称、错误的参数／结果数量及不支持的控制流
仍会被拒绝。

helper 的未注解参数可以接收字面量或调用者的 constexpr；它们保留弱标量
性质，不会仅因传参就变成 tensor。例如 `divide(x, 4)` 中的 `4` 与
`denominator = 4; divide(x, denominator)` 中的整数 tensor 保持区别。
helper 内部重新赋值时，右侧先读取旧值，随后按普通赋值规则创建局部 tensor。
对照清单必须同时包含 helper 的原始 Python 与 Lean 定义，不能只核对调用名称。

局部 `BLOCK_SIZE_OUT: tl.constexpr = BLOCK_SIZE_IN * 2` 可保留在原函数体中。
当前支持独立 Nat constexpr 的加法、乘法及先前局部别名；不能把 Python 的
减法直接解释为 Nat 的截断减法。FP4 的完整 helper/caller 示例位于
`bench/repairs/tritonbench_g/fp4_to_bf16/Kernels.lean`。

已知 uint8/uint16 输入及受支持的别名、交错和位运算路径保留无符号宽度，
同种类弱标量须落在该类型范围内，加减乘按保留的宽度取模。`tl.where` 的已知
有符号分支保留 int8/int32 信息，越界弱标量会报错。这不等于通用整数类型推断
已完成：例如 `tl.full([], 8, tl.int32)` 后接位运算的有符号宽度传播仍未支持。
不支持的构造须修复并重证，不能通过删掉原始 cast 或改写函数体登记为通过。

原生 rank-one block pointer 支持完整的 positional singleton tuple 参数，
如 `tl.make_block_ptr(X, (N,), (stride,), (offset,), (B,), order=(0,))`。
前端在构造时保存逻辑坐标与 shape，以带符号整数累加元素地址，最后才转成
内存索引；因此后续改写原始坐标变量不会改变已经构造的指针。`boundary_check`
核对轴 0 并保留上下边界；默认 masked 值保持 undefined，显式 zero padding
保留为零。原始 load/store 的 element dtype cast 继续执行。尚未支持的指针
复制、advance、跨分支逃逸和 padding 类型会报错；现有高秩 named block
pointer 路径另行保留。这个索引 lowering 不代替 native 地址范围与无溢出证明。

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

Python comparison/select numeric literals receive the tensor operand dtype only
in the copied-function-body frontend. The historical typed DSL retains its
existing literal convention. Internal source-origin annotations are compiler
metadata and are rejected in user bodies and registered helper definitions.

普通整数赋值（例如 `denominator = 4`）按 Triton 3.7.1 的规则创建整数张量：
先选 int32，再选 uint32、int64、uint64；超出 64 位范围直接拒绝。参与浮点
运算时保留整数到浮点的转换。行内 `value / 4` 的 `4` 仍是弱标量，不能用
省略转换的方式把两种源码路径混为一谈。类型选择在编译器内部完成，粘贴的
函数体不需要添加类型声明。当前范围不涵盖尚未实现的其他 Python 语法。

严格 FP32 软件模型对整数值字面量与整数转换使用同一个最近偶数舍入器。
`FP.IntegerLiteral.from_int_literal` 证明两条路径在该明确模型中相等；它不是
关于任意数值端口的公理，也不要求整数恰好可表示。模型标识包含
`integer-literals`，与旧解释区分。具体硬件一致性仍是独立边界。
