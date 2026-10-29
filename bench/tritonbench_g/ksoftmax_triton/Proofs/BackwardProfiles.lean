import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardSource
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.Typed
open VeriTile.Triton
set_option maxRecDepth 16000

/-! Each definition elaborates twelve complete quotations. Splitting the
Cartesian family keeps elaboration bounded without changing source coverage. -/

def kernel_fp32_fp32 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float32, tl.float32, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float32, tl.float32, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float32, tl.float32, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float32, tl.float32, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float32, tl.float32, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float32, tl.float32, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float32, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float32, tl.float32, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float32, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float32, tl.float32, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float32, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float32, tl.float32, tl.float8e5, indexCommuted]

def kernel_fp32_fp64 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float32, tl.float64, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float32, tl.float64, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float32, tl.float64, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float32, tl.float64, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float32, tl.float64, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float32, tl.float64, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float32, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float32, tl.float64, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float32, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float32, tl.float64, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float32, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float32, tl.float64, tl.float8e5, indexCommuted]

def kernel_fp32_fp16 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float32, tl.float16, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float32, tl.float16, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float32, tl.float16, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float32, tl.float16, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float32, tl.float16, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float32, tl.float16, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float32, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float32, tl.float16, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float32, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float32, tl.float16, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float32, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float32, tl.float16, tl.float8e5, indexCommuted]

def kernel_fp32_bf16 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float32, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float32, tl.bfloat16, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float32, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float32, tl.bfloat16, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float32, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float32, tl.bfloat16, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float32, tl.bfloat16, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float32, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float32, tl.bfloat16, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float32, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float32, tl.bfloat16, tl.float8e5, indexCommuted]

def kernel_fp32_f8e4 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float32, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float32, tl.float8e4nv, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float32, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float32, tl.float8e4nv, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float32, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float32, tl.float8e4nv, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float32, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float32, tl.float8e4nv, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float32, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float32, tl.float8e4nv, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float32, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float32, tl.float8e4nv, tl.float8e5, indexCommuted]

def kernel_fp32_f8e5 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float32, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float32, tl.float8e5, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float32, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float32, tl.float8e5, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float32, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float32, tl.float8e5, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float32, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float32, tl.float8e5, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float32, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float32, tl.float8e5, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float32, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float32, tl.float8e5, tl.float8e5, indexCommuted]

def kernel_fp64_fp32 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float64, tl.float32, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float64, tl.float32, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float64, tl.float32, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float64, tl.float32, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float64, tl.float32, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float64, tl.float32, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float64, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float64, tl.float32, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float64, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float64, tl.float32, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float64, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float64, tl.float32, tl.float8e5, indexCommuted]

def kernel_fp64_fp64 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float64, tl.float64, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float64, tl.float64, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float64, tl.float64, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float64, tl.float64, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float64, tl.float64, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float64, tl.float64, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float64, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float64, tl.float64, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float64, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float64, tl.float64, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float64, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float64, tl.float64, tl.float8e5, indexCommuted]

def kernel_fp64_fp16 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float64, tl.float16, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float64, tl.float16, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float64, tl.float16, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float64, tl.float16, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float64, tl.float16, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float64, tl.float16, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float64, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float64, tl.float16, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float64, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float64, tl.float16, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float64, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float64, tl.float16, tl.float8e5, indexCommuted]

def kernel_fp64_bf16 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float64, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float64, tl.bfloat16, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float64, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float64, tl.bfloat16, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float64, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float64, tl.bfloat16, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float64, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float64, tl.bfloat16, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float64, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float64, tl.bfloat16, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float64, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float64, tl.bfloat16, tl.float8e5, indexCommuted]

def kernel_fp64_f8e4 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float64, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float64, tl.float8e4nv, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float64, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float64, tl.float8e4nv, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float64, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float64, tl.float8e4nv, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float64, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float64, tl.float8e4nv, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float64, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float64, tl.float8e4nv, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float64, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float64, tl.float8e4nv, tl.float8e5, indexCommuted]

def kernel_fp64_f8e5 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float64, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float64, tl.float8e5, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float64, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float64, tl.float8e5, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float64, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float64, tl.float8e5, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float64, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float64, tl.float8e5, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float64, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float64, tl.float8e5, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float64, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float64, tl.float8e5, tl.float8e5, indexCommuted]

def kernel_fp16_fp32 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float16, tl.float32, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float16, tl.float32, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float16, tl.float32, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float16, tl.float32, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float16, tl.float32, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float16, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float16, tl.float32, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float16, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float16, tl.float32, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float16, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float16, tl.float32, tl.float8e5, indexCommuted]

def kernel_fp16_fp64 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float16, tl.float64, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float16, tl.float64, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float16, tl.float64, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float16, tl.float64, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float16, tl.float64, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float16, tl.float64, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float16, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float16, tl.float64, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float16, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float16, tl.float64, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float16, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float16, tl.float64, tl.float8e5, indexCommuted]

def kernel_fp16_fp16 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float16, tl.float16, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float16, tl.float16, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float16, tl.float16, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float16, tl.float16, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float16, tl.float16, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float16, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float16, tl.float16, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float16, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float16, tl.float16, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float16, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float16, tl.float16, tl.float8e5, indexCommuted]

def kernel_fp16_bf16 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float16, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float16, tl.bfloat16, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float16, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float16, tl.bfloat16, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float16, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float16, tl.bfloat16, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float16, tl.bfloat16, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float16, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float16, tl.bfloat16, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float16, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float16, tl.bfloat16, tl.float8e5, indexCommuted]

def kernel_fp16_f8e4 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float16, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float16, tl.float8e4nv, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float16, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float16, tl.float8e4nv, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float16, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float16, tl.float8e4nv, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float16, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float16, tl.float8e4nv, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float16, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float16, tl.float8e4nv, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float16, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float16, tl.float8e4nv, tl.float8e5, indexCommuted]

def kernel_fp16_f8e5 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float16, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float16, tl.float8e5, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float16, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float16, tl.float8e5, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float16, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float16, tl.float8e5, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float16, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float16, tl.float8e5, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float16, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float16, tl.float8e5, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float16, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float16, tl.float8e5, tl.float8e5, indexCommuted]

def kernel_bf16_fp32 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.bfloat16, tl.float32, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.bfloat16, tl.float32, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.bfloat16, tl.float32, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.bfloat16, tl.float32, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.bfloat16, tl.float32, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.bfloat16, tl.float32, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.bfloat16, tl.float32, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.bfloat16, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.bfloat16, tl.float32, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.bfloat16, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.bfloat16, tl.float32, tl.float8e5, indexCommuted]

def kernel_bf16_fp64 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.bfloat16, tl.float64, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.bfloat16, tl.float64, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.bfloat16, tl.float64, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.bfloat16, tl.float64, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.bfloat16, tl.float64, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.bfloat16, tl.float64, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.bfloat16, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.bfloat16, tl.float64, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.bfloat16, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.bfloat16, tl.float64, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.bfloat16, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.bfloat16, tl.float64, tl.float8e5, indexCommuted]

def kernel_bf16_fp16 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.bfloat16, tl.float16, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.bfloat16, tl.float16, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.bfloat16, tl.float16, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.bfloat16, tl.float16, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.bfloat16, tl.float16, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.bfloat16, tl.float16, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.bfloat16, tl.float16, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.bfloat16, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.bfloat16, tl.float16, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.bfloat16, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.bfloat16, tl.float16, tl.float8e5, indexCommuted]

def kernel_bf16_bf16 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.bfloat16, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.bfloat16, tl.bfloat16, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.bfloat16, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.bfloat16, tl.bfloat16, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.bfloat16, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.bfloat16, tl.bfloat16, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.bfloat16, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.bfloat16, tl.bfloat16, tl.float8e5, indexCommuted]

def kernel_bf16_f8e4 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.bfloat16, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.bfloat16, tl.float8e4nv, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.bfloat16, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.bfloat16, tl.float8e4nv, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.bfloat16, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.bfloat16, tl.float8e4nv, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.bfloat16, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.bfloat16, tl.float8e4nv, tl.float8e5, indexCommuted]

def kernel_bf16_f8e5 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.bfloat16, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.bfloat16, tl.float8e5, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.bfloat16, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.bfloat16, tl.float8e5, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.bfloat16, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.bfloat16, tl.float8e5, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.bfloat16, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.bfloat16, tl.float8e5, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.bfloat16, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.bfloat16, tl.float8e5, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.bfloat16, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.bfloat16, tl.float8e5, tl.float8e5, indexCommuted]

def kernel_f8e4_fp32 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float8e4nv, tl.float32, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float8e4nv, tl.float32, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float8e4nv, tl.float32, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float8e4nv, tl.float32, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float8e4nv, tl.float32, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float8e4nv, tl.float32, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float8e4nv, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float8e4nv, tl.float32, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float8e4nv, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float8e4nv, tl.float32, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float8e4nv, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float8e4nv, tl.float32, tl.float8e5, indexCommuted]

def kernel_f8e4_fp64 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float8e4nv, tl.float64, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float8e4nv, tl.float64, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float8e4nv, tl.float64, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float8e4nv, tl.float64, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float8e4nv, tl.float64, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float8e4nv, tl.float64, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float8e4nv, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float8e4nv, tl.float64, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float8e4nv, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float8e4nv, tl.float64, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float8e4nv, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float8e4nv, tl.float64, tl.float8e5, indexCommuted]

def kernel_f8e4_fp16 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float8e4nv, tl.float16, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float8e4nv, tl.float16, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float8e4nv, tl.float16, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float8e4nv, tl.float16, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float8e4nv, tl.float16, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float8e4nv, tl.float16, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float8e4nv, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float8e4nv, tl.float16, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float8e4nv, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float8e4nv, tl.float16, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float8e4nv, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float8e4nv, tl.float16, tl.float8e5, indexCommuted]

def kernel_f8e4_bf16 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float8e4nv, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float8e4nv, tl.bfloat16, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float8e4nv, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float8e4nv, tl.bfloat16, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float8e4nv, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float8e4nv, tl.bfloat16, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float8e4nv, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float8e4nv, tl.bfloat16, tl.float8e5, indexCommuted]

def kernel_f8e4_f8e4 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float8e4nv, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float8e4nv, tl.float8e4nv, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float8e4nv, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float8e4nv, tl.float8e4nv, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float8e4nv, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float8e4nv, tl.float8e4nv, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, indexCommuted]

def kernel_f8e4_f8e5 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float8e4nv, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float8e4nv, tl.float8e5, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float8e4nv, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float8e4nv, tl.float8e5, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float8e4nv, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float8e4nv, tl.float8e5, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float8e4nv, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float8e4nv, tl.float8e5, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float8e4nv, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float8e4nv, tl.float8e5, tl.float8e5, indexCommuted]

def kernel_f8e5_fp32 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float8e5, tl.float32, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float8e5, tl.float32, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float8e5, tl.float32, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float8e5, tl.float32, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float8e5, tl.float32, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float8e5, tl.float32, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float8e5, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float8e5, tl.float32, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float8e5, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float8e5, tl.float32, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float8e5, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float8e5, tl.float32, tl.float8e5, indexCommuted]

def kernel_f8e5_fp64 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float8e5, tl.float64, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float8e5, tl.float64, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float8e5, tl.float64, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float8e5, tl.float64, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float8e5, tl.float64, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float8e5, tl.float64, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float8e5, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float8e5, tl.float64, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float8e5, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float8e5, tl.float64, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float8e5, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float8e5, tl.float64, tl.float8e5, indexCommuted]

def kernel_f8e5_fp16 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float8e5, tl.float16, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float8e5, tl.float16, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float8e5, tl.float16, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float8e5, tl.float16, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float8e5, tl.float16, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float8e5, tl.float16, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float8e5, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float8e5, tl.float16, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float8e5, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float8e5, tl.float16, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float8e5, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float8e5, tl.float16, tl.float8e5, indexCommuted]

def kernel_f8e5_bf16 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float8e5, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float8e5, tl.bfloat16, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float8e5, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float8e5, tl.bfloat16, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float8e5, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float8e5, tl.bfloat16, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float8e5, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float8e5, tl.bfloat16, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float8e5, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float8e5, tl.bfloat16, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float8e5, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float8e5, tl.bfloat16, tl.float8e5, indexCommuted]

def kernel_f8e5_f8e4 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float8e5, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float8e5, tl.float8e4nv, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float8e5, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float8e5, tl.float8e4nv, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float8e5, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float8e5, tl.float8e4nv, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float8e5, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float8e5, tl.float8e4nv, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float8e5, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float8e5, tl.float8e4nv, tl.float8e5, indexCommuted]

def kernel_f8e5_f8e5 (destination : Storage) (commuted : Bool) : Source :=
  match destination, commuted with
  | .fp32, Bool.false => backwardSource%[tl.float8e5, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => backwardSource%[tl.float8e5, tl.float8e5, tl.float32, indexCommuted]
  | .fp64, Bool.false => backwardSource%[tl.float8e5, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => backwardSource%[tl.float8e5, tl.float8e5, tl.float64, indexCommuted]
  | .fp16, Bool.false => backwardSource%[tl.float8e5, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => backwardSource%[tl.float8e5, tl.float8e5, tl.float16, indexCommuted]
  | .bf16, Bool.false => backwardSource%[tl.float8e5, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => backwardSource%[tl.float8e5, tl.float8e5, tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => backwardSource%[tl.float8e5, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => backwardSource%[tl.float8e5, tl.float8e5, tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => backwardSource%[tl.float8e5, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => backwardSource%[tl.float8e5, tl.float8e5, tl.float8e5, indexCommuted]

/-- The full Cartesian family of the six declared floating storage profiles.
The Python wrapper correlates IS_FP16 with destination storage, but does not
require equal storage types across the three buffers. Every case expands the
complete source quotation. Native admissibility remains a separate check. -/
def kernel (gradient saved destination : Storage) (commuted : Bool) : Source :=
  match gradient, saved with
  | .fp32, .fp32 => kernel_fp32_fp32 destination commuted
  | .fp32, .fp64 => kernel_fp32_fp64 destination commuted
  | .fp32, .fp16 => kernel_fp32_fp16 destination commuted
  | .fp32, .bf16 => kernel_fp32_bf16 destination commuted
  | .fp32, .f8e4 => kernel_fp32_f8e4 destination commuted
  | .fp32, .f8e5 => kernel_fp32_f8e5 destination commuted
  | .fp64, .fp32 => kernel_fp64_fp32 destination commuted
  | .fp64, .fp64 => kernel_fp64_fp64 destination commuted
  | .fp64, .fp16 => kernel_fp64_fp16 destination commuted
  | .fp64, .bf16 => kernel_fp64_bf16 destination commuted
  | .fp64, .f8e4 => kernel_fp64_f8e4 destination commuted
  | .fp64, .f8e5 => kernel_fp64_f8e5 destination commuted
  | .fp16, .fp32 => kernel_fp16_fp32 destination commuted
  | .fp16, .fp64 => kernel_fp16_fp64 destination commuted
  | .fp16, .fp16 => kernel_fp16_fp16 destination commuted
  | .fp16, .bf16 => kernel_fp16_bf16 destination commuted
  | .fp16, .f8e4 => kernel_fp16_f8e4 destination commuted
  | .fp16, .f8e5 => kernel_fp16_f8e5 destination commuted
  | .bf16, .fp32 => kernel_bf16_fp32 destination commuted
  | .bf16, .fp64 => kernel_bf16_fp64 destination commuted
  | .bf16, .fp16 => kernel_bf16_fp16 destination commuted
  | .bf16, .bf16 => kernel_bf16_bf16 destination commuted
  | .bf16, .f8e4 => kernel_bf16_f8e4 destination commuted
  | .bf16, .f8e5 => kernel_bf16_f8e5 destination commuted
  | .f8e4, .fp32 => kernel_f8e4_fp32 destination commuted
  | .f8e4, .fp64 => kernel_f8e4_fp64 destination commuted
  | .f8e4, .fp16 => kernel_f8e4_fp16 destination commuted
  | .f8e4, .bf16 => kernel_f8e4_bf16 destination commuted
  | .f8e4, .f8e4 => kernel_f8e4_f8e4 destination commuted
  | .f8e4, .f8e5 => kernel_f8e4_f8e5 destination commuted
  | .f8e5, .fp32 => kernel_f8e5_fp32 destination commuted
  | .f8e5, .fp64 => kernel_f8e5_fp64 destination commuted
  | .f8e5, .fp16 => kernel_f8e5_fp16 destination commuted
  | .f8e5, .bf16 => kernel_f8e5_bf16 destination commuted
  | .f8e5, .f8e4 => kernel_f8e5_f8e4 destination commuted
  | .f8e5, .f8e5 => kernel_f8e5_f8e5 destination commuted

/-- The previously checked homogeneous instances are literal members of
this independently typed family, rather than a new implementation. -/
theorem homogeneous_agrees (storage : Storage) (v : Bool) :
    kernel storage storage storage v = homogeneousKernel storage v := by
  cases storage <;> cases v <;> rfl

#axiomsClean homogeneous_agrees
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.Typed
