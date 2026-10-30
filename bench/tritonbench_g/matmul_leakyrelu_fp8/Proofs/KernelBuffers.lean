import bench.tritonbench_g.matmul_leakyrelu_fp8.SourceComputation

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.KernelBuffers
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked
noncomputable section

/-- Raw buffer address/stride/capacity data; no Tensor identity or API flags. -/
structure Buffer where
  storage : Nat
  offset : Nat
  rowStride : Nat
  columnStride : Nat
  allocationBytes : Nat
  allocationBase : BitVec 64
  pointer : BitVec 64

structure Geometry where
  M : Nat
  N : Nat
  K : Nat
  enabled : Bool

structure Buffers where
  A : Buffer
  B : Buffer
  C : Buffer

def Geometry.problem (g : Geometry) : Problem := ⟨g.M,g.N,g.K,g.enabled,.fp32⟩

def Buffer.view (b : Buffer) (rows columns : Nat) (dtype : Storage) : Tensor :=
  ⟨0,b.storage,b.offset,rows,columns,b.rowStride,b.columnStride,dtype,0,
    Bool.false,Bool.false,b.allocationBytes,b.allocationBase,b.pointer⟩

def Buffers.views (b : Buffers) (g : Geometry) : MatmulLeakyreluFp8.Buffers :=
  ⟨b.A.view g.M g.K .fp32,b.B.view g.K g.N .fp32,b.C.view g.M g.N .fp16⟩

/-- Expanded premises are only raw shape, address/capacity, stride and separation facts. -/
abbrev Layout (g : Geometry) (b : Buffers) := MatmulLeakyreluFp8.Layout g.problem (b.views g)

def Buffers.arguments (b : Buffers) (g : Geometry) : Host.Arguments :=
  ⟨(b.views g).A,(b.views g).B,if g.enabled then "leaky_relu" else ""⟩

@[simp] theorem arguments_problem (b : Buffers) (g : Geometry) : (b.arguments g).problem=g.problem := by
  cases g with
  | mk m n k enabled => cases enabled <;> rfl

/-- The zero-K branch is the retained raw output-fill program, with a complete memory map. -/
theorem zero_fill (ops : Operations α) (g : Geometry) (b : Buffers) (memory : Memory α) :
    ((zeroCode (b.views g).C).eval ops memory).2=zeroMemory ops (b.views g).C memory :=
  congrArg Prod.snd (eval_zero ops (b.views g).C memory)

end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.KernelBuffers
