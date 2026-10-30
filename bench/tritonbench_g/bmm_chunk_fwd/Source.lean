import bench.tritonbench_g.bmm_chunk_fwd.Sources.Mathematical_Kernels
import VeriTile.Triton.Launch.Grid

/-! Source host selection, precision, copies, output allocation and exact grid.
This module contains no real correctness or FP equivalence proof. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.Host
open VeriTile.Triton
noncomputable section

structure Shape where
  batch : Nat
  seqlen : Nat
  ngroups : Nat
  K : Nat
  chunk : Nat
  grouped : Bool

def Shape.groups (h : Shape) := if h.grouped then h.ngroups else 1
def cdiv (n b : Nat) := (n + b - 1) / b
def Shape.chunks (h : Shape) := cdiv h.seqlen h.chunk
def Shape.inputShape (h : Shape) :=
  if h.grouped then [h.batch, h.seqlen, h.ngroups, h.K] else [h.batch, h.seqlen, h.K]
def Shape.outputShape (h : Shape) :=
  if h.grouped then [h.batch, h.chunks, h.ngroups, h.chunk, h.chunk]
  else [h.batch, h.chunks, h.chunk, h.chunk]
def Shape.outputSize (h : Shape) := h.batch * (h.chunks * h.groups) * h.chunk * h.chunk

structure Strides where
  batch : Nat
  row : Nat
  head : Nat
  column : Nat
  deriving DecidableEq

def contiguous (h : Shape) : Strides :=
  ⟨h.seqlen * h.groups * h.K, h.groups * h.K, if h.grouped then h.K else 0, 1⟩

/-- Exactly the source's `stride(-1) != 1 and stride(1) != 1` branch. -/
def copied (s : Strides) : Bool := decide (s.column ≠ 1 ∧ s.row ≠ 1)
def prepared (h : Shape) (s : Strides) : Strides :=
  if copied s then contiguous h else { s with head := if h.grouped then s.head else 0 }

def storageDType : TypedSource.Storage → FloatDType
  | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16
def outputDType (a : TypedSource.Storage) (requested : Option FloatDType) :=
  requested.getD (storageDType a)
def dotDType (a b : TypedSource.Storage) : FloatDType :=
  if a = .bf16 ∨ b = .bf16 then .bf16
  else if a = .fp16 ∨ b = .fp16 then .fp16 else .fp32

structure Config where
  M : Nat
  N : Nat
  K : Nat
  stages : Nat
  warps : Nat
  deriving DecidableEq

def configurations : List Config :=
  [⟨128,256,64,3,8⟩, ⟨64,256,32,4,4⟩, ⟨128,128,32,4,4⟩,
   ⟨128,64,32,4,4⟩, ⟨64,128,32,4,4⟩, ⟨128,32,32,4,4⟩,
   ⟨64,32,32,5,2⟩, ⟨32,64,32,5,2⟩, ⟨64,64,32,4,2⟩]

theorem configuration_positive (c : Config) (hc : c ∈ configurations) :
    0 < c.M ∧ 0 < c.N ∧ 0 < c.K := by
  simp only [configurations, List.mem_cons, List.not_mem_nil, or_false] at hc
  rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

structure Call where
  shape : Shape
  aStrides : Strides
  bStrides : Strides
  A : RegionName
  B : RegionName
  Out : RegionName
  Seq : Region .int
  seqBatch : Nat
  seqRow : Nat
  causal : Bool
  hasSeq : Bool
  aStorage : TypedSource.Storage := .fp32
  bStorage : TypedSource.Storage := .fp32
  requestedOutput : Option FloatDType := none

def Call.kernel (h : Call) (c : Config) (swapped : Bool) :=
  TypedSource.specialize h.aStorage h.bStorage swapped
    (outputDType h.aStorage h.requestedOutput) h.A h.B h.Out h.Seq
    h.shape.seqlen h.shape.chunk h.shape.K h.shape.groups
    (prepared h.shape h.aStrides).batch (prepared h.shape h.aStrides).row
    (prepared h.shape h.aStrides).head (prepared h.shape h.aStrides).column
    (prepared h.shape h.bStrides).batch (prepared h.shape h.bStrides).row
    (prepared h.shape h.bStrides).head (prepared h.shape h.bStrides).column
    ((h.shape.chunks * h.shape.groups) * h.shape.chunk * h.shape.chunk)
    (h.shape.groups * h.shape.chunk * h.shape.chunk)
    (if h.shape.grouped then h.shape.chunk * h.shape.chunk else 0) h.shape.chunk 1
    (if h.hasSeq then h.seqBatch else 0) (if h.hasSeq then h.seqRow else 0)
    c.M c.N c.K h.causal h.hasSeq

def Call.grid (h : Call) (c : Config) : Grid :=
  ⟨[cdiv h.shape.chunk c.M * cdiv h.shape.chunk c.N,
    h.shape.batch, h.shape.chunks * h.shape.groups]⟩

/-- Returned object: the source returns this fresh dense allocation directly. -/
structure Returned (h : Call) where
  region : RegionName := h.Out
  shape : List Nat := h.shape.outputShape
  dtype : FloatDType := outputDType h.aStorage h.requestedOutput

def returned (h : Call) : Returned h := {}

end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.Host

#guard_msgs (drop info) in
#auditModuleAxioms
