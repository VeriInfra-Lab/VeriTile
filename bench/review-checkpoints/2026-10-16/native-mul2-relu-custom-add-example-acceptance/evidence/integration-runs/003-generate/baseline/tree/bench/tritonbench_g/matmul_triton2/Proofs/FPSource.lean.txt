import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.TermModel
import VeriTile.Triton.Float.ScalarBackend
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.TritonBenchG.MatmulTriton2.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
structure Parameters where
  (input weight output : RegionName)
  (rows columns inputRow inputInner weightInner weightColumn outputRow outputColumn
    blockRows blockColumns blockInner groupRows steps : Nat)
def inner (p : Parameters) : Nat := p.blockInner*p.steps

/-- All original group calculations, input masks, K iterations and output mask.
Dot options remain unspecified, exactly as in Python; no allow_tf32=false
annotation is inserted. The instantiated storage profile is binary32. -/
def typed (dtype : FloatDType) (p : Parameters) : ComputeKernel :=
  let input := p.input
  let weight := p.weight
  let output := p.output
  triton {
  tl.region input = DTYPE, weight = DTYPE, output = DTYPE
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(p.rows), $(p.blockRows))
  num_pid_n = tl.cdiv($(p.columns), $(p.blockColumns))
  num_pid_in_group = $(p.groupRows) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(p.groupRows)
  group_size_m = min(num_pid_m - first_pid_m, $(p.groupRows))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = pid_m * $(p.blockRows) + tl.arange(0, $(p.blockRows))
  offs_bn = pid_n * $(p.blockColumns) + tl.arange(0, $(p.blockColumns))
  offs_k = tl.arange(0, $(p.blockInner))
  a_ptrs = input + offs_am[:, None] * $(p.inputRow) + offs_k[None, :] * $(p.inputInner)
  b_ptrs = weight + offs_k[:, None] * $(p.weightInner) + offs_bn[None, :] * $(p.weightColumn)
  accumulator = tl.zeros([$(p.blockRows), $(p.blockColumns)], dtype=DTYPE)
  for k in range($(0), tl.cdiv($(inner p), $(p.blockInner)), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(inner p) - k * $(p.blockInner), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(inner p) - k * $(p.blockInner), other=0.0)
    accumulator += (tl.dot(a, b)).to(DTYPE)
    a_ptrs += $(p.blockInner) * $(p.inputInner)
    b_ptrs += $(p.blockInner) * $(p.weightInner)
  }
  c_ptrs = output + offs_am[:, None] * $(p.outputRow) + offs_bn[None, :] * $(p.outputColumn)
  c_mask = (offs_am[:, None] < $(p.rows)) & (offs_bn[None, :] < $(p.columns))
  tl.store(c_ptrs, accumulator, mask=c_mask)
}
def source (p : Parameters) : ComputeKernel := typed .fp32 p
def prelude (p : Parameters) := (source p).surfaceBody.take 15
def loopStmt (p : Parameters) := (source p).surfaceBody[15]'(by simp [source,typed,ComputeKernel.surfaceBody])
def body (p : Parameters) : List ComputeStmt := match (source p).surfaceBody[15]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ code) => code
  | _ => []
def epilogue (p : Parameters) := (source p).surfaceBody.drop 16

theorem source_decomposition (p : Parameters) : (source p).surfaceBody =
    prelude p ++ [loopStmt p] ++ epilogue p := by rfl

abbrev AIndex (p : Parameters) := TileIndex [p.blockRows,p.blockInner]
abbrev BIndex (p : Parameters) := TileIndex [p.blockInner,p.blockColumns]
abbrev CIndex (p : Parameters) := TileIndex [p.blockRows,p.blockColumns]
structure Primitives (α : Type) where
  zero : α
  add : α → α → α
  dot : Option ComputeDType → DotOptions → {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch ++ [m,k]) → α) → (TileIndex (batch ++ [k,n]) → α) →
    TileIndex (batch ++ [m,n]) → α

/-- Generated real-tag zero and its fp32 cast carry the supplied machine zero.
The added same-format cast materializes only a typed frontend boundary. -/
def backend {α : Type} (rest : Algebra α) (P : Primitives α) : Algebra α :=
  {rest with
    literal := fun precision dtype value => if (dtype=.fp32 ∨ dtype=.real) ∧ value=0 then P.zero
      else rest.literal precision dtype value
    cast := fun precision src dst a => if (src=.fp32 ∨ src=.real) ∧ dst=.fp32 then a
      else rest.cast precision src dst a
    binary := fun precision dtype op a b => match dtype,op with
      | .fp32,.add => P.add a b
      | _,_ => rest.binary precision dtype op a b
    dotTyped := fun precision left right out options {batch} {m} {k} {n} a b =>
      match left,right,out with
      | .fp32,.fp32,.fp32 => some (P.dot precision options (batch:=batch) (m:=m) (k:=k) (n:=n) a b)
      | left,right,out => rest.dotTyped precision left right out options a b}

#axiomsClean source_decomposition
end
end VeriTile.Bench.TritonBenchG.MatmulTriton2.FPValue
