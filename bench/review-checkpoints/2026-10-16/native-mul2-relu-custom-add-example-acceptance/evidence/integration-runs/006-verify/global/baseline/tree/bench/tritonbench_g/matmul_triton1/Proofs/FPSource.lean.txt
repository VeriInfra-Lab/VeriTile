import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.TermModel
import VeriTile.Triton.Float.ScalarBackend

namespace VeriTile.Bench.TritonBenchG.MatmulTriton1.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

structure Parameters where
  (input weight output : RegionName)
  (rows columns blockRows blockInner blockColumns steps : Nat)
def inner (p : Parameters) : Nat := p.blockInner*p.steps

/-- Actual fp32 input/output storage, preserving the complete raw K loop,
allow_tf32=false, and the separate dot followed by accumulator addition.
The same-format cast on the dot result resolves the generic dtype boundary;
source fixes DTYPE to fp32 and backend preserves this cast bit-for-bit. -/
def typed (dtype : FloatDType) (p : Parameters) : ComputeKernel :=
  let input := p.input
  let weight := p.weight
  let output := p.output
  triton {
  tl.region input = DTYPE, weight = DTYPE, output = DTYPE
  pid = tl.program_id(0)
  num_n_blocks = tl.cdiv($(p.columns), $(p.blockColumns))
  m_block = pid // num_n_blocks
  n_block = pid % num_n_blocks
  m_offsets = tl.arange(0, $(p.blockRows)) + m_block * $(p.blockRows)
  n_offsets = tl.arange(0, $(p.blockColumns)) + n_block * $(p.blockColumns)
  k_offsets = tl.arange(0, $(p.blockInner))
  x_ptrs = input + m_offsets[:, None] * $(inner p) + k_offsets[None, :]
  y_ptrs = weight + k_offsets[:, None] * $(p.columns) + n_offsets[None, :]
  z_ptrs = output + m_offsets[:, None] * $(p.columns) + n_offsets[None, :]
  z = tl.zeros([$(p.blockRows), $(p.blockColumns)], dtype=DTYPE)
  for kk in range($(0), $(inner p), $(p.blockInner)) {
    x_sub = tl.load(x_ptrs)
    y_sub = tl.load(y_ptrs)
    z += (tl.dot(x_sub, y_sub, allow_tf32=false)).to(DTYPE)
    x_ptrs += $(p.blockInner)
    y_ptrs += $(p.blockInner) * $(p.columns)
  }
  tl.store(z_ptrs, z)
}

def source (p : Parameters) : ComputeKernel := typed .fp32 p
def prelude (p : Parameters) := (source p).surfaceBody.take 11
def body (p : Parameters) : List ComputeStmt := match (source p).surfaceBody[11]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code
  | _ => []
def epilogue (p : Parameters) := (source p).surfaceBody.drop 12

theorem source_decomposition (p : Parameters) : (source p).surfaceBody =
    prelude p ++ [.forRange "kk" 0 (inner p) p.blockInner (body p)] ++ epilogue p := by rfl

abbrev AIndex (p : Parameters) := TileIndex [p.blockRows,p.blockInner]
abbrev BIndex (p : Parameters) := TileIndex [p.blockInner,p.blockColumns]
abbrev CIndex (p : Parameters) := TileIndex [p.blockRows,p.blockColumns]

structure Primitives (α : Type) where
  zero : α
  add : α → α → α
  dot : Option ComputeDType → DotOptions → {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch ++ [m,k]) → α) → (TileIndex (batch ++ [k,n]) → α) →
    TileIndex (batch ++ [m,n]) → α

/-- The generated zero literal is tagged real before its fp32 cast. This
callback channel carries the fp32 zero payload; no mathematical Real evaluator
is involved. The only extra cast on the dot result is fp32-to-fp32 identity. -/
def backend {α : Type} (rest : Algebra α) (P : Primitives α) : Algebra α :=
  {rest with
    literal := fun precision dtype value => if (dtype = .fp32 ∨ dtype = .real) ∧ value = 0 then P.zero
      else rest.literal precision dtype value
    binary := fun precision dtype op a b => match dtype,op with
      | .fp32,.add => P.add a b
      | _,_ => rest.binary precision dtype op a b
    cast := fun precision src dst a => if (src=.fp32 ∨ src=.real) ∧ dst=.fp32 then a
      else rest.cast precision src dst a
    dotTyped := fun precision left right out options {batch} {m} {k} {n} a b =>
      match left,right,out with
      | .fp32,.fp32,.fp32 => some (P.dot precision options (batch:=batch) (m:=m) (k:=k) (n:=n) a b)
      | left,right,out => rest.dotTyped precision left right out options a b}

#axiomsClean source_decomposition
end
end VeriTile.Bench.TritonBenchG.MatmulTriton1.FPValue
