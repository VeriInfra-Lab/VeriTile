import bench.tritonbench_g.fused_rotary_embedding.Proofs.Core

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding

structure Ops (α : Type) where
  mul : α → α → α
  sub : α → α → α
  add : α → α → α

/-- Literal unchanged source arithmetic, retaining multiplication operands. -/
def sourcePair {α : Type} (p : Ops α) (x0 x1 c s : α) : α × α :=
  let a := p.mul x0 c
  let b := p.mul x1 s
  let u := p.mul x0 s
  let v := p.mul x1 c
  (p.sub a b,p.add u v)

noncomputable def realOps : Ops ℝ := ⟨(· * ·),(· - ·),(· + ·)⟩

end VeriTile.Bench.Repairs.FusedRotaryEmbedding
