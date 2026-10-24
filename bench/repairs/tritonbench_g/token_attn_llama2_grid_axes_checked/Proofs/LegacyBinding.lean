import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.IOBinding

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.LegacyBinding
open VeriTile Triton FP.Structural Whole
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- Logical channels extracted from exactly the original finite arrays and
signed metadata. Shared read addresses keep all physical read-only aliases. -/
def values {α : Type} [Inhabited α] (c : Call) (m : Metadata c)
    (x : Buffers α c) : Interface.Inputs α c
  | ⟨0,_⟩ => fun j => finiteRead (α:=α) x.q (Reference.queryAddress c j.1.val j.2.1.val j.2.2.1.val)
  | ⟨1,_⟩ => fun j => finiteRead (α:=α) x.k (Reference.keyAddress c j.1.val j.2.1.val j.2.2.1.val)
  | ⟨2,_⟩ => fun j => m.len j.1.val
  | ⟨3,_⟩ => fun j => m.begin j.1.val
  | ⟨4,_⟩ => fun j => (m.locations j.1).toNat
  | ⟨i+5,h⟩ => False.elim (by simp [Interface.inputs] at h)

def encoded {α β : Type} (c : Call) (encode : RegionName → Nat → α → β)
    (x : Buffers α c) : Buffers β c where
  q := fun i => encode "Q" i.val (x.q i)
  k := fun i => encode "K" i.val (x.k i)
  out := fun i => encode "Out" i.val (x.out i)

theorem metadata {α : Type} [Inhabited α] (c : Call) (m : Metadata c) (x : Buffers α c) :
    Interface.MetadataValues c m (values c m x) := ⟨fun _ => rfl,fun _ => rfl,fun _ => rfl⟩

/-- Exact source-read binding of the original initial state to the new shared
IO. This is independent of any execution or final-output hypothesis. -/
theorem input {α β : Type} [Inhabited α] [Inhabited β]
    (M : Program → Algebra β) (c : Call) (m : Metadata c) (scale : ℝ)
    (work : List Program) (encode : RegionName → Nat → α → β) (observe : β → α)
    (x : Buffers α c) (rest : State β) (v : Valid c m) :
    (Interface.io M c m scale work encode observe).InputBinding (values c m x)
      (initial c m (encoded c encode x) rest) := by
  intro i j _
  fin_cases i
  · have hb := v.queryBounds j.1.val j.1.isLt j.2.1.val j.2.1.isLt j.2.2.1.val j.2.2.1.isLt
    simp [Interface.io,Interface.inputs,Interface.read,values,initial,encoded,finiteRead,
      Reference.queryAddress,hb,mapValue,Cell.read]
    rfl
  · have hb := v.keyBounds j.1.val j.1.isLt j.2.1.val j.2.1.isLt j.2.2.1.val j.2.2.1.isLt
    simp [Interface.io,Interface.inputs,Interface.read,values,initial,encoded,finiteRead,
      Reference.keyAddress,hb,mapValue,Cell.read]
    rfl
  · rfl
  · rfl
  · simp [Interface.io,Interface.inputs,Interface.read,values,initial,mapValue,Cell.read]
    rfl

#axiomsClean input
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.LegacyBinding
