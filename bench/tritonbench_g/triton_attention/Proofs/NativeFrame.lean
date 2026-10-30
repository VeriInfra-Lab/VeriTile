import bench.tritonbench_g.triton_attention.Proofs.NativeConnections
namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Native
open Memory
noncomputable section

def Written (c : Shape) (r : Buffer) (o : Nat) : Prop :=
  ((r = .o ∨ r = .newDO ∨ r = .dq ∨ r = .dk ∨ r = .dv) ∧ o < c.elements) ∨
  ((r = .l ∨ r = .m ∨ r = .delta) ∧ o < c.groups * c.rows)

private theorem tensor_frame (c : Shape) (s : Memory α) (b : Buffer)
    (x : Coord c → α) (r : Buffer) (o : Nat) (hn : ¬Written c r o)
    (hb : b = .o ∨ b = .newDO ∨ b = .dq ∨ b = .dk ∨ b = .dv) :
    writeTensor c s b x r o = s r o := by
  by_cases hr : r = b
  · apply writeTensor_outside
    by_contra! ho
    exact hn (Or.inl ⟨hr ▸ hb,ho⟩)
  · exact writeTensor_other c s b x r o hr

private theorem tensorWhere_frame (c : Shape) (s : Memory α) (b : Buffer)
    (keep : Coord c → Prop) (x : Coord c → α) (r : Buffer) (o : Nat) (hn : ¬Written c r o)
    (hb : b = .o ∨ b = .newDO ∨ b = .dq ∨ b = .dk ∨ b = .dv) :
    writeTensorWhere c s b keep x r o = s r o := by
  by_cases hr : r = b
  · apply writeTensorWhere_outside
    by_contra! ho
    exact hn (Or.inl ⟨hr ▸ hb,ho⟩)
  · exact writeTensorWhere_other c s b keep x r o hr

private theorem rows_frame (c : Shape) (s : Memory α) (b : Buffer)
    (x : Fin c.groups × Fin c.rows → α) (r : Buffer) (o : Nat) (hn : ¬Written c r o)
    (hb : b = .l ∨ b = .m ∨ b = .delta) :
    writeRows c s b x r o = s r o := by
  by_cases hr : r = b
  · apply writeRows_outside
    by_contra! ho
    exact hn (Or.inr ⟨hr ▸ hb,ho⟩)
  · exact writeRows_other c s b x r o hr

theorem backwardPrefix_frame (c : Shape) (p : Ops α) (scale : α) (s : Memory α)
    (n : Nat) (r : Buffer) (o : Nat) (hn : ¬Written c r o) :
    backwardPrefix c p scale s n r o = s r o := by
  induction n with
  | zero => exact tensor_frame c s .dq _ r o hn (by simp)
  | succ n ih =>
    simp only [backwardPrefix,backwardStep]
    repeat' rw [tensorWhere_frame c _ _ _ _ r o hn (by simp)]
    exact ih

theorem whole_frame (c : Shape) (p : Ops α) (owners : Owners) (s : Memory α)
    (scale : α) (r : Buffer) (o : Nat) (hn : ¬Written c r o) : run c p owners s scale r o = s r o := by
  rw [run,backwardPrefix_frame _ _ _ _ _ _ _ hn]
  simp only [preprocess,forward]
  repeat' first
    | rw [tensor_frame c _ _ _ r o hn (by simp)]
    | rw [rows_frame c _ _ _ r o hn (by simp)]

theorem whole_inputs (c : Shape) (p : Ops α) (owners : Owners) (s : Memory α)
    (scale : α) (o : Nat) :
    run c p owners s scale .q o = s .q o ∧ run c p owners s scale .k o = s .k o ∧
    run c p owners s scale .v o = s .v o ∧ run c p owners s scale .g o = s .g o := by
  refine ⟨?_,?_,?_,?_⟩ <;> apply whole_frame <;> simp [Written]

#axiomsClean backwardPrefix_frame
#axiomsClean whole_frame
#axiomsClean whole_inputs
end
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Native
