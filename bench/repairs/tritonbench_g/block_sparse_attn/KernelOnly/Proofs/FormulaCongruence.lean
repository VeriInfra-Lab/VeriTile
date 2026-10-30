import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Loop

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

/-- Equality only of the finite reads consumed by the scalar formula. The
states are proof-only views; this record contains no execution or output fact. -/
structure InputAgreement (a : Arithmetic α) (p : Parameters)
    (s t : State (Replicas α)) : Prop where
  pids_eq : s.pids=t.pids
  first_eq : first p s=first p t
  last_eq : last p s=last p t
  column_eq : ∀ visit,visit<visitCount p s →
    csrColumn p s (first p s+visit)=csrColumn p t (first p t+visit)
  query_eq : ∀ visit,visit<visitCount p s → ∀ block,block<p.blocks →
    query a p s block=query a p t block
  key_eq : ∀ visit,visit<visitCount p s → ∀ block,block<p.blocks →
    key a p s (csrColumn p s (first p s+visit)) block=
      key a p t (csrColumn p t (first p t+visit)) block
  value_eq : ∀ visit,visit<visitCount p s →
    value a p s (csrColumn p s (first p s+visit))=
      value a p t (csrColumn p t (first p t+visit))

namespace InputAgreement
theorem count {a : Arithmetic α} {p : Parameters} {s t : State (Replicas α)}
    (h : InputAgreement a p s t) : visitCount p s=visitCount p t := by
  simp only [visitCount,h.first_eq,h.last_eq]

theorem scores {a : Arithmetic α} {p : Parameters} {s t : State (Replicas α)}
    (h : InputAgreement a p s t) (native : Bool) (visit : Nat) (hv : visit<visitCount p s)
    (n : Nat) (hn : n≤p.blocks) :
    scoreAccumulation a native p s (csrColumn p s (first p s+visit)) n=
      scoreAccumulation a native p t (csrColumn p t (first p t+visit)) n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [scoreAccumulation_succ,scoreAccumulation_succ,ih (by omega)]
    simp only [scoreStep,h.query_eq visit hv n (by omega),h.key_eq visit hv n (by omega)]

theorem update {a : Arithmetic α} {p : Parameters} {s t : State (Replicas α)}
    (h : InputAgreement a p s t) (native : Bool) (visit : Nat) (hv : visit<visitCount p s)
    (carry : Carry α) :
    KernelOnly.update a native p s (csrColumn p s (first p s+visit)) carry=
      KernelOnly.update a native p t (csrColumn p t (first p t+visit)) carry := by
  have hr : row s=row t := by funext i; simp only [row,h.pids_eq]
  unfold KernelOnly.update
  rw [h.scores native visit hv p.blocks (Nat.le_refl _),h.value_eq visit hv]
  simp only [h.column_eq visit hv,hr]

theorem accumulation {a : Arithmetic α} {p : Parameters} {s t : State (Replicas α)}
    (h : InputAgreement a p s t) (native : Bool) (n : Nat) (hn : n≤visitCount p s) :
    KernelOnly.accumulation a native p s n=KernelOnly.accumulation a native p t n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [accumulation_succ,accumulation_succ,ih (by omega),h.update native n (by omega)]

theorem output {a : Arithmetic α} {p : Parameters} {s t : State (Replicas α)}
    (h : InputAgreement a p s t) (native : Bool) :
    KernelOnly.output a native p s=KernelOnly.output a native p t := by
  rw [KernelOnly.output,KernelOnly.output,h.accumulation native _ (Nat.le_refl _),h.count]

end InputAgreement
#axiomsClean InputAgreement.output
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
