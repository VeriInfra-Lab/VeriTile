import bench.repairs.tritonbench_g.flash_attn.KernelOnly.Layout

namespace VeriTile.Bench.Repairs.FlashAttn.ActualSource
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernel.Whole001.Scatter
noncomputable section
set_option maxHeartbeats 8000000
variable {α : Type} [Inhabited α]

def InputsEq (a b : Mem α) : Prop := ∀ r∈(["Q","K","V"] : List RegionName),∀ off,a r off=b r off

theorem accumulated_input_eq (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (job : Program) (a b : Mem α) (h : InputsEq a b) :
    accumulated M dot p (stateAt p job a) scale=accumulated M dot p (stateAt p job b) scale := by
  have q : query M p (stateAt p job a) scale=query M p (stateAt p job b) scale := by
    funext i
    simp only [query,qptr,base,stateAt,h "Q" (by simp)]
  have k : ∀ start,key M p (stateAt p job a) start=key M p (stateAt p job b) start := by
    intro start;funext i
    simp only [key,kptr,base,stateAt,h "K" (by simp)]
  have v : ∀ start,value M p (stateAt p job a) start=value M p (stateAt p job b) start := by
    intro start;funext i
    simp only [value,vptr,base,stateAt,h "V" (by simp)]
  have step : update M dot p (stateAt p job a) scale=update M dot p (stateAt p job b) scale := by
    funext start acc
    delta update maximum probability rescale scores contribution
    simp only [q,k,v]
    rfl
  unfold accumulated
  rw [step]
  rfl

theorem grid_inputs (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (jobs : List Program) (memory : Mem α) : InputsEq (referenceGrid M dot p scale jobs memory) memory :=
  fun r hr off=>input_frame M dot p scale jobs memory r hr off

theorem program_output (M : Algebra α) (dot : Dot α) (x : Layout) (hs : x.shape.Valid)
    (ho : OutputInjective x) (scale : α) (job : Program) (valid : job.Valid x) (memory : Mem α)
    (i : TileIndex [128,(parameters x).DIM]) (hi : active (parameters x) (stateAt (parameters x) job memory) i) :
    referenceProgram M dot (parameters x) scale job memory "O"
      (address (parameters x) (stateAt (parameters x) job memory) i)=
      .mk .real (outputValue M (parameters x)
        (accumulated M dot (parameters x) (stateAt (parameters x) job memory) scale) i) := by
  unfold referenceProgram written
  apply masked_readback_active
  · exact hi
  · intro a b ha hb he
    have batch : job.2/x.shape.heads<x.shape.batches := (Nat.div_lt_iff_lt_mul hs.2.1).mpr valid.2
    have head : job.2%x.shape.heads<x.shape.heads := Nat.mod_lt _ hs.2.1
    have same:=ho (job.2/x.shape.heads) (job.2%x.shape.heads) (job.1*128+a.1.val) a.2.1.val
      (job.2/x.shape.heads) (job.2%x.shape.heads) (job.1*128+b.1.val) b.2.1.val
      ⟨batch,head,ha,a.2.1.isLt⟩ ⟨batch,head,hb,b.2.1.isLt⟩ (by
        simpa [address,base,stateAt,parameters,FlashAttn.address] using he)
    apply Prod.ext
    · apply Fin.ext;omega
    · apply Prod.ext
      · exact Fin.ext same.2.2.2
      · exact Subsingleton.elim _ _

theorem program_log (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (job : Program) (memory : Mem α) (i : TileIndex [128])
    (hi : lActive p (stateAt p job memory) i) :
    referenceProgram M dot p scale job memory "L" (lAddress p (stateAt p job memory) i)=
      .mk .real (logValue M p (accumulated M dot p (stateAt p job memory) scale) i) := by
  unfold referenceProgram written
  rw [State.masked_scatter_frame]
  · apply masked_readback_active
    · exact hi
    · intro a b _ _ he
      apply Prod.ext
      · apply Fin.ext
        simp only [lAddress,stateAt] at he
        omega
      · exact Subsingleton.elim _ _
  · exact Or.inl (by decide)

#axiomsClean accumulated_input_eq
#axiomsClean program_output
#axiomsClean program_log
end
end VeriTile.Bench.Repairs.FlashAttn.ActualSource
