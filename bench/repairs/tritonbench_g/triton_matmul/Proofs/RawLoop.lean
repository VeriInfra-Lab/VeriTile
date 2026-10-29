import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawBodyContext
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.SignedRange

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
open VeriTile.Triton FP.Structural FPValue
open ParallelAttention.FPStages (foldRange)
noncomputable section

def blockCount (p : Parameters) : Nat := (signedCeil p.K (BK p)).toNat

def accumulation {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) : CIndex p → α :=
  foldRange (RawSource.update M D p st s) 0 (blockCount p) 1
    (fun _ => M.literal (some .fp32) .real 0)

/-- The recurrence count is computed by the source, including the zero and
partial final block cases; no independent loop-count input is used. -/
theorem loopExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α)
    (ctx : Context p initial current)
    (ha : HasAcc p (fun _ => M.literal (some .fp32) .real 0) current) :
    ∃ t, rangeInt (backend M D) "k" 0 (signedCeil p.K (BK p)) 1 (body p st) current = some t ∧
      Context p initial t ∧ HasAcc p (RawSource.accumulation M D p st initial) t := by
  have hn : (blockCount p : Int) = signedCeil p.K (BK p) :=
    Int.toNat_of_nonneg (signedCeil_nonnegative _ _)
  rw [← hn]
  let I := fun (_ : Nat) (a : CIndex p → α) (s : State α) => Context p initial s ∧ HasAcc p a s
  exact VeriTile.Bench.Repairs.LayernormFwdTriton.SignedRange.values
    (backend M D) "k" (body p st) (RawSource.update M D p st initial)
    I (fun a s => Context p initial s ∧ HasAcc p a s) 0 (blockCount p) 1 _ current
    (by
      intro k a s h hk
      let t := bodyExecution M D p st initial s k a h.1 h.2
      exact ⟨t.val,t.property,bodyContext M D p st initial s k a h.1 h.2⟩)
    (by intro n a s h; exact h) ⟨ctx,ha⟩

#axiomsClean loopExecution
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
