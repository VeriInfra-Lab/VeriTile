import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawCoordinates

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
open VeriTile.Triton FP.Structural FPValue
noncomputable section

theorem innerMask_model (p : Parameters) (k : Nat) (i : Fin (BK p)) :
    innerMask p k i = decide (i.val < p.K - k * BK p) := by
  have h : ((i.val : Int) < (p.K : Int) - (k : Int) * (BK p : Int)) ↔
      i.val < p.K - k * BK p := by
    simp only [← Int.natCast_mul]
    omega
  simp only [innerMask,h]

theorem loadA_model {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (st : Storage) (s : State α) (hm : (control p s).pidM = (pidM p s : Int))
    (k : Nat) (i : AIndex p) : RawSource.loadA M p st s k i = aValue M p st s k i := by
  simp only [RawSource.loadA,innerMask_model,aValue,decide_eq_true_eq]
  by_cases h : i.2.1.val < p.K - k * BK p
  · simp only [h,if_true]
    congr 2
    simp only [RawSource.addressA,offsetM_model p s hm,innerOffset,innerMask_model,h,
      decide_true,ite_true,← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast,aOffset]
    ring
  · simp [h]

theorem loadB_model {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (st : Storage) (s : State α) (hn : (control p s).pidN = (pidN p s : Int))
    (k : Nat) (i : BIndex p) : RawSource.loadB M p st s k i = bValue M p st s k i := by
  simp only [RawSource.loadB,innerMask_model,bValue,decide_eq_true_eq]
  by_cases h : i.1.val < p.K - k * BK p
  · simp only [h,if_true]
    congr 2
    simp only [RawSource.addressB,offsetN_model p s hn,innerOffset,innerMask_model,h,
      decide_true,ite_true,← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast,bOffset]
    ring
  · simp [h]

/-- Identical primitive calls, in the same order, for an arbitrary floating
algebra and dot primitive; the bridge uses only integer-address equalities. -/
theorem update_model {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α)
    (hm : (control p s).pidM = (pidM p s : Int))
    (hn : (control p s).pidN = (pidN p s : Int)) (k : Nat) (a : CIndex p → α) :
    RawSource.update M D p st s k a = FPValue.update M D p st s k a := by
  unfold RawSource.update FPValue.update
  apply congrArg₂ (fun (av : AIndex p → α) (bv : BIndex p → α) =>
    D.apply (some .fp32) (dtype st) (dtype st) .real {} (batch := []) av bv a)
  · funext i; exact loadA_model M p st s hm k i
  · funext i; exact loadB_model M p st s hn k i

theorem accumulation_model {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α)
    (hm : (control p s).pidM = (pidM p s : Int))
    (hn : (control p s).pidN = (pidN p s : Int))
    (hk : blockCount p = p.numKBlocks) :
    RawSource.accumulation M D p st s = FPValue.accumulation M D p st s := by
  have hu : RawSource.update M D p st s = FPValue.update M D p st s := by
    funext k a; exact update_model M D p st s hm hn k a
  simp only [RawSource.accumulation,FPValue.accumulation,hu,hk]

#axiomsClean innerMask_model
#axiomsClean loadA_model
#axiomsClean loadB_model
#axiomsClean update_model
#axiomsClean accumulation_model
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
