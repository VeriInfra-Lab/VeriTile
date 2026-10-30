import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawInactive
import bench.repairs.tritonbench_g.triton_matmul.Proofs.SourceArithmetic
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
open VeriTile.Triton FP.Structural FPValue
noncomputable section

def cSafeOffset {α : Type} (p : Parameters) (initial : State α) (i : CIndex p) :=
  p.stride_cm*(if row p initial i.1<p.M then row p initial i.1 else 0) +
  p.stride_cn*(if col p initial i.2.1<p.N then col p initial i.2.1 else 0)

def writtenSafe {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial : State α) : State α :=
  (TileShape.allIndices [BM p,BN p]).foldl (fun t i => if active p initial i then
    t.write p.C (cSafeOffset p initial i) (stored st (outputValue M D p st initial i)) else t) initial

theorem writtenSafe_eq {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial : State α) : writtenSafe M D p st initial=written M D p st initial := by
  have stepEq (t : State α) (i : CIndex p) :
      (if active p initial i then t.write p.C (cSafeOffset p initial i)
        (stored st (outputValue M D p st initial i)) else t) =
      (if active p initial i then t.write p.C (cOffset p initial i)
        (stored st (outputValue M D p st initial i)) else t) := by
    by_cases h : active p initial i
    · simp [h,cSafeOffset,cOffset,h.1,h.2]
    · simp [h]
  unfold writtenSafe written
  simp_rw [stepEq]

/-- The pasted source computes its loop count from K. The host parameters
supply precisely this count. Invalid generic helper ids still write no output. -/
theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (r : Request) (initial : State α) :
    ∃ final, exec (backend M D) (typed r.parameters r.storage) initial = some final ∧
      final.mem = (written M D r.parameters r.storage initial).mem := by
  let p := r.parameters
  have hm : RawSource.signedCeil p.M (BM p) = (blocksM p : Int) := by
    simpa [p,Request.parameters,SourceScope.parameters,BM,blocksM] using
      (RawSource.signedCeil_configs r.rows).2.1
  have hn : RawSource.signedCeil p.N (BN p) = (blocksN p : Int) := by
    simpa [p,Request.parameters,SourceScope.parameters,BN,blocksN] using
      (RawSource.signedCeil_configs r.columns).2.2
  have hk : RawSource.blockCount p = p.numKBlocks := by
    cases hst : r.storage
    · simpa [p,Request.parameters,SourceScope.parameters,SourceScope.blockK,BK,RawSource.blockCount,hst] using
        congrArg Int.toNat (RawSource.signedCeil_configs r.inner).1
    · simpa [p,Request.parameters,SourceScope.parameters,SourceScope.blockK,BK,RawSource.blockCount,hst] using
        congrArg Int.toNat (RawSource.signedCeil_configs r.inner).2.1
  by_cases hf : firstM p initial ≤ blocksM p
  · obtain ⟨hpm,hpn⟩ := RawSource.control_coordinates p initial hm hn hf
    exact RawSource.execution_model M D p r.storage initial hpm hpn hk
  · have hcover : p.M ≤ blocksM p * BM p := by
      change r.rows ≤ ((r.rows+127)/128)*128
      omega
    obtain ⟨final,he,hmem⟩ := RawSource.execution_memory M D p r.storage initial
    refine ⟨final,he,?_⟩
    rw [hmem,RawSource.written_outside M D p r.storage initial hm hn hcover (by omega)]

#axiomsClean writtenSafe_eq
#axiomsClean execution_memory
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
