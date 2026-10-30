import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawLaunch
import bench.tritonbench_g.triton_matmul.Proofs.FPComplete

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
open VeriTile.Triton FP.Structural FPValue
noncomputable section

theorem active_model {α : Type} (p : Parameters) (s : State α)
    (hm : (control p s).pidM = (pidM p s : Int))
    (hn : (control p s).pidN = (pidN p s : Int)) (i : CIndex p) :
    RawSource.active p s i = decide (FPValue.active p s i) := by
  simp [RawSource.active,maskM_model p s hm,maskN_model p s hn,FPValue.active]

theorem outputOffset_model {α : Type} (p : Parameters) (s : State α)
    (hm : (control p s).pidM = (pidM p s : Int))
    (hn : (control p s).pidN = (pidN p s : Int)) (i : CIndex p)
    (hi : FPValue.active p s i) :
    RawSource.outputOffset p s i = cOffset p s i := by
  simp only [RawSource.outputOffset,offsetM_model p s hm,offsetN_model p s hn,
    aRow,bCol,hi.1,hi.2,if_true,cOffset,← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast]

/-- Final-memory equality to the independent coordinate recurrence. This does
not claim AST equality or equality of the source's temporary registers. -/
theorem written_model {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α)
    (hm : (control p s).pidM = (pidM p s : Int))
    (hn : (control p s).pidN = (pidN p s : Int))
    (hk : blockCount p = p.numKBlocks) :
    RawSource.written M D p st s = FPValue.written M D p st s := by
  have hv (i : CIndex p) :
      RawSource.outputValue M D p st s i = FPValue.outputValue M D p st s i := by
    simp only [RawSource.outputValue,FPValue.outputValue,accumulation_model M D p st s hm hn hk]
  have he (t : State α) (i : CIndex p) :
      (if RawSource.active p s i then
        t.write p.C (RawSource.outputOffset p s i) (stored st (RawSource.outputValue M D p st s i)) else t) =
      (if FPValue.active p s i then t.write p.C (cOffset p s i)
        (stored st (FPValue.outputValue M D p st s i)) else t) := by
    rw [active_model p s hm hn]
    by_cases hi : FPValue.active p s i
    · simp [hi,hv,outputOffset_model p s hm hn i hi]
    · simp [hi]
  unfold RawSource.written FPValue.written
  simp_rw [he]

theorem execution_model {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α)
    (hm : (control p s).pidM = (pidM p s : Int))
    (hn : (control p s).pidN = (pidN p s : Int))
    (hk : blockCount p = p.numKBlocks) :
    ∃ final, exec (backend M D) (Source.typed p st) s = some final ∧
      final.mem = (FPValue.written M D p st s).mem := by
  obtain ⟨final,he,hmem⟩ := RawSource.execution_memory M D p st s
  exact ⟨final,he,hmem.trans (congrArg State.mem (written_model M D p st s hm hn hk))⟩

#axiomsClean active_model
#axiomsClean outputOffset_model
#axiomsClean written_model
#axiomsClean execution_model
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
