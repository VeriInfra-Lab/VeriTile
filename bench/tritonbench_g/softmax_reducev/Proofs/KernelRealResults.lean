import bench.tritonbench_g.softmax_reducev.Proofs.KernelInterface
import bench.tritonbench_g.softmax_reducev.Proofs.RealPublic

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.KernelRealResults
open VeriTile.Triton FP.Structural
noncomputable section

/-- Mathematical interpretation of the raw caller words and integer metadata. -/
def evaluate (c : Host.Call) (m : Host.Memory (WithBot ℝ)) :=
  Host.evaluate RealModel.model c (Host.decodeMemory Host.realIndex m)

/-- Complete shared-source grid, independent normalized weighted sums and the
exact raw-memory frame. Negative infinity has zero mass; every legal row must
have a finite active logit and finite active gathered values. -/
theorem kernel_real (c : Host.Call) (m : Host.Memory (WithBot ℝ))
    (logics : Nat → WithBot ℝ) (values : Nat → ℝ)
    (normalizable : NormalizableInputs c (Host.decodeMemory Host.realIndex m) logics values)
    (work : List Host.Job) (complete : work.Perm (Host.workset c)) :
    ∃ final,RawKernel.runGrid c (evaluate c m) work m=.ok () final ∧
      (∀ j,Host.Legal c j → ∀ d,d<c.dimensions →
        final "Out" (Host.address c j d)=
          ↑(mathematical c (Host.decodeMemory Host.realIndex m) logics values j d)) ∧
      (∀ r i,Host.Outside c 0 r i → final r i=m r i) := by
  let x := Host.decodeMemory Host.realIndex m
  obtain ⟨final,run,results,frame⟩ := Checked.source_whole c (evaluate c m)
    (Host.row RealModel.model c x) ⟨Host.source_law RealModel.model c x⟩
    (Host.source_total RealModel.model c x) work complete m
  refine ⟨final,run,?_,frame⟩
  intro j hj d hd
  rw [results j hj d hd]
  have positive : 0<x.counts j.batch := by
    obtain ⟨k,r,hk,_⟩ := normalizable.positive j hj
    exact lt_of_le_of_lt (Nat.zero_le k) hk
  have result := RealModel.source_value (Host.parameters c)
    (Host.launch c x (fun _ => default) j)
    (fun k => logics (logicAddress c x j k))
    (fun k d => values (valueAddress c x j k d.val))
    (real_inputs c x logics values normalizable j hj)
    (by change 0<64;decide)
    (by simpa [FPValue.sequenceLength,Host.parameters,Host.launch,Region.cast] using positive)
    ((⟨d,hd⟩ : Fin c.dimensions),())
  simpa only [Host.row,dif_pos hd,mathematical,FPValue.sequenceLength,
    Host.parameters,Host.launch,Region.cast,Cell.read_same,↓reduceIte,reduceCtorEq] using result

#axiomsClean kernel_real
#specNonCircular mathematical avoiding [evaluate,RawKernel.runGrid,FPValue.source,FPValue.outputValues,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.KernelRealResults
