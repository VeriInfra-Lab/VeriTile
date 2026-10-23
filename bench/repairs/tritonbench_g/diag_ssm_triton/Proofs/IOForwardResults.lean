import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.IOBounds
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.IOComplexValues
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.IOForwardSafety

namespace VeriTile.Bench.Repairs.DiagSsmTriton.Interface
open VeriTile.Triton
open scoped UKernelIO
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

theorem pair_surjective {m n : Nat} (k : Fin (m*n)) :
    ∃ t : Fin m, ∃ j : Fin n, pair t j=k := by
  refine ⟨k.divNat,k.modNat,Fin.ext ?_⟩
  change k.val/n*n+k.val%n=k.val
  simpa only [Nat.mul_comm] using Nat.div_add_mod k.val n

theorem forward_real_correct (p : ForwardLayout) (length : Nat) (sep : p.x≠p.y) :
    forwardIO Bool.false p length ⊨ᵤᵣ
      (fun _ _ v _ k => forwardFormula (α := ℝ) p.B (forwardRealInputs p length v) k.val) := by
  intro u v hp
  refine ⟨?_,?_⟩
  · intro bounds hb
    exact IOSafety.forward_trace p length sep u bounds (forward_bounds_real p length u v bounds hb)
  · obtain ⟨final,he,ho,hf⟩ := RegionCorrectness.forward p length sep u
    refine ⟨final,he,?_,?_⟩
    · intro o k ha
      change Fin 1 at o
      fin_cases o
      change Fin (length*p.B) at k
      obtain ⟨t,j,rfl⟩ := pair_surjective k
      have hj : active p.batch p.dim p.B (u.pids 0) j.val := by
        simpa only [forwardIO,pair_mod] using ha
      have h := (ho t j hj).trans (forward_real_value p length u v hp t j hj)
      change final.readMem p.y (streamAddress p.batch p.dim p.B (u.pids 0) (pair t j).val*1+0)=_
      simpa only [streamAddress,pair_div,pair_mod,RealReference.timeOffset,RealReference.colOffset,column,Nat.mul_one,Nat.add_zero] using h
    · intro r address hoc _
      apply hf r address
      by_cases hr : r=p.y
      · right
        intro t j hj
        have h := hoc (0 : Fin 1) (pair t j) (by simpa only [forwardIO,pair_mod] using hj)
        change r≠p.y ∨ address≠streamAddress p.batch p.dim p.B (u.pids 0) (pair t j).val*1+0 at h
        simp only [Nat.mul_one,Nat.add_zero] at h
        rcases h with hn | hn
        · exact False.elim (hn hr)
        · simpa only [streamAddress,pair_div,pair_mod,RealReference.timeOffset,RealReference.colOffset,column,Nat.mul_one,Nat.add_zero] using hn
      · exact Or.inl hr

theorem forward_complex_correct (p : ForwardLayout) (length : Nat) :
    forwardIO Bool.true p length ⊨ᵤᵣ
      (fun _ _ v o k => if o.val=0 then (forwardFormula p.B (forwardComplexInputs p length v) k.val).re
        else (forwardFormula p.B (forwardComplexInputs p length v) k.val).im) := by
  intro u v hp
  refine ⟨?_,?_⟩
  · intro bounds hb
    have h := ComplexForwardSafety.trace_safe p u length bounds (forward_bounds_complex p length u v bounds hb)
    simpa only [forwardIO,forwardKernel,↓reduceIte,Kernel.TraceSafe,Kernels.complexForwardTyped,
      Erasure.complexForward_body] using h
  · obtain ⟨final,he,ho,hf⟩ := RegionCorrectness.complex_forward p length u
    refine ⟨final,he,?_,?_⟩
    · intro o k ha
      change Fin (length*p.B) at k
      obtain ⟨t,j,rfl⟩ := pair_surjective k
      have hj : active p.batch p.dim p.B (u.pids 0) j.val := by
        simpa only [forwardIO,pair_mod] using ha
      have hm : ComplexForwardLoop.mask p u ⟨j,PUnit.unit⟩=Bool.true := decide_eq_true hj
      have h := (ho t.val t.isLt ⟨j,PUnit.unit⟩ hm).trans (forward_complex_value p length u v hp t j hj)
      change Fin 2 at o
      fin_cases o
      · have h' := congrArg Complex.re h
        change final.readMem p.y (streamAddress p.batch p.dim p.B (u.pids 0) (pair t j).val*2+0)=_
        simpa only [streamAddress,pair_div,pair_mod,ComplexForwardLoop.offset,ComplexForwardStep.offset,
          ComplexForwardLoop.columns,ComplexForwardInit.columns,column,ComplexMemory.read,Nat.add_zero] using h'
      · have h' := congrArg Complex.im h
        change final.readMem p.y (streamAddress p.batch p.dim p.B (u.pids 0) (pair t j).val*2+1)=_
        simpa only [streamAddress,pair_div,pair_mod,ComplexForwardLoop.offset,ComplexForwardStep.offset,
          ComplexForwardLoop.columns,ComplexForwardInit.columns,column,ComplexMemory.read] using h'
    · intro r address hoc _
      apply hf r address
      by_cases hr : r=p.y
      · right
        intro t ht j hj
        have h c := hoc (c : Fin 2) (pair (⟨t,ht⟩ : Fin length) j.1)
          (by simpa only [forwardIO,pair_mod] using of_decide_eq_true hj)
        constructor
        · have h0 := h 0
          change r≠p.y ∨ address≠streamAddress p.batch p.dim p.B (u.pids 0) (pair (⟨t,ht⟩ : Fin length) j.1).val*2+0 at h0
          rcases h0 with hn | hn
          · exact False.elim (hn hr)
          · simpa only [streamAddress,pair_div,pair_mod,ComplexForwardLoop.offset,ComplexForwardStep.offset,
              ComplexForwardLoop.columns,ComplexForwardInit.columns,column,Nat.add_zero,ne_comm] using hn
        · have h1 := h 1
          change r≠p.y ∨ address≠streamAddress p.batch p.dim p.B (u.pids 0) (pair (⟨t,ht⟩ : Fin length) j.1).val*2+1 at h1
          rcases h1 with hn | hn
          · exact False.elim (hn hr)
          · simpa only [streamAddress,pair_div,pair_mod,ComplexForwardLoop.offset,ComplexForwardStep.offset,
              ComplexForwardLoop.columns,ComplexForwardInit.columns,column,ne_comm] using hn
      · exact Or.inl hr

#axiomsClean forward_real_correct
#axiomsClean forward_complex_correct
end
end VeriTile.Bench.Repairs.DiagSsmTriton.Interface
