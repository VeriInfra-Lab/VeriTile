import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.ReadableSource
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.IOForwardResults
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.IOSafety

namespace VeriTile.Bench.Repairs.DiagSsmTriton.Interface
open VeriTile.Triton
open scoped UKernelIO
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem backward_real_correct (p : BackwardLayout) (length : Nat)
    (sep : RegionCorrectness.BackwardSeparation p) :
    backwardIO Bool.false p length ⊨ᵤᵣ
      (fun _ _ v o k => backwardFormula (α := ℝ) id p.B length (backwardRealInputs p length v) o k.val) := by
  intro u v hp
  refine ⟨?_,?_⟩
  · intro bounds hb
    exact IOSafety.backward_trace p length sep u bounds (backward_bounds_real p length u v bounds hb)
  · obtain ⟨final,he,hx,hs,hl,hf⟩ := RegionCorrectness.backward p length sep u
    refine ⟨final,he,?_,?_⟩
    · intro o k ha
      change Fin 3 at o
      fin_cases o
      · change Fin (length*p.B) at k
        obtain ⟨t,j,rfl⟩ := pair_surjective k
        have hj : active p.batch p.dim p.B (u.pids 0) j.val := by
          simpa only [backwardIO,backwardOutLane,Fin.divNat,Fin.val_zero,Nat.zero_div,↓reduceIte,pair_mod] using ha
        have h := (hx t j hj).trans (backward_real_gx p length u v hp t j hj)
        change final.readMem p.gx (streamAddress p.batch p.dim p.B (u.pids 0) (pair t j).val*1+0)=_
        simpa only [streamAddress,pair_div,pair_mod,RealReference.timeOffset,RealReference.colOffset,column,Nat.mul_one,Nat.add_zero] using h
      · change Fin p.B at k
        have hj : active p.batch p.dim p.B (u.pids 0) k.val := ha
        have h := (hs k hj).trans (backward_real_carry p length u v hp k hj length (le_refl _))
        change final.readMem p.gs (column p.B (u.pids 0) k.val*1+0)=backwardFormula id p.B length (backwardRealInputs p length v) 1 k.val
        simpa only [Nat.mul_one,Nat.add_zero] using h
      · change Fin p.B at k
        have hj : active p.batch p.dim p.B (u.pids 0) k.val := ha
        have h := (hl k hj).trans (backward_real_gl p length u v hp k hj)
        change final.readMem p.gl (column p.B (u.pids 0) k.val*1+0)=backwardFormula id p.B length (backwardRealInputs p length v) 2 k.val
        simpa only [Nat.mul_one,Nat.add_zero] using h
    · intro r address hoc _
      apply hf r address
      · by_cases hr : r=p.gx
        · right
          intro t j hj
          have h := hoc (0 : Fin 3) (pair t j)
            (by change active p.batch p.dim p.B (u.pids 0) ((pair t j).val%p.B); simpa only [pair_mod] using hj)
          change r≠p.gx ∨ address≠streamAddress p.batch p.dim p.B (u.pids 0) (pair t j).val*1+0 at h
          simp only [Nat.mul_one,Nat.add_zero] at h
          rcases h with hn | hn
          · exact False.elim (hn hr)
          · simpa only [streamAddress,pair_div,pair_mod,RealReference.timeOffset,RealReference.colOffset,column,Nat.mul_one,Nat.add_zero] using hn
        · exact Or.inl hr
      · by_cases hr : r=p.gs
        · right
          intro j hj
          have h := hoc (1 : Fin 3) j hj
          change r≠p.gs ∨ address≠column p.B (u.pids 0) j.val*1+0 at h
          simp only [Nat.mul_one,Nat.add_zero] at h
          exact h.resolve_left (fun hn => hn hr)
        · exact Or.inl hr
      · by_cases hr : r=p.gl
        · right
          intro j hj
          have h := hoc (2 : Fin 3) j hj
          change r≠p.gl ∨ address≠column p.B (u.pids 0) j.val*1+0 at h
          simp only [Nat.mul_one,Nat.add_zero] at h
          exact h.resolve_left (fun hn => hn hr)
        · exact Or.inl hr

theorem backward_complex_correct (p : BackwardLayout) (length : Nat)
    (sep : ComplexBackwardRun.Separation p) :
    backwardIO Bool.true p length ⊨ᵤᵣ
      (fun _ _ v o k =>
        let z := backwardFormula (starRingEnd ℂ) p.B length (backwardComplexInputs p length v) o.divNat k.val
        if o.modNat.val=0 then z.re else z.im) := by
  intro u v hp
  refine ⟨?_,?_⟩
  · intro bounds hb
    have h := ComplexBackwardSafety.trace_safe p u length bounds
      (backward_bounds_complex p length u v bounds hb) sep.toReadSeparation
    have actual := (SourceComplexBackward.backward_trace p length u bounds).mpr h
    simpa only [backwardIO, backwardKernel, ↓reduceIte] using actual
  · obtain ⟨final,he,hx,hs,hl,hf⟩ := RegionCorrectness.complex_backward p length sep u
    refine ⟨final,he,?_,?_⟩
    · intro o k ha
      change Fin 6 at o
      fin_cases o
      · change Fin (length*p.B) at k
        obtain ⟨t,j,rfl⟩ := pair_surjective k
        have hj : active p.batch p.dim p.B (u.pids 0) j.val := by
          change active p.batch p.dim p.B (u.pids 0) ((pair t j).val%p.B) at ha
          simpa only [pair_mod] using ha
        have hm : ComplexBackwardLoop.mask p u ⟨j,PUnit.unit⟩=Bool.true := decide_eq_true hj
        have h := (hx t.val t.isLt ⟨j,PUnit.unit⟩ hm).trans (backward_complex_gx p length u v hp t j hj)
        have h' := congrArg Complex.re h
        change final.readMem p.gx (streamAddress p.batch p.dim p.B (u.pids 0) (pair t j).val*2+0)=_
        simpa only [streamAddress,pair_div,pair_mod,ComplexBackwardLoop.offset,ComplexForwardStep.offset,
          ComplexBackwardLoop.columns,ComplexForwardInit.columns,column,ComplexMemory.read,Nat.add_zero] using h'
      · change Fin (length*p.B) at k
        obtain ⟨t,j,rfl⟩ := pair_surjective k
        have hj : active p.batch p.dim p.B (u.pids 0) j.val := by
          change active p.batch p.dim p.B (u.pids 0) ((pair t j).val%p.B) at ha
          simpa only [pair_mod] using ha
        have hm : ComplexBackwardLoop.mask p u ⟨j,PUnit.unit⟩=Bool.true := decide_eq_true hj
        have h := (hx t.val t.isLt ⟨j,PUnit.unit⟩ hm).trans (backward_complex_gx p length u v hp t j hj)
        have h' := congrArg Complex.im h
        change final.readMem p.gx (streamAddress p.batch p.dim p.B (u.pids 0) (pair t j).val*2+1)=_
        simpa only [streamAddress,pair_div,pair_mod,ComplexBackwardLoop.offset,ComplexForwardStep.offset,
          ComplexBackwardLoop.columns,ComplexForwardInit.columns,column,ComplexMemory.read,Nat.add_zero] using h'
      · change Fin p.B at k
        have hj : active p.batch p.dim p.B (u.pids 0) k.val := ha
        have hm : ComplexBackwardLoop.mask p u ⟨k,PUnit.unit⟩=Bool.true := decide_eq_true hj
        have h := (hs ⟨k,PUnit.unit⟩ hm).trans (backward_complex_carry p length u v hp k hj length (le_refl _))
        have h' := congrArg Complex.re h
        change final.readMem p.gs (column p.B (u.pids 0) k.val*2+0)=_
        simpa only [streamAddress,pair_div,pair_mod,ComplexBackwardLoop.offset,ComplexForwardStep.offset,
          ComplexBackwardLoop.columns,ComplexForwardInit.columns,column,ComplexMemory.read,Nat.add_zero] using h'
      · change Fin p.B at k
        have hj : active p.batch p.dim p.B (u.pids 0) k.val := ha
        have hm : ComplexBackwardLoop.mask p u ⟨k,PUnit.unit⟩=Bool.true := decide_eq_true hj
        have h := (hs ⟨k,PUnit.unit⟩ hm).trans (backward_complex_carry p length u v hp k hj length (le_refl _))
        have h' := congrArg Complex.im h
        change final.readMem p.gs (column p.B (u.pids 0) k.val*2+1)=_
        simpa only [streamAddress,pair_div,pair_mod,ComplexBackwardLoop.offset,ComplexForwardStep.offset,
          ComplexBackwardLoop.columns,ComplexForwardInit.columns,column,ComplexMemory.read,Nat.add_zero] using h'
      · change Fin p.B at k
        have hj : active p.batch p.dim p.B (u.pids 0) k.val := ha
        have hm : ComplexBackwardLoop.mask p u ⟨k,PUnit.unit⟩=Bool.true := decide_eq_true hj
        have h := (hl ⟨k,PUnit.unit⟩ hm).trans (backward_complex_gl p length u v hp k hj)
        have h' := congrArg Complex.re h
        change final.readMem p.gl (column p.B (u.pids 0) k.val*2+0)=_
        simpa only [streamAddress,pair_div,pair_mod,ComplexBackwardLoop.offset,ComplexForwardStep.offset,
          ComplexBackwardLoop.columns,ComplexForwardInit.columns,column,ComplexMemory.read,Nat.add_zero] using h'
      · change Fin p.B at k
        have hj : active p.batch p.dim p.B (u.pids 0) k.val := ha
        have hm : ComplexBackwardLoop.mask p u ⟨k,PUnit.unit⟩=Bool.true := decide_eq_true hj
        have h := (hl ⟨k,PUnit.unit⟩ hm).trans (backward_complex_gl p length u v hp k hj)
        have h' := congrArg Complex.im h
        change final.readMem p.gl (column p.B (u.pids 0) k.val*2+1)=_
        simpa only [streamAddress,pair_div,pair_mod,ComplexBackwardLoop.offset,ComplexForwardStep.offset,
          ComplexBackwardLoop.columns,ComplexForwardInit.columns,column,ComplexMemory.read,Nat.add_zero] using h'
    · intro r address hoc _
      apply hf r address
      refine ⟨?_,?_,?_⟩
      · by_cases hr : r=p.gx
        · right
          intro t _ ht j hj
          have h0 := hoc (0 : Fin 6) (pair (⟨t,ht⟩ : Fin length) j.1)
            (by change active p.batch p.dim p.B (u.pids 0) ((pair (⟨t,ht⟩ : Fin length) j.1).val%p.B); simpa only [pair_mod] using of_decide_eq_true hj)
          have h1 := hoc (1 : Fin 6) (pair (⟨t,ht⟩ : Fin length) j.1)
            (by change active p.batch p.dim p.B (u.pids 0) ((pair (⟨t,ht⟩ : Fin length) j.1).val%p.B); simpa only [pair_mod] using of_decide_eq_true hj)
          constructor
          · have h' := h0
            change r≠p.gx ∨ address≠streamAddress p.batch p.dim p.B (u.pids 0) (pair (⟨t,ht⟩ : Fin length) j.1).val*2+0 at h'
            rcases h' with hn | hn
            · exact False.elim (hn hr)
            · simpa only [streamAddress,pair_div,pair_mod,ComplexBackwardLoop.offset,ComplexForwardStep.offset,
                ComplexBackwardLoop.columns,ComplexForwardInit.columns,column,Nat.add_zero,ne_comm] using hn
          · have h' := h1
            change r≠p.gx ∨ address≠streamAddress p.batch p.dim p.B (u.pids 0) (pair (⟨t,ht⟩ : Fin length) j.1).val*2+1 at h'
            rcases h' with hn | hn
            · exact False.elim (hn hr)
            · simpa only [streamAddress,pair_div,pair_mod,ComplexBackwardLoop.offset,ComplexForwardStep.offset,
                ComplexBackwardLoop.columns,ComplexForwardInit.columns,column,Nat.add_zero,ne_comm] using hn
        · exact Or.inl hr
      · by_cases hr : r=p.gs
        · right
          intro j hj
          have h0 := hoc (2 : Fin 6) j.1 (by change u.pids 0*p.B+j.1.val<p.batch*p.dim; exact of_decide_eq_true hj)
          have h1 := hoc (3 : Fin 6) j.1 (by change u.pids 0*p.B+j.1.val<p.batch*p.dim; exact of_decide_eq_true hj)
          constructor
          · have h' := h0
            change r≠p.gs ∨ address≠column p.B (u.pids 0) j.1.val*2+0 at h'
            rcases h' with hn | hn
            · exact False.elim (hn hr)
            · simpa only [streamAddress,pair_div,pair_mod,ComplexBackwardLoop.offset,ComplexForwardStep.offset,
                ComplexBackwardLoop.columns,ComplexForwardInit.columns,column,Nat.add_zero,ne_comm] using hn
          · have h' := h1
            change r≠p.gs ∨ address≠column p.B (u.pids 0) j.1.val*2+1 at h'
            rcases h' with hn | hn
            · exact False.elim (hn hr)
            · simpa only [streamAddress,pair_div,pair_mod,ComplexBackwardLoop.offset,ComplexForwardStep.offset,
                ComplexBackwardLoop.columns,ComplexForwardInit.columns,column,Nat.add_zero,ne_comm] using hn
        · exact Or.inl hr
      · by_cases hr : r=p.gl
        · right
          intro j hj
          have h0 := hoc (4 : Fin 6) j.1 (by change u.pids 0*p.B+j.1.val<p.batch*p.dim; exact of_decide_eq_true hj)
          have h1 := hoc (5 : Fin 6) j.1 (by change u.pids 0*p.B+j.1.val<p.batch*p.dim; exact of_decide_eq_true hj)
          constructor
          · have h' := h0
            change r≠p.gl ∨ address≠column p.B (u.pids 0) j.1.val*2+0 at h'
            rcases h' with hn | hn
            · exact False.elim (hn hr)
            · simpa only [streamAddress,pair_div,pair_mod,ComplexBackwardLoop.offset,ComplexForwardStep.offset,
                ComplexBackwardLoop.columns,ComplexForwardInit.columns,column,Nat.add_zero,ne_comm] using hn
          · have h' := h1
            change r≠p.gl ∨ address≠column p.B (u.pids 0) j.1.val*2+1 at h'
            rcases h' with hn | hn
            · exact False.elim (hn hr)
            · simpa only [streamAddress,pair_div,pair_mod,ComplexBackwardLoop.offset,ComplexForwardStep.offset,
                ComplexBackwardLoop.columns,ComplexForwardInit.columns,column,Nat.add_zero,ne_comm] using hn
        · exact Or.inl hr

#axiomsClean backward_real_correct
#axiomsClean backward_complex_correct
end
end VeriTile.Bench.Repairs.DiagSsmTriton.Interface
