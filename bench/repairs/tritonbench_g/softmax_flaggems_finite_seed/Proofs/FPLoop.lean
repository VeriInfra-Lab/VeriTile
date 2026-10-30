import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPReference

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPLoop
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
variable {α β : Type} [Inhabited α]

/-- Pure range graph: no registers, source statements, or interpreter. -/
def walk (f : Nat → β → β) (stop stride cur : Nat) (value : β) : Nat × β :=
  if stride=0 then (cur,value)
  else if cur<stop then walk f stop stride (cur+stride) (f cur value)
  else (cur,value)
termination_by stop-cur

theorem range_model (M : Algebra α) (idx : RegName) (body : List ComputeStmt)
    (f : Nat → β → β) (Inv : Nat → β → State α → Prop)
    (stop stride : Nat)
    (body_step : ∀ cur value s, cur<stop → Inv cur value s →
      ∃ t, run M body (s.setReg idx .nat [] (fun _ => cur))=some t ∧
        Inv (cur+stride) (f cur value) t)
    (cur : Nat) (value : β) (s : State α) (hs : Inv cur value s) :
    ∃ t, range M idx cur stop stride body s=some t ∧
      Inv (walk f stop stride cur value).1 (walk f stop stride cur value).2 t := by
  by_cases hz : stride=0
  · exact ⟨s,by simp [range,hz],by simpa [walk,hz] using hs⟩
  by_cases hc : cur<stop
  · obtain ⟨mid,he,hi⟩ := body_step cur value s hc hs
    obtain ⟨t,ht,hI⟩ := range_model M idx body f Inv stop stride body_step
      (cur+stride) (f cur value) mid hi
    refine ⟨t,?_,?_⟩
    · rw [range, if_neg hz, if_pos hc, he]
      exact ht
    · rw [walk, if_neg hz, if_pos hc]
      exact hI
  · exact ⟨s,by simp [range,hz,hc],by simpa [walk,hz,hc] using hs⟩
termination_by stop-cur

def branch (kernel : ComputeKernel) (index : Nat) : List ComputeStmt :=
  match kernel.surfaceBody[index]? with
  | some (.ifThenElse _ _ no) => no
  | _ => []

def rangeBody (body : List ComputeStmt) (index : Nat) : List ComputeStmt :=
  match body[index]? with
  | some (.forRange _ _ _ _ code) | some (.forRangeDyn _ _ _ _ code) => code
  | _ => []

@[simp] theorem masked_regs {ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (xs : List ι) (s : State α) :
    (xs.foldl (fun st i => if active i then st.write r (off i) (val i) else st) s).regs = s.regs := by
  rw [State.masked_scatter_eq_filter, State.scatter_regs]

theorem unmasked_mem {ι : Type} (xs : List ι) (r : ι → RegionName) (off : ι → Nat)
    (val : ι → Cell α) (s t : State α) (h : s.mem=t.mem) :
    (xs.foldl (fun st i => st.write (r i) (off i) (val i)) s).mem =
    (xs.foldl (fun st i => st.write (r i) (off i) (val i)) t).mem := by
  simpa using FPReference.writeFold_mem_congr xs r off (fun _ => True) val s t h

#axiomsClean range_model
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPLoop
