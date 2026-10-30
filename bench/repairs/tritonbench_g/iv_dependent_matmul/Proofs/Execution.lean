import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.Native
import bench.tritonbench_g.sgmv_expand_slice.Proofs.FPContract

/-! Functional execution of the finite source, in source order. Scalar algebra
is a parameter: FP instantiation preserves exact source Dot+Add or independent
DotAcc and both casts; ideal Real instantiation erases numerical rounding only.
The registers determining source pointer staging are Native.cacheAt. -/
namespace VeriTile.Bench.Repairs.IvDependentMatmul.Execution
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Kernels Whole
open TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 4000000

theorem sourceRange_eq {β : Type} (update : Nat → β → β) (cur stop stride : Nat) (initial : β) :
    sourceRange update cur stop stride initial=foldRange update cur stop stride initial := by
  conv_lhs => rw [sourceRange]
  conv_rhs => rw [foldRange]
  split_ifs with hz hk
  · rfl
  · exact sourceRange_eq update (cur+stride) stop stride (update cur initial)
  · rfl
termination_by stop-cur
decreasing_by omega

def fp (cfg : FP.Config) (D : Dot) (F : DotAcc) (r : Request) : Arithmetic Word r where
  zero := FP.Value.zero .fp32
  add := FP.add cfg
  dot := D.apply (some .fp32) .fp32 .fp32 .fp32 {} (batch := [])
  dotAcc := F.apply (some .fp32) .fp32 .fp32 .real {} (batch := [])
  narrow := castWord cfg .single .half
  widen := castWord cfg .half .single

def ideal (r : Request) : Arithmetic ℝ r where
  zero := 0
  add := (·+·)
  dot := fun A B q => ∑ j : Fin r.blockInner, A (q.1,j,PUnit.unit)*B (j,q.2.1,PUnit.unit)
  dotAcc := fun A B acc q => acc q+∑ j : Fin r.blockInner, A (q.1,j,PUnit.unit)*B (j,q.2.1,PUnit.unit)
  narrow := id
  widen := id

theorem decode_address (base off : Nat) : (base+4*off-base)/4=off := by omega

theorem a_operand [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (r : Request) (initial current : State Word) (a : Allocation r initial)
    (d : Index64.Addressable r initial a) (l : Index64.Launchable r)
    (pid k : Nat) (hp : pid<programs r) (hk : k<count r.parameters) :
    aLoad (fp cfg D F r) a.baseA a.baseB pid k current=
      aValues r.parameters (instanceState pid current) k := by
  have h := Native.correspondence r initial a d l
  funext i
  simp only [aLoad,(h.stage pid k).1,h.kMask pid k hp hk i.2.1,decide_eq_true_eq]
  by_cases hm : i.2.1.val<r.inner-k*r.blockInner
  · rw [if_pos hm,h.activeA pid k hp i hm,decode_address]
    simp only [aValues,Request.parameters,if_pos hm]
    rfl
  · simp only [if_neg hm,aValues,Request.parameters,fp]

theorem b_operand [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (r : Request) (initial current : State Word) (a : Allocation r initial)
    (d : Index64.Addressable r initial a) (l : Index64.Launchable r)
    (pid k : Nat) (hp : pid<programs r) (hk : k<count r.parameters) :
    bLoad (fp cfg D F r) a.baseA a.baseB pid k current=
      bValues r.parameters (instanceState pid current) k := by
  have h := Native.correspondence r initial a d l
  funext i
  simp only [bLoad,(h.stage pid k).2,h.kMask pid k hp hk i.1,decide_eq_true_eq]
  by_cases hm : i.1.val<r.inner-k*r.blockInner
  · rw [if_pos hm,h.activeB pid k hp i hm,decode_address]
    simp only [bValues,Request.parameters,if_pos hm]
    rfl
  · simp only [if_neg hm,bValues,Request.parameters,fp]

theorem step_projection [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (r : Request) (initial current : State Word) (a : Allocation r initial)
    (d : Index64.Addressable r initial a) (l : Index64.Launchable r)
    (pid k : Nat) (hp : pid<programs r) (hk : k<count r.parameters) (acc : Acc Word r) :
    step (fp cfg D F r) g a.baseA a.baseB pid current k acc=
      update cfg D F g r.parameters (instanceState pid current) k acc := by
  cases g <;> simp only [step]
  all_goals rw [a_operand cfg D F r initial current a d l pid k hp hk,
    b_operand cfg D F r initial current a d l pid k hp hk]
  all_goals rfl

private theorem range_projection [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (r : Request) (initial current : State Word) (a : Allocation r initial)
    (d : Index64.Addressable r initial a) (l : Index64.Launchable r)
    (pid cur : Nat) (hp : pid<programs r) (acc : Acc Word r) :
    foldRange (step (fp cfg D F r) g a.baseA a.baseB pid current) cur (count r.parameters) 1 acc=
      foldRange (update cfg D F g r.parameters (instanceState pid current)) cur (count r.parameters) 1 acc := by
  by_cases hk : cur<count r.parameters
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simp only [Nat.one_ne_zero,hk,↓reduceDIte]
    rw [step_projection cfg D F g r initial current a d l pid cur hp hk acc]
    exact range_projection cfg D F g r initial current a d l pid (cur+1) hp _
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simp only [Nat.one_ne_zero,hk,↓reduceDIte]
termination_by count r.parameters-cur
decreasing_by omega

theorem accumulated_projection [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (r : Request) (initial current : State Word) (a : Allocation r initial)
    (d : Index64.Addressable r initial a) (l : Index64.Launchable r)
    (pid : Nat) (hp : pid<programs r) :
    accumulated (fp cfg D F r) g a.baseA a.baseB pid current=
      FPValue.accumulated cfg D F g r.parameters (instanceState pid current) := by
  have h := Native.correspondence r initial a d l
  have hc := (Index64.counts_control r initial a d l pid hp).2.2
  unfold accumulated FPValue.accumulated
  rw [sourceRange_eq]
  rw [h.counter pid hp,Native.unsigned_word _ (by omega)]
  exact range_projection cfg D F g r initial current a d l pid 0 hp _

theorem program_projection [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (r : Request) (initial current : State Word) (a : Allocation r initial)
    (d : Index64.Addressable r initial a) (l : Index64.Launchable r)
    (pid : Nat) (hp : pid<programs r) :
    program (fp cfg D F r) g a.baseA a.baseB a.baseC pid current=
      written cfg r.parameters (instanceState pid current)
        (FPValue.accumulated cfg D F g r.parameters (instanceState pid current)) := by
  have h := Native.correspondence r initial a d l
  unfold program
  rw [accumulated_projection cfg D F g r initial current a d l pid hp]
  unfold written
  dsimp only
  apply congrArg (fun f : State Word → CIndex r → State Word =>
    (TileShape.allIndices [r.blockRows,r.blockColumns]).foldl f (instanceState pid current))
  funext t q
  rw [h.cMask pid hp q]
  have mask : active r.parameters (instanceState pid current) q=active r.parameters (instanceState pid initial) q := rfl
  rw [mask]
  by_cases hm : active r.parameters (instanceState pid initial) q=Bool.true
  · simp [hm]
    rw [h.activeC pid hp q hm,decode_address]
    rfl
  · simp [hm]

private theorem written_congr [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (r : Request) (s t : State Word) (pid : Nat) (hm : s.mem=t.mem) :
    (written cfg r.parameters (instanceState pid s)
      (FPValue.accumulated cfg D F g r.parameters (instanceState pid s))).mem=
    (written cfg r.parameters (instanceState pid t)
      (FPValue.accumulated cfg D F g r.parameters (instanceState pid t))).mem := by
  rw [accumulated_inputs cfg D F g r t s pid (by intro o; rw [hm]) (by intro o; rw [hm])]
  unfold written
  apply TritonBenchG.SgmvExpandSlice.FPValue.fold_memory_congr
  exact hm

theorem programs_projection [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (r : Request) (initial : State Word) (a : Allocation r initial)
    (d : Index64.Addressable r initial a) (l : Index64.Launchable r)
    (ids : List Nat) (valid : ∀ pid∈ids,pid<programs r) (s t : State Word) (hm : s.mem=t.mem) :
    ∃ final, Whole.runPrograms rest cfg D F g r ids s=some final ∧
      final.mem=(runPrograms (fp cfg D F r) g a.baseA a.baseB a.baseC ids t).mem := by
  induction ids generalizing s t with
  | nil => exact ⟨s,rfl,hm⟩
  | cons pid ids ih =>
    have hp := valid pid (by simp)
    obtain ⟨next,he,hw⟩ := Schedules.execution_memory rest cfg D F r.schedule g r.parameters (instanceState pid s)
    have hn : next.mem=(program (fp cfg D F r) g a.baseA a.baseB a.baseC pid t).mem := by
      rw [program_projection cfg D F g r initial t a d l pid hp]
      exact hw.trans (written_congr cfg D F g r s t pid hm)
    obtain ⟨final,hf,hfm⟩ := ih (by intro p h; exact valid p (by simp [h])) next _ hn
    exact ⟨final,by simp only [Whole.runPrograms,he,Option.bind,bind,hf],hfm⟩

theorem host_projection [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (r : Request) (initial : State Word) (a : Allocation r initial)
    (d : Index64.Addressable r initial a) (l : Index64.Launchable r) :
    ∃ final, Whole.host rest cfg D F g r initial=some (outputMetadata r,final) ∧
      final.mem=(host (fp cfg D F r) g a.baseA a.baseB a.baseC initial).2.mem ∧
      Result cfg D F g r initial (host (fp cfg D F r) g a.baseA a.baseB a.baseC initial).2 := by
  obtain ⟨final,he,hm⟩ := programs_projection rest cfg D F g r initial a d l
    (List.range (programs r)) (by intro pid hp; exact List.mem_range.mp hp) initial initial rfl
  obtain ⟨other,ho,hv⟩ := grid_correct rest cfg D F g r initial a.freshA a.freshB
  have eq : final=other := Option.some.inj (he.symm.trans ho)
  subst other
  refine ⟨final,by simp [Whole.host,he],hm,?_⟩
  simpa only [Result,Good,hm] using hv

#axiomsClean step_projection
#axiomsClean program_projection
#axiomsClean host_projection
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IvDependentMatmul.Execution
