import bench.tritonbench_g.iv_dependent_matmul.Proofs.Execution
import bench.tritonbench_g.iv_dependent_matmul.Proofs.Mathematical_Real

/-! The independent mathematical interpretation runs the same finite masks,
addresses, pointer-register schedules, program order and stores. Only arithmetic
is interpreted in ℝ. It places no exactness assumption on FP primitives. -/
namespace VeriTile.Bench.Repairs.IvDependentMatmul.Ideal
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Kernels Whole
open TritonBenchG.ParallelAttention.FPStages (foldRange)
open RealTarget (mathematical blockSum)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000

def inputA (r : Request) (s : State ℝ) (i k : Nat) : ℝ :=
  (s.mem r.A (i*max r.inner 1+k)).read .fp32

def inputB (r : Request) (s : State ℝ) (k j : Nat) : ℝ :=
  (s.mem r.B (k*max r.columns 1+j)).read .fp32

def Good (r : Request) (initial current : State ℝ) (i j : Nat) : Prop :=
  current.mem r.C (offset r i j)=Cell.mk .fp32
    (mathematical r.inner (inputA r initial) (inputB r initial) i j)

def Result (r : Request) (initial final : State ℝ) : Prop :=
  (∀ i<r.rows, ∀ j<r.columns, Good r initial final i j) ∧
  (∀ region o, ¬Footprint r region o → final.mem region o=initial.mem region o)

theorem loaded_block (r : Request) (layout : State Word) (a : Allocation r layout)
    (d : Index64.Addressable r layout a) (l : Index64.Launchable r)
    (s : State ℝ) (pid k : Nat) (hp : pid<programs r) (hk : k<count r.parameters)
    (q : CIndex r) (hq : active r.parameters (instanceState pid layout) q=Bool.true) :
    (∑ j : Fin r.blockInner,
      Execution.aLoad (Execution.ideal r) a.baseA a.baseB pid k s (q.1,j,PUnit.unit)*
      Execution.bLoad (Execution.ideal r) a.baseA a.baseB pid k s (j,q.2.1,PUnit.unit))=
    blockSum r.inner r.blockInner (inputA r s) (inputB r s)
      (row r.parameters (instanceState pid layout) q.1)
      (col r.parameters (instanceState pid layout) q.2.1) k := by
  have h := Native.correspondence r layout a d l
  have hq' : row r.parameters (instanceState pid layout) q.1<r.rows ∧
      col r.parameters (instanceState pid layout) q.2.1<r.columns := by
    simpa only [active,Bool.and_eq_true,decide_eq_true_eq,Request.parameters] using hq
  unfold blockSum
  apply Finset.sum_congr rfl
  intro j _
  simp only [Execution.aLoad,Execution.bLoad,(h.stage pid k).1,(h.stage pid k).2,
    h.kMask pid k hp hk j,decide_eq_true_eq]
  by_cases hm : j.val<r.inner-k*r.blockInner
  · rw [if_pos hm,if_pos hm,h.activeA pid k hp _ hm,h.activeB pid k hp _ hm,
      Execution.decode_address,Execution.decode_address,if_pos (by omega)]
    have ha : aOffset r.parameters (instanceState pid layout) k (q.1,j,PUnit.unit)=
        row r.parameters (instanceState pid layout) q.1*max r.inner 1+(k*r.blockInner+j.val) := by
      simp only [aOffset,aBase]
      change (row r.parameters (instanceState pid layout) q.1%r.rows)*max r.inner 1+j.val*1+k*r.blockInner*1=_
      rw [Nat.mod_eq_of_lt hq'.1]
      ring
    have hb : bOffset r.parameters (instanceState pid layout) k (j,q.2.1,PUnit.unit)=
        (k*r.blockInner+j.val)*max r.columns 1+col r.parameters (instanceState pid layout) q.2.1 := by
      simp only [bOffset,bBase]
      change j.val*max r.columns 1+(col r.parameters (instanceState pid layout) q.2.1%r.columns)*1+k*r.blockInner*max r.columns 1=_
      rw [Nat.mod_eq_of_lt hq'.2]
      ring
    exact congrArg₂ (fun x y : ℝ => x*y)
      (congrArg (fun o => (s.mem r.A o).read .fp32) ha)
      (congrArg (fun o => (s.mem r.B o).read .fp32) hb)
  · simp only [if_neg hm,Execution.ideal,zero_mul,if_neg (by omega : ¬k*r.blockInner+j.val<r.inner)]

theorem step_value (g : Graph) (r : Request) (layout : State Word) (a : Allocation r layout)
    (d : Index64.Addressable r layout a) (l : Index64.Launchable r)
    (s : State ℝ) (pid k : Nat) (hp : pid<programs r) (hk : k<count r.parameters)
    (acc : Execution.Acc ℝ r) (q : CIndex r)
    (hq : active r.parameters (instanceState pid layout) q=Bool.true) :
    Execution.step (Execution.ideal r) g a.baseA a.baseB pid s k acc q=
      acc q+blockSum r.inner r.blockInner (inputA r s) (inputB r s)
        (row r.parameters (instanceState pid layout) q.1)
        (col r.parameters (instanceState pid layout) q.2.1) k := by
  cases g <;> simp only [Execution.step,Execution.ideal]
  all_goals exact congrArg (acc q+·) (loaded_block r layout a d l s pid k hp hk q hq)

private theorem range_value (g : Graph) (r : Request) (layout : State Word) (a : Allocation r layout)
    (d : Index64.Addressable r layout a) (l : Index64.Launchable r)
    (s : State ℝ) (pid cur : Nat) (hp : pid<programs r) (hc : cur≤count r.parameters)
    (acc : Execution.Acc ℝ r) (q : CIndex r)
    (hq : active r.parameters (instanceState pid layout) q=Bool.true) :
    foldRange (Execution.step (Execution.ideal r) g a.baseA a.baseB pid s)
      cur (count r.parameters) 1 acc q=
    acc q+
      (∑ k∈Finset.range (count r.parameters),blockSum r.inner r.blockInner (inputA r s) (inputB r s)
        (row r.parameters (instanceState pid layout) q.1) (col r.parameters (instanceState pid layout) q.2.1) k)-
      (∑ k∈Finset.range cur,blockSum r.inner r.blockInner (inputA r s) (inputB r s)
        (row r.parameters (instanceState pid layout) q.1) (col r.parameters (instanceState pid layout) q.2.1) k) := by
  by_cases hs : cur<count r.parameters
  · rw [foldRange]
    simp only [Nat.one_ne_zero,hs,↓reduceDIte]
    rw [range_value g r layout a d l s pid (cur+1) hp (by omega) _ q hq,
      step_value g r layout a d l s pid cur hp hs acc q hq,Finset.sum_range_succ]
    ring
  · have he : cur=count r.parameters := by omega
    subst cur
    rw [foldRange]
    simp only [Nat.one_ne_zero,lt_self_iff_false,↓reduceDIte]
    ring
termination_by count r.parameters-cur
decreasing_by omega

theorem accumulated_value (g : Graph) (r : Request) (layout : State Word) (a : Allocation r layout)
    (d : Index64.Addressable r layout a) (l : Index64.Launchable r)
    (s : State ℝ) (pid : Nat) (hp : pid<programs r) (q : CIndex r)
    (hq : active r.parameters (instanceState pid layout) q=Bool.true) :
    Execution.accumulated (Execution.ideal r) g a.baseA a.baseB pid s q=
      mathematical r.inner (inputA r s) (inputB r s)
        (row r.parameters (instanceState pid layout) q.1) (col r.parameters (instanceState pid layout) q.2.1) := by
  have h := Native.correspondence r layout a d l
  have hc := (Index64.counts_control r layout a d l pid hp).2.2
  unfold Execution.accumulated
  rw [Execution.sourceRange_eq]
  rw [h.counter pid hp,Native.unsigned_word _ (by omega)]
  rw [range_value g r layout a d l s pid 0 hp (Nat.zero_le _) _ q hq]
  simp only [Execution.ideal,Finset.range_zero,Finset.sum_empty,sub_zero,zero_add]
  exact RealTarget.masked_blocks_matmul r.inner r.blockInner r.positiveInnerBlock _ _ _ _

/-- A natural-address scatter is used only after the finite address projection. -/
def written (g : Graph) (r : Request) (layout : State Word) (a : Allocation r layout)
    (pid : Nat) (s : State ℝ) : State ℝ :=
  (TileShape.allIndices [r.blockRows,r.blockColumns]).foldl
    (fun t q => if active r.parameters (instanceState pid layout) q then
      t.write r.C (outputOffset r.parameters (instanceState pid layout) q)
        (Cell.mk .fp32 (Execution.accumulated (Execution.ideal r) g a.baseA a.baseB pid s q)) else t)
    {s with pids:=fun ax => if ax=0 then pid else 0}

theorem program_projection (g : Graph) (r : Request) (layout : State Word) (a : Allocation r layout)
    (d : Index64.Addressable r layout a) (l : Index64.Launchable r)
    (s : State ℝ) (pid : Nat) (hp : pid<programs r) :
    Execution.program (Execution.ideal r) g a.baseA a.baseB a.baseC pid s=written g r layout a pid s := by
  have h := Native.correspondence r layout a d l
  unfold Execution.program written
  dsimp only
  apply congrArg (fun f : State ℝ → CIndex r → State ℝ =>
    (TileShape.allIndices [r.blockRows,r.blockColumns]).foldl f
      {s with pids:=fun ax => if ax=0 then pid else 0})
  funext t q
  rw [h.cMask pid hp q]
  by_cases hm : active r.parameters (instanceState pid layout) q=Bool.true
  · simp [hm]
    rw [h.activeC pid hp q hm,Execution.decode_address]
    rfl
  · simp [hm]

theorem program_post (g : Graph) (r : Request) (layout : State Word) (a : Allocation r layout)
    (d : Index64.Addressable r layout a) (l : Index64.Launchable r)
    (s : State ℝ) (pid : Nat) (hp : pid<programs r) :
    (∀ q : CIndex r, active r.parameters (instanceState pid layout) q=Bool.true →
      (Execution.program (Execution.ideal r) g a.baseA a.baseB a.baseC pid s).mem r.C
        (outputOffset r.parameters (instanceState pid layout) q)=
      Cell.mk .fp32 (mathematical r.inner (inputA r s) (inputB r s)
        (row r.parameters (instanceState pid layout) q.1) (col r.parameters (instanceState pid layout) q.2.1))) ∧
    (∀ region o, (region≠r.C ∨ ∀ q : CIndex r,
      active r.parameters (instanceState pid layout) q=Bool.true → o≠outputOffset r.parameters (instanceState pid layout) q) →
      (Execution.program (Execution.ideal r) g a.baseA a.baseB a.baseC pid s).mem region o=s.mem region o) := by
  rw [program_projection g r layout a d l s pid hp]
  constructor
  · intro q hq
    have hv := masked_readback_active (shape := [r.blockRows,r.blockColumns]) r.C
      (fun q : CIndex r => outputOffset r.parameters (instanceState pid layout) q)
      (fun q : CIndex r => Cell.mk .fp32 (Execution.accumulated (Execution.ideal r) g a.baseA a.baseB pid s q))
      (fun q : CIndex r => active r.parameters (instanceState pid layout) q=Bool.true)
      {s with pids:=fun ax => if ax=0 then pid else 0} q hq
      (fun j hj he => output_injective_active r.parameters (instanceState pid layout) rfl (le_max_left _ _) q j hq hj he)
    exact hv.trans (congrArg (fun v : ℝ => Cell.mk .fp32 v) (accumulated_value g r layout a d l s pid hp q hq))
  · intro region o ho
    exact State.masked_scatter_frame _ _ _ (fun q => active r.parameters (instanceState pid layout) q=Bool.true) _ region o
      (ho.imp_right fun hh q _ => hh q) _

/-- Lifts the generic scatter facts through evolving input-preserving memory.
Geometry, active uniqueness and grid coverage reuse the original helper proofs. -/
theorem program_correct (g : Graph) (r : Request) (layout : State Word) (a : Allocation r layout)
    (d : Index64.Addressable r layout a) (l : Index64.Launchable r)
    (initial current : State ℝ) (pid : Nat) (hp : pid<programs r)
    (hA : ∀ o,current.mem r.A o=initial.mem r.A o)
    (hB : ∀ o,current.mem r.B o=initial.mem r.B o) :
    let next := Execution.program (Execution.ideal r) g a.baseA a.baseB a.baseC pid current
    (∀ o,next.mem r.A o=initial.mem r.A o) ∧
    (∀ o,next.mem r.B o=initial.mem r.B o) ∧
    (∀ region o,¬Footprint r region o → next.mem region o=current.mem region o) ∧
    (∀ i<r.rows,∀ j<r.columns,Good r initial current i j → Good r initial next i j) ∧
    (∀ i<r.rows,∀ j<r.columns,Covered r layout [pid] i j → Good r initial next i j) := by
  dsimp only
  obtain ⟨hv,hf⟩ := program_post g r layout a d l current pid hp
  have ha : inputA r current=inputA r initial := by funext i k; unfold inputA; rw [hA]
  have hb : inputB r current=inputB r initial := by funext k j; unfold inputB; rw [hB]
  have value (q : CIndex r) (hq : active r.parameters (instanceState pid layout) q=Bool.true) :
      Good r initial (Execution.program (Execution.ideal r) g a.baseA a.baseB a.baseC pid current)
        (row r.parameters (instanceState pid layout) q.1) (col r.parameters (instanceState pid layout) q.2.1) := by
    have h := hv q hq
    rw [source_offset,ha,hb] at h
    exact h
  refine ⟨?_,?_,?_,?_,?_⟩
  · intro o
    exact (hf r.A o (Or.inl (Ne.symm a.freshA))).trans (hA o)
  · intro o
    exact (hf r.B o (Or.inl (Ne.symm a.freshB))).trans (hB o)
  · intro region o hn
    apply hf
    by_cases hr : region=r.C
    · right
      intro q hq heq
      apply hn
      have hq' : row r.parameters (instanceState pid layout) q.1<r.rows ∧
          col r.parameters (instanceState pid layout) q.2.1<r.columns := by
        simpa only [active,Bool.and_eq_true,decide_eq_true_eq,Request.parameters] using hq
      refine ⟨hr,?_⟩
      rw [heq,source_offset]
      exact output_bound r _ _ hq'.1 hq'.2
    · exact Or.inl hr
  · intro i hi j hj hg
    by_cases hw : ∃ q : CIndex r,active r.parameters (instanceState pid layout) q=Bool.true ∧
        offset r i j=outputOffset r.parameters (instanceState pid layout) q
    · obtain ⟨q,hq,hw⟩ := hw
      have hq' : row r.parameters (instanceState pid layout) q.1<r.rows ∧
          col r.parameters (instanceState pid layout) q.2.1<r.columns := by
        simpa only [active,Bool.and_eq_true,decide_eq_true_eq,Request.parameters] using hq
      rw [source_offset] at hw
      have hh := offset_injective r hj hq'.2 hw
      simpa only [hh.1,hh.2] using value q hq
    · exact (hf r.C (offset r i j) (Or.inr (by intro q hq heq; exact hw ⟨q,hq,heq⟩))).trans hg
  · intro i hi j hj hcov
    obtain ⟨p,hp,q,hrow,hcol⟩ := hcov
    have hp' : p=pid := by simpa using hp
    subst p
    have hq : active r.parameters (instanceState pid layout) q=Bool.true := by
      simp only [active,Bool.and_eq_true,decide_eq_true_eq]
      change row r.parameters (instanceState pid layout) q.1<r.rows ∧ col r.parameters (instanceState pid layout) q.2.1<r.columns
      rw [hrow,hcol]
      exact ⟨hi,hj⟩
    simpa only [hrow,hcol] using value q hq

theorem programs_correct (g : Graph) (r : Request) (layout : State Word) (a : Allocation r layout)
    (d : Index64.Addressable r layout a) (l : Index64.Launchable r)
    (initial current : State ℝ) (ids : List Nat) (valid : ∀ pid∈ids,pid<programs r)
    (hA : ∀ o,current.mem r.A o=initial.mem r.A o)
    (hB : ∀ o,current.mem r.B o=initial.mem r.B o) :
    let final := Execution.runPrograms (Execution.ideal r) g a.baseA a.baseB a.baseC ids current
    (∀ o,final.mem r.A o=initial.mem r.A o) ∧
    (∀ o,final.mem r.B o=initial.mem r.B o) ∧
    (∀ region o,¬Footprint r region o → final.mem region o=current.mem region o) ∧
    (∀ i<r.rows,∀ j<r.columns,Good r initial current i j → Good r initial final i j) ∧
    (∀ i<r.rows,∀ j<r.columns,Covered r layout ids i j → Good r initial final i j) := by
  induction ids generalizing current with
  | nil =>
    refine ⟨hA,hB,fun _ _ _ => rfl,fun _ _ _ _ h => h,?_⟩
    intro i hi j hj hcov
    obtain ⟨p,hp,_⟩ := hcov
    simp at hp
  | cons pid ids ih =>
    have hp := valid pid (by simp)
    obtain ⟨ha,hb,hframe,hkeep,hvalue⟩ := program_correct g r layout a d l initial current pid hp hA hB
    obtain ⟨ha',hb',hframe',hkeep',hvalue'⟩ := ih
      (Execution.program (Execution.ideal r) g a.baseA a.baseB a.baseC pid current)
      (by intro p h; exact valid p (by simp [h])) ha hb
    refine ⟨ha',hb',?_,?_,?_⟩
    · intro region o hn
      exact (hframe' region o hn).trans (hframe region o hn)
    · intro i hi j hj hg
      exact hkeep' i hi j hj (hkeep i hi j hj hg)
    · intro i hi j hj hcov
      obtain ⟨p,hp,q,hrow,hcol⟩ := hcov
      rcases List.mem_cons.mp hp with hp | hp
      · subst p
        exact hkeep' i hi j hj (hvalue i hi j hj ⟨pid,by simp,q,hrow,hcol⟩)
      · exact hvalue' i hi j hj ⟨p,hp,q,hrow,hcol⟩

theorem host_correct (g : Graph) (r : Request) (layout : State Word) (a : Allocation r layout)
    (d : Index64.Addressable r layout a) (l : Index64.Launchable r) (initial : State ℝ) :
    (Execution.host (Execution.ideal r) g a.baseA a.baseB a.baseC initial).1=outputMetadata r ∧
    Result r initial (Execution.host (Execution.ideal r) g a.baseA a.baseB a.baseC initial).2 := by
  obtain ⟨_,_,hf,_,hv⟩ := programs_correct g r layout a d l initial initial
    (List.range (programs r)) (by intro pid hp; exact List.mem_range.mp hp) (fun _ => rfl) (fun _ => rfl)
  exact ⟨rfl,fun i hi j hj => hv i hi j hj (cell_covered r layout i j hi hj),hf⟩

#axiomsClean loaded_block
#axiomsClean program_post
#axiomsClean host_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IvDependentMatmul.Ideal
