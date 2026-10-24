import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.FPStep

namespace VeriTile.Bench.Repairs.ReversedCumsum.FPExecution
open VeriTile.Triton FP.Structural FPProgram FPValues IndexProgram
noncomputable section
set_option maxHeartbeats 4800000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Descending carry defined only from the original logical input. -/
def carry (M : Algebra α) (h : Call) (c : Config) (f : Nat) (input : Nat → Nat → α) (q : Nat) : Values α .real [BS] :=
  if q<Geometry.chunks h c then updated M h c f input q (carry M h c f input (q+1))
  else fun _ => M.literal (some .fp32) .real 0
termination_by Geometry.chunks h c-q

theorem carry_step (M : Algebra α) (h : Call) (c : Config) (f : Nat) (input : Nat → Nat → α) (q : Nat)
    (hq : q<Geometry.chunks h c) : updated M h c f input q (carry M h c f input (q+1))=carry M h c f input q := by
  symm
  rw [carry,if_pos hq]

def expected (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config) (positive : 0<c.BT)
    (f : Nat) (input : Nat → Nat → α) (t : Nat) (j : Fin BS) : α :=
  output M dot h c f input (t/c.BT) (carry M h c f input (t/c.BT+1))
    (⟨t%c.BT,Nat.mod_lt _ positive⟩,j,PUnit.unit)

structure Holds (M : Algebra α) (dot : DotBackend α) (initial : State α) (h : Call) (c : Config)
    (row f : Nat) (input : Nat → Nat → α) (remaining : Nat) (u : State α) : Prop where
  count_le : remaining≤Geometry.chunks h c
  head : u.regs .nat [] "i_bh"=some (fun _ => row)
  feature : u.regs .nat [] "i_s"=some (fun _ => f)
  triangular : u.regs .real [c.BT,c.BT] "m_s"=some (FPValues.triangular M c)
  carryReg : u.regs .real [BS] "b_z"=some (carry M h c f input remaining)
  reads : ∀ t s, t<h.T → s<h.S → toFloat h.inputDType
    ((u.mem h.input (Geometry.address h row t s)).read h.inputDType.toTileDType)=input t s
  output : ∀ q, remaining≤q → q<Geometry.chunks h c → ∀ i : Fin c.BT, ∀ j : Fin BS,
    q*c.BT+i.val<h.T → f*BS+j.val<h.S →
      u.mem h.output (Geometry.address h row (q*c.BT+i.val) (f*BS+j.val))=
        FPMemory.stored M h (FPValues.output M dot h c f input q (carry M h c f input (q+1)) (i,j,PUnit.unit))
  frame : FPMemory.Frame h row f initial u

theorem iteration (M : Algebra α) (dot : DotBackend α) (initial : State α)
    (h : Call) (distinct : h.input≠h.output) (c : Config) (positive : 0<c.BT)
    (row : Geometry.Row h) (f : Geometry.FeatureBlock h) (input : Nat → Nat → α)
    (j : Int) (u : State α) (hj : j<1) (hjreg : u.regs .int [] "__rev_i_t"=some (fun _ => j))
    (inv : Holds M dot initial h c row.val f.val input ((-j).toNat+1) u) :
    ∃ v, run (machine M dot) (loopBody h c) u=some v ∧ Holds M dot initial h c row.val f.val input (-j).toNat v := by
  let q := (-j).toNat
  let acc := carry M h c f.val input (q+1)
  let pre := FPStep.beforeStore M dot h c row.val q f.val input acc u
  let values := FPValues.output M dot h c f.val input q acc
  let v := FPStep.result M dot h c row.val q f.val input acc u
  have inside : q<Geometry.chunks h c := by have := inv.count_le; omega
  have start := Geometry.chunk_start h c positive q inside
  have next : updated M h c f.val input q acc=carry M h c f.val input q := carry_step M h c f.val input q inside
  have he := FPStep.executes M dot h c row.val f input acc j u hj start hjreg inv.feature inv.head inv.triangular inv.carryReg inv.reads
  refine ⟨v,he,Nat.le_of_lt inside,?_,?_,?_,?_,?_,?_,?_⟩
  · simp [v,FPStep.result,FPMemory.scatter,FPStep.beforeStore,FPStep.indexed,State.masked_scatter_eq_filter,inv.head]
  · simp [v,FPStep.result,FPMemory.scatter,FPStep.beforeStore,FPStep.indexed,State.masked_scatter_eq_filter,inv.feature]
  · simp [v,FPStep.result,FPMemory.scatter,FPStep.beforeStore,FPStep.indexed,State.masked_scatter_eq_filter,inv.triangular]
  · simp [v,FPStep.result,next,q]
  · intro t s ht hs
    change toFloat h.inputDType (((FPMemory.scatter M h c row.val q f.val values pre).mem h.input _).read _)=_
    rw [FPMemory.other_region M h c row.val q f.val values pre h.input distinct]
    simpa [pre,FPStep.beforeStore,FPStep.indexed] using inv.reads t s ht hs
  · intro k lower ki i e hi hs
    change (FPMemory.scatter M h c row.val q f.val values pre).mem h.output _=_
    by_cases same : k=q
    · subst k
      exact FPMemory.cell M h c row q f start values pre i e hi hs
    · have later : (q+1)*c.BT≤k*c.BT+i.val := by
        change q≤k at lower
        exact (Nat.mul_le_mul_right c.BT (by omega)).trans (Nat.le_add_right _ _)
      rw [FPMemory.preserves_later M h c row q f start values pre _ _ hi hs later]
      exact inv.output k (by change q≤k at lower; omega) ki i e hi hs
  · have frame := FPMemory.frame M h c row.val q f start values pre
    intro r off hn
    exact (inv.frame r off hn).trans (frame r off hn)

def initial (M : Algebra α) (c : Config) (u : State α) : State α :=
  let u1 := u.setReg "i_s" .nat [] (fun _ => u.pids 0)
  let u2 := u1.setReg "i_bh" .nat [] (fun _ => u.pids 1)
  let u3 := u2.setReg "o_i" .nat [c.BT] (FPIndices.lanes c.BT)
  let u4 := u3.setReg "m_s" .real [c.BT,c.BT] (triangular M c)
  u4.setReg "b_z" .real [BS] (fun _ => M.literal (some .fp32) .real 0)

theorem init_executes (M : Algebra α) (dot : DotBackend α) (c : Config) (u : State α) :
    run (machine M dot) (initCode c) u=some (initial M c u) := by
  unfold initCode
  rw [ReversedCumsumScalar.FPStep.assign_alg _ _ _ _ _ _ _ _ (show FP.Structural.evalOp (machine M dot) none (.programId 0) u=some (fun _ => u.pids 0) from rfl)]
  rw [ReversedCumsumScalar.FPStep.assign_alg _ _ _ _ _ _ _ _ (show FP.Structural.evalOp (machine M dot) none (.programId 1) (u.setReg "i_s" .nat [] (fun _ => u.pids 0))=some (fun _ => u.pids 1) from rfl)]
  rw [ReversedCumsumScalar.FPStep.assign_alg _ _ _ _ _ _ _ _ (show FP.Structural.evalOp (machine M dot) none (.arange c.BT) _=some (FPIndices.lanes c.BT) from rfl)]
  rw [ReversedCumsumScalar.FPStep.assign_alg _ _ _ _ _ _ _ _ (triangular_eval M dot c _ (by simp))]
  simp [run,step,FP.Structural.evalExpr,evalComputeOp,FP.Structural.evalOp,
    machine,Algebra.withDefaultPrecision,resolvePrecision,initial]
  rfl

def input (h : Call) (u : State α) (row t s : Nat) : α :=
  toFloat h.inputDType ((u.mem h.input (Geometry.address h row t s)).read h.inputDType.toTileDType)

theorem invariant (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config) (u : State α) :
    Holds M dot u h c (u.pids 1) (u.pids 0) (input h u (u.pids 1)) (Geometry.chunks h c) (initial M c u) := by
  refine ⟨le_rfl,?_,?_,?_,?_,by intros; rfl,?_,fun _ _ _ => rfl⟩
  · simp [initial]
  · simp [initial]
  · simp [initial]
  · rw [carry]
    simp [initial]
  · intro q lower inside
    omega

theorem mirror_invariant (M : Algebra α) (dot : DotBackend α) (initial : State α)
    (h : Call) (c : Config) (row f : Nat) (input : Nat → Nat → α) (q : Nat) (u : State α)
    (inv : Holds M dot initial h c row f input q u) (j : Int) :
    Holds M dot initial h c row f input q (u.setReg "__rev_i_t" .int [] (fun _ => j)) := by
  refine ⟨inv.count_le,?_,?_,?_,?_,inv.reads,inv.output,inv.frame⟩
  · simpa using inv.head
  · simpa using inv.feature
  · simpa using inv.triangular
  · simpa using inv.carryReg

theorem loop_executes (M : Algebra α) (dot : DotBackend α) (initial : State α)
    (h : Call) (distinct : h.input≠h.output) (c : Config) (positive : 0<c.BT)
    (row : Geometry.Row h) (f : Geometry.FeatureBlock h) (input : Nat → Nat → α)
    (j : Int) (u : State α) (inv : Holds M dot initial h c row.val f.val input (1-j).toNat u) :
    ∃ v, rangeInt (machine M dot) "__rev_i_t" j 1 1 (loopBody h c) u=some v ∧
      Holds M dot initial h c row.val f.val input 0 v := by
  by_cases hj : j<1
  · have hc : (1-j).toNat=(-j).toNat+1 := by omega
    have hs : (1-(j+1)).toNat=(-j).toNat := by omega
    have hm := mirror_invariant M dot initial h c row.val f.val input _ u inv j
    rw [hc] at hm
    obtain ⟨v,he,hv⟩ := iteration M dot initial h distinct c positive row f input j
      (u.setReg "__rev_i_t" .int [] (fun _ => j)) hj (by simp) hm
    obtain ⟨w,hw,hf⟩ := loop_executes M dot initial h distinct c positive row f input (j+1) v (by simpa only [hs] using hv)
    refine ⟨w,?_,hf⟩
    rw [rangeInt]
    simp only [Nat.one_ne_zero,if_false,hj,if_true,he,bind,Option.bind_some]
    exact hw
  · refine ⟨u,?_,?_⟩
    · rw [rangeInt]
      simp only [Nat.one_ne_zero,if_false,hj]
    · have he : (1-j).toNat=0 := by omega
      simpa only [he] using inv
termination_by (1-j).toNat

theorem statement_eval (M : Algebra α) (h : Call) (c : Config) (u : State α) :
    step M (statement h c) u=rangeInt M "__rev_i_t" (1-(Geometry.chunks h c:Int)) 1 1 (loopBody h c) u := by
  simp [statement,reverseStart,stopOp,step,FP.Structural.evalOp,numeric,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Geometry.chunks,ReversedCumsumScalar.Geometry.chunks,Call.timeView]

theorem source_correct (M : Algebra α) (dot : DotBackend α) (h : Call) (distinct : h.input≠h.output)
    (c : Config) (positive : 0<c.BT) (row : Geometry.Row h) (f : Geometry.FeatureBlock h)
    (u : State α) (pid0 : u.pids 0=f.val) (pid1 : u.pids 1=row.val) :
    ∃ v, FP.Structural.exec (machine M dot) (Kernels.source h c) u=some v ∧
      (∀ t, ∀ j : Fin BS, t<h.T → f.val*BS+j.val<h.S →
        v.mem h.output (Geometry.address h row.val t (f.val*BS+j.val))=
          FPMemory.stored M h (expected M dot h c positive f.val (input h u row.val) t j)) ∧
      FPMemory.Frame h row.val f.val u v := by
  have hi : (1-(1-(Geometry.chunks h c:Int))).toNat=Geometry.chunks h c := by omega
  obtain ⟨v,he,inv⟩ := loop_executes M dot u h distinct c positive row f _
    (1-(Geometry.chunks h c:Int)) (initial M c u) (by simpa only [hi,pid0,pid1] using invariant M dot h c u)
  refine ⟨v,?_,?_,inv.frame⟩
  · change run (machine M dot) (Kernels.source h c).surfaceBody u=some v
    rw [source_body,body,run_append,init_executes]
    simp only [Option.bind_some,run,statement_eval,he]
    rfl
  · intro t j ht hs
    have qi : t/c.BT<Geometry.chunks h c := (Nat.div_lt_iff_lt_mul positive).mpr
      (ht.trans_le (ReversedCumsumScalar.Geometry.chunks_cover h.timeView c positive))
    let i : Fin c.BT := ⟨t%c.BT,Nat.mod_lt _ positive⟩
    have eq : t/c.BT*c.BT+i.val=t := by simpa [i,Nat.mul_comm] using Nat.div_add_mod t c.BT
    have out := inv.output (t/c.BT) (Nat.zero_le _) qi i j (by simpa [eq] using ht) hs
    simpa only [eq] using out

#print axioms source_correct
end
end VeriTile.Bench.Repairs.ReversedCumsum.FPExecution
