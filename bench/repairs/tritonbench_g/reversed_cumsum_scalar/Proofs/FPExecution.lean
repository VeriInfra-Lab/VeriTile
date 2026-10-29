import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.FPStep

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.FPExecution
open VeriTile.Triton FP.Structural FPProgram FPValues
noncomputable section
set_option maxHeartbeats 4800000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Independent logical-input recurrence. No register or execution result is used. -/
def carry (M : Algebra α) (h : Call) (c : Config) (input : Nat → α) (q : Nat) : α :=
  if q<Geometry.chunks h c then updated M h c input q (carry M h c input (q+1))
  else M.literal (some .fp32) .real 0
termination_by Geometry.chunks h c-q

theorem carry_step (M : Algebra α) (h : Call) (c : Config) (input : Nat → α) (q : Nat)
    (hq : q<Geometry.chunks h c) :
    updated M h c input q (carry M h c input (q+1))=carry M h c input q := by
  symm
  rw [carry,if_pos hq]

def expected (M : Algebra α) (scan : ScanBackend α) (h : Call) (c : Config) (positive : 0<c.BT)
    (input : Nat → α) (time : Nat) : α :=
  output M scan h c input (time/c.BT) (carry M h c input (time/c.BT))
    (⟨time%c.BT,Nat.mod_lt _ positive⟩,PUnit.unit)

structure Holds (M : Algebra α) (scan : ScanBackend α) (initial : State α) (h : Call) (c : Config)
    (row : Nat) (input : Nat → α) (remaining : Nat) (u : State α) : Prop where
  count_le : remaining≤Geometry.chunks h c
  head : u.regs .nat [] "i_bh"=some (fun _ => row)
  carryReg : u.regs .real [] "b_z"=some (fun _ => carry M h c input remaining)
  reads : ∀ t, t<h.T → toFloat h.inputDType
    ((u.mem h.input (h.inputStrides.address h.H row t)).read h.inputDType.toTileDType)=input t
  output : ∀ q, remaining≤q → q<Geometry.chunks h c → ∀ i : Fin c.BT, q*c.BT+i.val<h.T →
    u.mem h.output (h.outputStrides.address h.H row (q*c.BT+i.val))=
      FPMemory.stored M h (FPValues.output M scan h c input q (carry M h c input q) (i,PUnit.unit))
  frame : FPMemory.Frame h row initial u

theorem iteration (M : Algebra α) (scan : ScanBackend α) (initial : State α)
    (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (positive : 0<c.BT)
    (row : Nat) (hr : row<h.B*h.H) (input : Nat → α) (j : Int) (u : State α)
    (hj : j<1) (hjreg : u.regs .int [] "__rev_i_t"=some (fun _ => j))
    (inv : Holds M scan initial h c row input ((-j).toNat+1) u) :
    ∃ v, run (machine M scan) (loopBody h c) u=some v ∧ Holds M scan initial h c row input (-j).toNat v := by
  let q := (-j).toNat
  let acc := carry M h c input (q+1)
  let pre := FPStep.beforeStore M scan h c row q input acc u
  let values := FPValues.output M scan h c input q (updated M h c input q acc)
  let v := FPStep.result M scan h c row q input acc u
  have inside : q<Geometry.chunks h c := by have := inv.count_le; omega
  have start := Geometry.chunk_start h c positive q inside
  have next : updated M h c input q acc=carry M h c input q := carry_step M h c input q inside
  have he := FPStep.executes M scan h c row input acc j u hj start hjreg inv.head inv.carryReg inv.reads
  refine ⟨v,he,Nat.le_of_lt inside,?_,?_,?_,?_,?_⟩
  · simp [v,FPStep.result,FPMemory.scatter,FPStep.beforeStore,State.masked_scatter_eq_filter,inv.head]
  · simp [v,FPStep.result,FPMemory.scatter,FPStep.beforeStore,State.masked_scatter_eq_filter,next]
    rfl
  · intro t ht
    change toFloat h.inputDType (((FPMemory.scatter M h c row q values pre).mem h.input _).read _)=_
    rw [FPMemory.other_region M h c row q values pre h.input layout.distinct]
    simpa [pre,FPStep.beforeStore] using inv.reads t ht
  · intro k lower ki i hi
    change (FPMemory.scatter M h c row q values pre).mem h.output _=_
    by_cases same : k=q
    · subst k
      rw [FPMemory.cell M h layout c row q hr start values pre i hi]
      exact congrArg (fun acc => FPMemory.stored M h (FPValues.output M scan h c input q acc (i,PUnit.unit))) next
    · have later : (q+1)*c.BT≤k*c.BT+i.val := by
        change q≤k at lower
        exact (Nat.mul_le_mul_right c.BT (by omega)).trans (Nat.le_add_right _ _)
      rw [FPMemory.preserves_later M h layout c row q hr start values pre _ hi later]
      exact inv.output k (by change q≤k at lower; omega) ki i hi
  · have frame := FPMemory.frame M h c row q start values pre
    intro r off hn
    exact (inv.frame r off hn).trans (frame r off hn)

def initial (M : Algebra α) (u : State α) : State α :=
  (u.setReg "i_bh" .nat [] (fun _ => u.pids 0)).setReg "b_z" .real [] (fun _ => M.literal (some .fp32) .real 0)

theorem init_executes (M : Algebra α) (scan : ScanBackend α) (u : State α) :
    run (machine M scan) initCode u=some (initial M u) := by
  simp [initCode,run,step,FP.Structural.evalExpr,evalComputeOp,FP.Structural.evalOp,
    machine,Algebra.withDefaultPrecision,resolvePrecision,initial]
  rfl

def input (h : Call) (u : State α) (row time : Nat) : α :=
  toFloat h.inputDType ((u.mem h.input (h.inputStrides.address h.H row time)).read h.inputDType.toTileDType)

theorem invariant (M : Algebra α) (scan : ScanBackend α) (h : Call) (c : Config) (u : State α) :
    Holds M scan u h c (u.pids 0) (input h u (u.pids 0)) (Geometry.chunks h c) (initial M u) := by
  refine ⟨le_rfl,?_,?_,by intros; rfl,?_,fun _ _ _ => rfl⟩
  · simp [initial]
  · rw [carry]
    simp [initial]
  · intro q lower inside
    omega

theorem mirror_invariant (M : Algebra α) (scan : ScanBackend α) (initial : State α)
    (h : Call) (c : Config) (row : Nat) (input : Nat → α) (q : Nat) (u : State α)
    (inv : Holds M scan initial h c row input q u) (j : Int) :
    Holds M scan initial h c row input q (u.setReg "__rev_i_t" .int [] (fun _ => j)) := by
  refine ⟨inv.count_le,?_,?_,inv.reads,inv.output,inv.frame⟩
  · simpa using inv.head
  · simpa using inv.carryReg

/-- Complete signed range executes for every opaque numerical backend. -/
theorem loop_executes (M : Algebra α) (scan : ScanBackend α) (initial : State α)
    (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (positive : 0<c.BT)
    (row : Nat) (hr : row<h.B*h.H) (input : Nat → α) (j : Int) (u : State α)
    (inv : Holds M scan initial h c row input (1-j).toNat u) :
    ∃ v, rangeInt (machine M scan) "__rev_i_t" j 1 1 (loopBody h c) u=some v ∧
      Holds M scan initial h c row input 0 v := by
  by_cases hj : j<1
  · have hc : (1-j).toNat=(-j).toNat+1 := by omega
    have hs : (1-(j+1)).toNat=(-j).toNat := by omega
    have hm := mirror_invariant M scan initial h c row input _ u inv j
    rw [hc] at hm
    obtain ⟨v,he,hv⟩ := iteration M scan initial h layout c positive row hr input j
      (u.setReg "__rev_i_t" .int [] (fun _ => j)) hj (by simp) hm
    obtain ⟨w,hw,hf⟩ := loop_executes M scan initial h layout c positive row hr input (j+1) v (by simpa only [hs] using hv)
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
  simp [statement,loopStart,countOp,step,FP.Structural.evalOp,numeric,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Geometry.chunks]

def Post (M : Algebra α) (scan : ScanBackend α) (h : Call) (c : Config) (positive : 0<c.BT)
    (u v : State α) : Prop :=
  (∀ time, time<h.T → v.mem h.output (h.outputStrides.address h.H (u.pids 0) time)=
    FPMemory.stored M h (expected M scan h c positive (input h u (u.pids 0)) time)) ∧
  FPMemory.Frame h (u.pids 0) u v

theorem source_correct (M : Algebra α) (scan : ScanBackend α) (h : Call) (layout : Geometry.OutputLayout h)
    (c : Config) (positive : 0<c.BT) (u : State α) (hr : u.pids 0<h.B*h.H) :
    ∃ v, FP.Structural.exec (machine M scan) (Kernels.source h c) u=some v ∧ Post M scan h c positive u v := by
  have hi : (1-(1-(Geometry.chunks h c:Int))).toNat=Geometry.chunks h c := by omega
  obtain ⟨v,he,inv⟩ := loop_executes M scan u h layout c positive (u.pids 0) hr _
    (1-(Geometry.chunks h c:Int)) (initial M u) (by simpa only [hi] using invariant M scan h c u)
  refine ⟨v,?_,?_,inv.frame⟩
  · change run (machine M scan) (Kernels.source h c).surfaceBody u=some v
    rw [source_body,body,run_append,init_executes]
    simp only [Option.bind_some,run,statement_eval,he]
    rfl
  · intro time ht
    have qi : time/c.BT<Geometry.chunks h c := (Nat.div_lt_iff_lt_mul positive).mpr
      (ht.trans_le (Geometry.chunks_cover h c positive))
    let i : Fin c.BT := ⟨time%c.BT,Nat.mod_lt _ positive⟩
    have eq : time/c.BT*c.BT+i.val=time := by
      dsimp only [i]
      rw [Nat.mul_comm]
      exact Nat.div_add_mod time c.BT
    have out := inv.output (time/c.BT) (Nat.zero_le _) qi i (by simpa [eq] using ht)
    simpa only [eq] using out

#print axioms iteration
#print axioms loop_executes
#print axioms source_correct
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.FPExecution
