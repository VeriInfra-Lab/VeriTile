import bench.tritonbench_g.softmax_flaggems.Proofs.IO
import bench.tritonbench_g.softmax_flaggems.Proofs.GridSafety
import bench.tritonbench_g.softmax_flaggems.Proofs.RawCorrect

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Interface
open VeriTile Triton FP.Structural
open SoftmaxFlaggemsFiniteSeed
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

@[simp] theorem extend_at {α : Type} [Inhabited α] {n : Nat} (v : Fin n → α) (k : Fin n) :
    extend v k.val=v k := by simp [extend,k.isLt]

theorem forward_pins (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (v : (forwardIO X Y d c).Inputs) (s : State (WithBot ℝ))
    (h : (forwardIO X Y d c).InputBinding v s) (k : Nat) (hk : k<count d) :
    FPReference.read s X k=extend (forwardInput X Y d c v) k := by
  have pin := h (0 : Fin 1) (⟨k,hk⟩,PUnit.unit) trivial
  simpa [forwardIO,layout,channels,index,ProgramIO.InputBinding,mapValue,
    FPReference.read,extend,forwardInput,hk] using pin

theorem column_congr (d : Dimensions) (q p : Nat → WithBot ℝ)
    (h : ∀ k,k<count d → q k=p k) (row col : Nat) (hr : row<d.M) (hc : col<d.K) :
    Target.column q d.N d.K row col=Target.column p d.N d.K row col := by
  funext j
  apply h
  exact RealAddress.non_inner_bound hr j.isLt hc

theorem forward_target_congr (d : Dimensions) (q p : Nat → WithBot ℝ)
    (hN : 0<d.N) (hK : 0<d.K) (h : ∀ k,k<count d → q k=p k)
    (k : Nat) (hk : k<count d) : Target.flatForward q d.N d.K k=Target.flatForward p d.N d.K k := by
  obtain ⟨hr,_,hc,_⟩ := coordinate_noninner hN hK hk
  simp only [Target.flatForward,h k hk,column_congr d q p h _ _ hr hc]

theorem forward_real (X Y : RegionName) (separate : X≠Y) (d : Dimensions) (c : LaunchConfig)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B) :
    (forwardIO X Y d c).ImplementsOn (ForwardDomain X Y d c) (forwardFormula X Y d c) := by
  intro v admitted bounds backed s inputs
  have hx : ∀ k,k<count d → k<bounds X := by
    intro k hk
    exact backed.2.1 (0 : Fin 1) (⟨k,hk⟩,PUnit.unit) trivial
  have hy : ∀ k,k<count d → k<bounds Y := by
    intro k hk
    exact backed.2.2.1 (0 : Fin 1) (⟨k,hk⟩,PUnit.unit) trivial
  refine ⟨forward_safe Ideal.algebra X Y d c hN hK hB hone bounds hx hy s,?_⟩
  have pins := forward_pins X Y d c v s inputs
  have defined : RealForward.Defined (FPReference.read s X) d := by
    intro row hr col hc
    rw [column_congr d _ _ pins row col hr hc]
    exact admitted row hr col hc
  obtain ⟨final,ran,memory⟩ := RawCorrect.kernel_forward X Y separate d c s hN hK hB hC hone defined
  refine ⟨final,?_,?_,?_⟩
  · exact (forwardJobs_run Ideal.algebra X Y d c s).trans ran
  · intro o j _
    obtain ⟨o,ho⟩ := o
    have he : o=0 := by simpa [forwardIO,layout] using Nat.lt_one_iff.mp ho
    subst o
    change mapCell id (final.mem Y j.1.val)=Cell.mk .real _
    rw [memory,if_pos ⟨rfl,j.1.isLt⟩]
    exact congrArg (Cell.mk .real) (congrArg (fun x : ℝ => (x : WithBot ℝ))
      (forward_target_congr d _ _ hN hK pins j.1.val j.1.isLt))
  · intro r k outside
    rw [memory,if_neg]
    rintro ⟨rfl,hk⟩
    exact (outside.1 (0 : Fin 1) (⟨k,hk⟩,PUnit.unit) trivial).elim (fun h => h rfl) (fun h => h rfl)

theorem backward_pins (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (v : (backwardIO P G Y d c).Inputs) (s : State (WithBot ℝ))
    (h : (backwardIO P G Y d c).InputBinding v s) :
    RealBackward.Input P G (extend (backwardInput P G Y d c v 0))
      (extend (backwardInput P G Y d c v 1)) (count d) s := by
  constructor
  · intro k hk
    have pin := h (0 : Fin 2) (⟨k,hk⟩,PUnit.unit) trivial
    simpa [backwardIO,layout,channels,index,ProgramIO.InputBinding,mapValue,
      FPReference.read,extend,backwardInput,hk] using pin
  · intro k hk
    have pin := h (1 : Fin 2) (⟨k,hk⟩,PUnit.unit) trivial
    simpa [backwardIO,layout,channels,index,ProgramIO.InputBinding,mapValue,
      FPReference.read,extend,backwardInput,hk] using pin

theorem backward_real (P G Y : RegionName) (hPY : P≠Y) (hGY : G≠Y)
    (d : Dimensions) (c : LaunchConfig) (hN : 0<d.N) (hK : 0<d.K)
    (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B) :
    (backwardIO P G Y d c).Implements (backwardFormula P G Y d c) := by
  apply ProgramIO.Implements.intro
  · intro v bounds backed s _
    have hp : ∀ k,k<count d → k<bounds P := by
      intro k hk
      exact backed.2.1 (0 : Fin 2) (⟨k,hk⟩,PUnit.unit) trivial
    have hg : ∀ k,k<count d → k<bounds G := by
      intro k hk
      exact backed.2.1 (1 : Fin 2) (⟨k,hk⟩,PUnit.unit) trivial
    have hy : ∀ k,k<count d → k<bounds Y := by
      intro k hk
      exact backed.2.2.1 (0 : Fin 1) (⟨k,hk⟩,PUnit.unit) trivial
    exact backward_safe Ideal.algebra P G Y d c hK bounds hp hg hy s
  · intro v s inputs
    obtain ⟨final,ran,memory⟩ := RawCorrect.kernel_backward P G Y hPY hGY d c
      (extend (backwardInput P G Y d c v 0)) (extend (backwardInput P G Y d c v 1)) s
      hN hK hB hC hone (backward_pins P G Y d c v s inputs)
    refine ⟨final,?_,?_,?_⟩
    · exact (backwardJobs_run Ideal.algebra P G Y d c s).trans ran
    · intro o j _
      obtain ⟨o,ho⟩ := o
      have he : o=0 := Nat.lt_one_iff.mp ho
      subst o
      change mapCell (fun x : WithBot ℝ => x.unbotD 0) (final.mem Y j.1.val)=Cell.mk .real _
      rw [memory,if_pos ⟨rfl,j.1.isLt⟩]
      rfl
    · intro r k outside
      rw [memory,if_neg]
      rintro ⟨rfl,hk⟩
      exact (outside.1 (0 : Fin 1) (⟨k,hk⟩,PUnit.unit) trivial).elim (fun h => h rfl) (fun h => h rfl)

#axiomsClean forward_real
#axiomsClean backward_real
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Interface
