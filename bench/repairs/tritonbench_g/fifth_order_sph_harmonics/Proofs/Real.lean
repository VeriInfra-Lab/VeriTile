import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Grid
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.ScalarReal

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.RealProof
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 64000

/-- Independent ideal values at element-start addresses. Arbitrary Reals are
never encoded as finite floating-point words. -/
abbrev Heap := Nat → ℝ

def initial (coordinates upstream output : Nat) (memory : Heap) (backward : Bool) : State ℝ where
  mem region index:=.mk .real (
    if region=coordinateRegion then memory (coordinates+4*index)
    else if region=upstreamRegion then memory (upstream+4*index)
    else if region=gradientRegion ∧ backward=Bool.true then 0
    else memory (output+4*index))
  regs _ _ _:=none
  pids _:=0
  numPids _:=1
  undef dtype _ _:=defaultValue dtype

def coordinate (coordinates : Nat) (memory : Heap) (r : Nat) (c : Fin 3) : ℝ :=
  memory (coordinates+4*(r*3+c.val))
def upstreamValue (p : Parameters) (upstream : Nat) (memory : Heap) (r : Nat) (c : Fin 11) : ℝ :=
  memory (upstream+4*(r*p.upstreamRowStride+(p.columnOffset+c.val)*p.upstreamColStride))

theorem forward_value (rest : Algebra ℝ) (p : Parameters) (coordinates upstream output : Nat)
    (memory : Heap) (r : Nat) (c : Fin 11) :
    Grid.value (ScalarReal.backend rest) .forward p (initial coordinates upstream output memory Bool.false) r c=
      Polynomial.value c (coordinate coordinates memory r 0)
        (coordinate coordinates memory r 1) (coordinate coordinates memory r 2) := by
  simp only [Grid.value,initial,ScalarReal.backend,coordinate,coordinateRegion,upstreamRegion,
    Cell.read_same,Fin.val_zero,Fin.val_one,Fin.val_two,Nat.add_zero,ite_true]
  exact ScalarReal.forward rest _ _ _ c

theorem backward_value (rest : Algebra ℝ) (p : Parameters) (coordinates upstream output : Nat)
    (memory : Heap) (r : Nat) (c : Fin 3) :
    Grid.value (ScalarReal.backend rest) .backward p (initial coordinates upstream output memory Bool.true) r c=
      Polynomial.vjp (coordinate coordinates memory r 0) (coordinate coordinates memory r 1)
        (coordinate coordinates memory r 2) (upstreamValue p upstream memory r) c := by
  simp only [Grid.value,Readback.backwardExpected,initial,coordinate,upstreamValue,
    coordinateRegion,upstreamRegion,gradientRegion,Cell.read_same,
    Fin.val_zero,Fin.val_one,Fin.val_two,Nat.add_zero,ScalarReal.backend]
  simp only [reduceCtorEq,ite_true,ite_false,and_self]
  exact ScalarReal.backward rest _ _ _ _ c

theorem forward (rest : Algebra ℝ) (p : Parameters) (fit : p.columnOffset+11≤p.outputWidth)
    (coordinates output : Nat) (memory : Heap) :
    ∃ final,Grid.launch (ScalarReal.backend rest) .forward p (initial coordinates 0 output memory Bool.false)=some final ∧
      (∀ r,r<p.rows → ∀ c : Fin 11,final.mem outputRegion (r*p.outputWidth+p.columnOffset+c.val)=
        .mk .real (Polynomial.value c (coordinate coordinates memory r 0)
          (coordinate coordinates memory r 1) (coordinate coordinates memory r 2))) ∧
      (∀ region index,(region≠outputRegion ∨ ∀ r,r<p.rows → ∀ c : Fin 11,
        index≠r*p.outputWidth+p.columnOffset+c.val) →
        final.mem region index=(initial coordinates 0 output memory Bool.false).mem region index) := by
  obtain ⟨final,run,result,frame⟩:=Grid.launch_correct (ScalarReal.backend rest) .forward p
    (initial coordinates 0 output memory Bool.false) fit
  exact ⟨final,run,fun r hr c=>(result r hr c).trans (congrArg (Cell.mk .real) (forward_value rest p coordinates 0 output memory r c)),frame⟩

theorem backward (rest : Algebra ℝ) (p : Parameters) (fit : p.columnOffset+11≤p.outputWidth)
    (coordinates upstream output : Nat) (memory : Heap) :
    ∃ final,Grid.launch (ScalarReal.backend rest) .backward p (initial coordinates upstream output memory Bool.true)=some final ∧
      (∀ r,r<p.rows → ∀ c : Fin 3,final.mem gradientRegion (r*3+c.val)=
        .mk .real (Polynomial.vjp (coordinate coordinates memory r 0)
          (coordinate coordinates memory r 1) (coordinate coordinates memory r 2)
          (upstreamValue p upstream memory r) c)) ∧
      (∀ region index,(region≠gradientRegion ∨ ∀ r,r<p.rows → ∀ c : Fin 3,index≠r*3+c.val) →
        final.mem region index=(initial coordinates upstream output memory Bool.true).mem region index) := by
  obtain ⟨final,run,result,frame⟩:=Grid.launch_correct (ScalarReal.backend rest) .backward p
    (initial coordinates upstream output memory Bool.true) fit
  exact ⟨final,run,fun r hr c=>(result r hr c).trans (congrArg (Cell.mk .real) (backward_value rest p coordinates upstream output memory r c)),frame⟩

#axiomsClean forward
#axiomsClean backward
end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.RealProof
