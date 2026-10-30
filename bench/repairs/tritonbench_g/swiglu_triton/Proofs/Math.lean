import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Grid
import bench.tritonbench_g.swiglu_triton.Proofs.Math
import Mathlib.Analysis.SpecialFunctions.Sigmoid

namespace VeriTile.Bench.Repairs.SwigluTriton.Math
open VeriTile.Triton FP.Structural
noncomputable section

abbrev gate := VeriTile.Bench.TritonBenchG.SwigluTriton.Math.gate
abbrev forward := VeriTile.Bench.TritonBenchG.SwigluTriton.Math.forward
abbrev backward := VeriTile.Bench.TritonBenchG.SwigluTriton.Math.backward

def formula (ch : Channel) (a b dc : ℝ) : ℝ :=
  match ch with | .c => forward a b | .da => (backward dc a b).1 | .db => (backward dc a b).2

def algebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .mul => x*y | .add => x+y | .sub => x-y | .div => x/y | _ => 0
  unary := fun _ op x => match op with | .sigmoid => Real.sigmoid x | _ => 0
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => 0
  reduceSum := fun _ {_} _ _ _ => fun _ => 0

theorem scalar_real (ch : Channel) (a b dc : ℝ) : Tile.scalar algebra ch a b dc=formula ch a b dc := by
  cases ch <;> simp only [Tile.scalar,algebra,formula,forward,backward,gate,
    VeriTile.Bench.TritonBenchG.SwigluTriton.Math.forward,
    VeriTile.Bench.TritonBenchG.SwigluTriton.Math.backward,
    VeriTile.Bench.TritonBenchG.SwigluTriton.Math.gate,Real.sigmoid,id_eq] <;> ring

/-- Derivatives of the independent forward formula, without executing a
kernel or appealing to an output oracle. -/
theorem gate_derivative (a b : ℝ) :
    HasDerivAt (fun x => forward x b) (b*gate a*(1+a*(1-gate a))) a := by
  have h := ((hasDerivAt_id a).mul (Real.hasDerivAt_sigmoid a)).mul_const b
  convert h using 1
  · funext x
    simp only [forward,VeriTile.Bench.TritonBenchG.SwigluTriton.Math.forward,
      gate,VeriTile.Bench.TritonBenchG.SwigluTriton.Math.gate,Real.sigmoid,id_eq,Pi.mul_apply]
    ring
  · simp only [gate,VeriTile.Bench.TritonBenchG.SwigluTriton.Math.gate,Real.sigmoid,id_eq,Pi.mul_apply]
    ring

theorem value_derivative (a b : ℝ) : HasDerivAt (fun x => forward a x) (a*gate a) b := by
  convert (hasDerivAt_id b).const_mul (a*gate a) using 1
  · funext x
    simp only [forward,VeriTile.Bench.TritonBenchG.SwigluTriton.Math.forward,id_eq]
    ring
  · simp

theorem whole (isBackward : Bool) (rows cols block : Nat) (bounded : cols≤block)
    (schedule : List Nat) (allRows : schedule.Perm (List.range rows)) (s : State ℝ) :
    ∃ final,Grid.runRows algebra isBackward block cols schedule s=some final ∧
      (∀ ch,enabled isBackward ch → ∀ row col,row<rows → col<cols →
        final.mem (outputRegion ch) (row*block+col)=Cell.mk .real
          (formula ch ((s.mem "A" (row*block+col)).read .real)
            ((s.mem "B" (row*block+col)).read .real)
            ((s.mem "DC" (row*block+col)).read .real))) ∧
      (∀ region address,Grid.Outside isBackward rows cols block region address →
        final.mem region address=s.mem region address) := by
  simpa only [scalar_real] using Grid.whole algebra isBackward rows cols block bounded schedule allRows s

#specNonCircular formula avoiding [Tile.scalar,Tile.sourceRun,Grid.runRows]
#axiomsClean whole
#axiomsClean gate_derivative
#axiomsClean value_derivative
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.Math
