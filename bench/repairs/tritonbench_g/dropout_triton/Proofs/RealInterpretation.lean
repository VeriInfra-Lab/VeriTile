import bench.repairs.tritonbench_g.dropout_triton.Proofs.Wrapper
import bench.repairs.tritonbench_g.dropout_triton.Proofs.RealGrid

namespace VeriTile.Bench.Repairs.DropoutTriton.RealInterpretation
open VeriTile.Triton FP.Structural
noncomputable section

/-- Precision-erased interpretation. Only literal, load, cast, subtraction and
 division are reached by this source; the other required algebra fields are
 irrelevant total functions. This is not a physical floating implementation. -/
def model : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y
    | .max => max x y | .pow => 0
  unary := fun _ _ x => x
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 0
  scan := fun _ {_} _ _ _ v => some v
  dot := fun _ {_} {_} {_} {_} _ _ => some (fun _ => 0)
  dotAcc := fun _ _ _ out _ {_} {_} {_} {_} _ _ _ => some (fun _ => ofFloat out 0)
  dotTyped := fun _ _ _ out _ {_} {_} {_} {_} _ _ => some (fun _ => ofFloat out 0)

theorem scalar (n : Nat) (p : ℝ) (mem : RegionName → Nat → Cell ℝ) (i : Nat) :
    Typed.resultCell model (Wrapper.parameters n) p mem i =
      .mk .real (expected (fun i => (mem "x" i).read .real)
        (fun i => (mem "keep" i).read .int) p i) := rfl

theorem same_erased_source (n : Nat) (p : ℝ) :
    (kernel n p).eraseDType.toAlgKernel.body =
      (RealGrid.kernel .int32 "x" "keep" "out" n p).body :=
  VeriTile.Bench.TritonBenchG.DropoutTriton.Erasure.body .fp32 .int32 Bool.false
    "x" "keep" "out" n 1024 p

#axiomsClean same_erased_source
#axiomsClean scalar
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DropoutTriton.RealInterpretation
