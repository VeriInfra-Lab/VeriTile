import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.Source
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Target

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Native
open VeriTile Triton Source
open VeriTile.Bench.Repairs.AttentionLlama NativeSchedule
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000

/-- Source literal interpretation, used only at zero and one by this kernel.
Raw Python scale is a distinct bound leaf, so signed zero and integer tags do
not pass through this Real literal port. -/
def literal (A : NativeGraph.Ops α) (x : ℝ) : α :=
  if x=0 then A.zero else A.one

/-- Integer one is a compile-time literal. Every other specialization binds
the converted native parameter, including a runtime floating value of one. -/
def argument (C : ScalarABI.Conversions α) : ScalarABI.Argument → Term α
  | .one => .literal 1
  | a => .leaf (ScalarABI.convert C a)

def scaled (A : NativeGraph.Ops α) (scale : Term α) (runtime dot : α) : α :=
  match scale with
  | .literal x => if x=1 then dot else A.mul runtime dot
  | _ => A.mul runtime dot

/-- Task-local SM80 source selection. These are syntax-directed per-operation
rules: seeded K16 dot composition, the observed normalizer FMA, the separately
rounded carry product, layout-selected trees, exp2 lowering and explicit half
conversions. The graph interpreter is a model boundary, not a deferred
numerical experiment and not a whole-kernel correspondence hypothesis. -/
def lower (A : NativeGraph.Ops α) (fp8 : Bool) (D : Nat) (l : Fin 4) : Term α → α
  | .leaf x => x
  | .literal x => literal A x
  | .negativeInfinity => A.negativeInfinity
  | .add (.literal x) .negativeInfinity =>
      if x=0 then A.negativeInfinity else A.add (literal A x) A.negativeInfinity
  | .add a (.dot n x y) =>
      mmaFold A.mma16 (mmaCoordinate fp8 n)
        (fun j=>if h:j<n then lower A fp8 D (lane j) (x ⟨j,h⟩) else A.zero)
        (fun j=>if h:j<n then lower A fp8 D (lane j) (y ⟨j,h⟩) else A.zero)
        (lower A fp8 D l a) (n/16)
  | .add (.sum n xs) (.mul a b) =>
      A.fma (lower A fp8 D l a) (lower A fp8 D l b)
        ((selected D l).eval A.add (fun j=>if h:j.val<n then lower A fp8 D (lane j.val) (xs ⟨j.val,h⟩) else A.zero))
  | .add (.mul a b) (.sum n xs) =>
      A.fma (lower A fp8 D l a) (lower A fp8 D l b)
        ((selected D l).eval A.add (fun j=>if h:j.val<n then lower A fp8 D (lane j.val) (xs ⟨j.val,h⟩) else A.zero))
  | .add a b => A.add (lower A fp8 D l a) (lower A fp8 D l b)
  | .sub a b => A.sub (lower A fp8 D l a) (lower A fp8 D l b)
  | .mul (.add (.literal z) (.dot n x y)) scale =>
      let dot := mmaFold A.mma16 (mmaCoordinate fp8 n)
        (fun j=>if h:j<n then lower A fp8 D (lane j) (x ⟨j,h⟩) else A.zero)
        (fun j=>if h:j<n then lower A fp8 D (lane j) (y ⟨j,h⟩) else A.zero)
        (literal A z) (n/16)
      if z=0 then scaled A scale (lower A fp8 D l scale) dot
      else A.mul dot (lower A fp8 D l scale)
  | .mul a b => A.mul (lower A fp8 D l a) (lower A fp8 D l b)
  | .div a b => A.div (lower A fp8 D l a) (lower A fp8 D l b)
  | .maximum a b => A.maximum (lower A fp8 D l a) (lower A fp8 D l b)
  | .exp a => NativeGraph.exp A (lower A fp8 D l a)
  | .half (.byte b) => A.e5m2ToHalf b
  | .half (.literal x) => if x=0 then A.zero else A.half (literal A x)
  | .half a => A.half (lower A fp8 D l a)
  | .byte b => A.e5m2ToHalf b
  | .dot n x y =>
      mmaFold A.mma16 (mmaCoordinate fp8 n)
        (fun j=>if h:j<n then lower A fp8 D (lane j) (x ⟨j,h⟩) else A.zero)
        (fun j=>if h:j<n then lower A fp8 D (lane j) (y ⟨j,h⟩) else A.zero) A.zero (n/16)
  | .sum n xs => (selected D l).eval A.add
      (fun j=>if h:j.val<n then lower A fp8 D (lane j.val) (xs ⟨j.val,h⟩) else A.zero)
  | .max n xs => (selected D l).eval A.maximum
      (fun j=>if h:j.val<n then lower A fp8 D (lane j.val) (xs ⟨j.val,h⟩) else A.zero)

/-- Both source positions of the rescaled carry select the same native FMA,
with identical operand order. This is not split-add commutativity. -/
theorem carry_first_fma (A : NativeGraph.Ops α) (fp8 : Bool) (D : Nat) (l : Fin 4)
    (a b : Term α) (n : Nat) (xs : Fin n → Term α) :
    lower A fp8 D l (.add (.mul a b) (.sum n xs)) =
      lower A fp8 D l (.add (.sum n xs) (.mul a b)) := by rfl

 theorem leaf_identity (A : NativeGraph.Ops α) (fp8 : Bool) (D : Nat) (l : Fin 4) (scale : α) :
    lower A fp8 D l (.leaf scale)=scale := rfl

 theorem coordinate64 (fp8 : Bool) (g : Fin 4) (i : Fin 16) :
    mmaCoordinate fp8 64 g.val i<64 := by
  cases fp8 <;> revert g i <;> decide

theorem scale_binding (A : NativeGraph.Ops α) (C : ScalarABI.Conversions α)
    (arg : ScalarABI.Argument) (fp8 : Bool) (D : Nat) (l : Fin 4) (dot : α) :
    scaled A (argument C arg) (lower A fp8 D l (argument C arg)) dot=
      Target.scaleProduct A C arg dot := by
  cases arg <;> simp [argument,scaled,Target.scaleProduct,lower]

#axiomsClean carry_first_fma
#axiomsClean leaf_identity
#axiomsClean coordinate64
#axiomsClean scale_binding
#specNonCircular lower avoiding [FP.Structural.exec,FP.Structural.run,NativeGraph.run,NativeGraph.returnedWord]
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Native
