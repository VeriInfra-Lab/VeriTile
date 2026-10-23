import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.NativeProfile
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.Kernels

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2
open VeriTile.Triton FP.Structural
noncomputable section
namespace Scalar
abbrev Term := SoftmaxFlaggemsFiniteSeed.Native.Term
abbrev Shape := SoftmaxFlaggemsFiniteSeed.Native.Shape
abbrev Tree := SoftmaxFlaggemsFiniteSeed.Native.Tree
abbrev Plan := SoftmaxFlaggemsFiniteSeed.Native.Plan
abbrev Reductions := SoftmaxFlaggemsFiniteSeed.Native.Reductions
abbrev Schedule := SoftmaxFlaggemsFiniteSeed.Native.Schedule
abbrev Instructions := SoftmaxFlaggemsFiniteSeed.Native.Instructions
abbrev eval := SoftmaxFlaggemsFiniteSeed.Native.eval
abbrev lower := SoftmaxFlaggemsFiniteSeed.NativeProfile.lower
abbrev instructions := SoftmaxFlaggemsFiniteSeed.NativeProfile.instructions
end Scalar
namespace Native
abbrev Word := BitVec 32
/-- Actual destination cell. The selected layout maps this coordinate to its
physical consumer lane, warp and register. It never depends on input words. -/
abbrev Consumer := RegionName × Nat
/-- A logical scalar retains every consumer's scalar expression until store
observation. Broadcasting a logical row never forces its replicas equal. -/
abbrev Term := Consumer → Scalar.Term
abbrev Reductions := Consumer → Scalar.Reductions
abbrev Schedule := Consumer → Scalar.Schedule

def algebra (R : Reductions) : Algebra Term where
  literal := fun p d x _ => .literal p d x
  negInf := fun _ => .negInf
  binary := fun p d op a b consumer => .binary p d op (a consumer) (b consumer)
  unary := fun p op a consumer => .unary p op (a consumer)
  cast := fun p src dst a consumer => .cast p src dst (a consumer)
  fromNat := fun p n _ => .fromNat p n
  fromInt := fun p n _ => .fromInt p n
  fp32Bits := fun w _ => .bits w.bits
  fp32Load := fun a consumer => .load (a consumer)
  reduceMax := fun p {_} ax keep xs out consumer =>
    SoftmaxFlaggemsFiniteSeed.Native.reduction (R consumer) Bool.true p ax keep (fun i => xs i consumer) out
  reduceSum := fun p {_} ax keep xs out consumer =>
    SoftmaxFlaggemsFiniteSeed.Native.reduction (R consumer) Bool.false p ax keep (fun i => xs i consumer) out

/-- Each new kernel reads fixed, already-stored FP32 words. Producer expressions
are not carried across this boundary and cannot be retargeted by a consumer. -/
def entry (memory : RegionName → Nat → Word) : State Term where
  mem := fun r o => .mk .real (fun _ => .leaf r o (memory r o))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

def observe (I : Scalar.Instructions) (S : Schedule) (s : State Term) (r : RegionName) (o : Nat) : Word :=
  Scalar.eval I (S (r,o)) (((s.mem r o).read .real) (r,o))
def freeze (I : Scalar.Instructions) (S : Schedule) (s : State Term) : RegionName → Nat → Word :=
  observe I S s

@[simp] theorem entry_observe (I : Scalar.Instructions) (S : Schedule)
    (memory : RegionName → Nat → Word) (r : RegionName) (o : Nat) : observe I S (entry memory) r o=memory r o := rfl

/-- Word-level handoff remains the exact producer word under any later kernel
profile or consumer. No replica equality or numerical accuracy premise occurs. -/
theorem stored_word_handoff (I J : Scalar.Instructions) (S T : Schedule) (s : State Term)
    (r : RegionName) (o : Nat) (consumer : Consumer) :
    Scalar.eval J (T consumer) (((entry (freeze I S s)).mem r o).read .real consumer)=observe I S s r o := rfl

/-- Reduction selection retains both the logical reduced coordinate and the
actual output consumer for every shape, axis and branch, including keep-dim. -/
theorem reduction_consumer (R : Reductions) (isMax : Bool) (p : Option ComputeDType)
    {s : TileShape} (ax : Fin s.length) (keep : Bool) (xs : Values Term .real s)
    (out : TileIndex (TileShape.reduceShape s ax keep)) (consumer : Consumer) :
    (if isMax then (algebra R).reduceMax p ax keep xs out else (algebra R).reduceSum p ax keep xs out) consumer =
      SoftmaxFlaggemsFiniteSeed.Native.reduction (R consumer) isMax p ax keep (fun i => xs i consumer) out := by
  cases isMax <;> rfl

#axiomsClean reduction_consumer
#axiomsClean stored_word_handoff
#specNonCircular observe avoiding [FP.Structural.exec]
end Native
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2
