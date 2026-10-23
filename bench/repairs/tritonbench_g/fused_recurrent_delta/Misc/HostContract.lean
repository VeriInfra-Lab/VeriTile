import bench.repairs.tritonbench_g.fused_recurrent_delta.ABISource
import bench.repairs.tritonbench_g.fused_recurrent_delta.GeometrySource

namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Admission

def regions : List Region := [.q,.k,.v,.beta,.h0,.o,.ht,.do_,.dht,.dq,.dk,.dv,.dbeta,.dh0,.retq,.retk,.retv,.retbeta]
def span (s : Shape) (f : Flags) (r : Region) : Nat :=
  Geometry.capacity s f r+Geometry.rows s r*Geometry.cols s f r+
  Geometry.cols s f r+s.BK*s.V+s.BBV+s.BK

instance (s : Shape) : Decidable s.Positive := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _))

def shapeCheck (c : ABI.Call) : Bool :=
  decide (c.shape.Positive ∧ c.shape.BH≤65535 ∧ c.shape.FNV<2^31 ∧ c.shape.BNV<2^31 ∧
    c.shape.BK*c.shape.BBV≤1048576) &&
  regions.all (fun r => decide (span c.shape (ABI.flags c) r<2^31))

structure Descriptor where
  object : Nat
  storage : Nat
  base : Nat
  device : Nat
  dimensions : List Nat
  strides : List Nat
  isTensor : Bool
  fp32 : Bool
  strided : Bool
  cuda : Bool
  contiguous : Bool
  noNeg : Bool
  noConj : Bool

def denseStrides : List Nat → List Nat
  | [] => []
  | _::tail => (tail.foldl (fun product dimension => product*max dimension 1) 1)::denseStrides tail

def descriptorCheck (d : Descriptor) (shape : List Nat) (device : Nat) : Bool :=
  d.isTensor && d.fp32 && d.strided && d.cuda && d.contiguous && d.noNeg && d.noConj &&
    decide (d.dimensions=shape ∧ d.strides=denseStrides shape ∧ d.device=device)

def windowCheck (c : ABI.Call) (r : Region) (d : Descriptor) : Bool :=
  decide (d.base%4=0 ∧ 4*max (max c.shape.K c.shape.V) 1≤d.base ∧
    d.base+4*span c.shape (ABI.flags c) r+4≤2^64)

def separationCheck (c : ABI.Call) (tensors : List (Region×Descriptor)) : Bool :=
  tensors.all fun x => tensors.all fun y =>
    decide (x.1=y.1 ∨ x.2.base+4*Geometry.capacity c.shape (ABI.flags c) x.1≤y.2.base ∨
      y.2.base+4*Geometry.capacity c.shape (ABI.flags c) y.1≤x.2.base)

def inputRegions (c : ABI.Call) : List Region :=
  [.q,.k,.v] ++ (if c.defaultBeta then [] else [.beta]) ++ (if c.initial then [.h0] else [])

def tensorChecks (c : ABI.Call) (device : Nat) (tensors : List (Region×Descriptor)) : Bool :=
  tensors.all (fun x => descriptorCheck x.2 (ABI.tensorMeta c.shape (ABI.flags c) x.1).shape device &&
    windowCheck c x.1 x.2) && separationCheck c tensors &&
    (inputRegions c).all (fun r => (tensors.map Prod.fst).contains r)

def flagsCheck (c : ABI.Call) : Bool := decide
  ((c.defaultBeta=Bool.true → c.vectorBeta=Bool.false) ∧
   (c.terminalCotangent=Bool.true → c.outputFinal=Bool.true))

def check (c : ABI.Call) (device : Nat) (tensors : List (Region×Descriptor))
    (outputIsBool scaleAccepted : Bool) : Bool :=
  outputIsBool && scaleAccepted && shapeCheck c && flagsCheck c &&
    c.scale.launch.toRat?.isSome && tensorChecks c device tensors

def backwardCheck (alreadyUsed higherOrder : Bool) : Bool := !alreadyUsed && !higherOrder
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Admission
