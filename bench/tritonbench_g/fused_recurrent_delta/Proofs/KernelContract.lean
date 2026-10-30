import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelProgram
import bench.tritonbench_g.fused_recurrent_delta.Proofs.Index
import bench.tritonbench_g.fast_layernorm.Contract
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelContract
open VeriTile.Triton VeriTile.Torch KernelProgram

def span (s : Shape) (f : Flags) (r : Region) :=
  Geometry.capacity s f r+Geometry.rows s r*Geometry.cols s f r+
    Geometry.cols s f r+s.BK*s.V+s.BBV+s.BK
structure Domain (s : Shape) (f : Flags) : Prop extends Index.Safe s f where
  block : s.BK*s.BBV≤1048576
  initialGradient : f.initialGrad=Bool.true → f.initial=Bool.true
  terminalGradient : f.terminalGrad=Bool.true → f.final=Bool.true
  defaultBeta : f.defaultBeta=Bool.true → f.vectorBeta=Bool.false

def dimensions (s : Shape) (f : Flags) : Region → List Nat
  | .q | .k | .retq | .retk => [s.B,s.H,s.T,s.K]
  | .h0 | .ht | .dht | .dh0 => [s.B,s.H,s.K,s.V]
  | .dq | .dk => [s.BNV,s.B,s.H,s.T,s.K]
  | .dbeta => if f.vectorBeta then [s.BNV,1,s.B,s.H,s.T,s.V] else [s.BNV,s.B,s.H,s.T]
  | .beta | .retbeta => if f.vectorBeta then [s.B,s.H,s.T,s.V] else [s.B,s.H,s.T]
  | .o | .dv => [1,s.B,s.H,s.T,s.V]
  | _ => [s.B,s.H,s.T,s.V]
def denseStrides : List Nat → List Nat
  | [] => []
  | _::tail => (tail.map (max 1)).prod::denseStrides tail
def roles (kind : Kind) (f : Flags) : List Region :=
  match kind with
  | .forward => [.q,.k,.v,.beta,.o]++(if f.initial then [.h0] else [])++(if f.final then [.ht] else [])
  | .backward => [.q,.k,.v,.beta,.do_,.dq,.dk,.dv,.dbeta]++
      (if f.initial then [.h0] else [])++(if f.initialGrad then [.dh0] else [])++(if f.terminalGrad then [.dht] else [])
def readRoles (kind : Kind) (f : Flags) : List Region :=
  match kind with
  | .forward => [.q,.k,.v,.beta]++(if f.initial then [.h0] else [])
  | .backward => [.q,.k,.v,.beta,.do_]++(if f.initial then [.h0] else [])++(if f.terminalGrad then [.dht] else [])
def outputRoles (kind : Kind) (f : Flags) : List Region :=
  match kind with
  | .forward => [.v,.o]++(if f.final then [.ht] else [])
  | .backward => [.dq,.dk,.dv,.dbeta]++(if f.initialGrad then [.dh0] else [])

/-- These two observations supplement shared TensorMetadata, whose core
currently has no strides or conjugate-view field. -/
structure Observation where
  metadata : TensorMetadata
  strides : List Nat
  isConj : Bool
structure TensorCheck (s : Shape) (f : Flags) (r : Region) (device : Device) (o : Observation) : Prop extends
    VeriTile.Bench.Repairs.FastLayernorm.Contract.TensorContract o.metadata Bool.true where
  shape : o.metadata.shape=dimensions s f r
  strides : o.strides=denseStrides (dimensions s f r)
  noConj : o.isConj=Bool.false
  device : o.metadata.device=device
  lower : 4*max (max s.K s.V) 1≤o.metadata.dataPtr.toNat
  window : o.metadata.dataPtr.toNat+4*span s f r+4≤2^64
structure InputContract (kind : Kind) (s : Shape) (f : Flags) (o : Region → Observation) : Prop where
  domain : Domain s f
  tensors : ∀ r,r∈roles kind f → TensorCheck s f r (o .q).metadata.device (o r)
  separated : ∀ r,r∈roles kind f → ∀ r',r'∈roles kind f → r≠r' →
    (o r).metadata.dataPtr.toNat+4*(o r).metadata.numel≤(o r').metadata.dataPtr.toNat ∨
    (o r').metadata.dataPtr.toNat+4*(o r').metadata.numel≤(o r).metadata.dataPtr.toNat

theorem capacity (s : Shape) (f : Flags) (r : Region) :
    (dimensions s f r).prod=Geometry.capacity s f r := by
  cases r <;> cases hv : f.vectorBeta <;>
    simp [dimensions,Geometry.capacity,Geometry.planes,Geometry.rows,Geometry.cols,Shape.BH,hv,Nat.mul_assoc]

#axiomsClean capacity
#specNonCircular InputContract avoiding [FP.Structural.exec,FP.Structural.run,KernelProgram.source]
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelContract
