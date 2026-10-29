import bench.repairs.tritonbench_g.quantize_global.Kernels
import bench.tritonbench_g.quantize_global.Kernels
import bench.tritonbench_g.quantize_global.Proofs.IntegerSemantics
/-! Separate repair candidate. Source FP32, output int8, and the original two
autotune configurations are retained. The candidate deliberately changes the
reciprocal graph to div.rn.f32 followed by mul.f32. `source` is the natural-index
projection: the i64 casts are justified in Proofs/Indices; its `/` node denotes
the explicit div.rn operation, never Triton's approximate `/` lowering.
The native source and this projection are pinned by the evidence source audit.
-/
namespace QuantizeGlobalRepair
open VeriTile.Triton FP.Structural
open QuantizeIntegerStage
abbrev Config := QuantizeGlobalFinite.Config
def span : Nat := 1073741824

noncomputable def kernel (X Maximum Out : RegionName) (N base B : Nat) : TypedKernel :=
  ⟨source X Maximum Out N base B, byteOutput Out, outputRegister⟩

/-- Local scalar operations only. No output, tensor, index, memory or launch
oracle is admitted here. FP32 casts/loads preserve the selected word. -/
structure Scalar (α : Type) where
  literal : ℝ → α
  neg : α → α
  mul : α → α → α
  /-- The Python source's explicit `tl.div_rn` primitive. The DSL `/` node is
  interpreted through this binding, not through Triton's approximate division. -/
  div : α → α → α
  lt : α → α → Bool
  eq : α → α → Bool
  round64 : α → BitVec 64

def denominator (a : Scalar α) (maximum : α) : α :=
  if a.eq maximum (a.literal 0) then a.literal 1 else maximum

def scaled (a : Scalar α) (x maximum : α) : α :=
  a.mul (a.literal 127) (a.div x (denominator a maximum))

def bounded (a : Scalar α) (v : α) : α :=
  let lower := if a.lt v (a.literal (-127)) then a.literal (-127) else v
  if a.lt (a.literal 127) lower then a.literal 127 else lower

def scalar (a : Scalar α) (x maximum : α) : BitVec 8 :=
  BitVec.ofInt 8 (a.round64 (bounded a (scaled a x maximum))).toInt

noncomputable def backend (rest : Algebra α) (a : Scalar α) : Algebra α :=
  { rest with
    literal := fun _ _ => a.literal
    binary := fun p d op => match op with
      | .mul => a.mul | .div => a.div | op => rest.binary p d op
    unary := fun p op => match op with
      | .neg _ => a.neg | op => rest.unary p op
    compareLt := fun _ _ => some a.lt
    compareEq := fun _ _ => some a.eq
    cast := fun _ _ _ => id
    fp32Load := id }

structure Invocation where
  base : Nat
  config : Config
  pid : Nat

def Invocation.offset (v : Invocation) (lane : Nat) := v.base + v.pid * v.config.block + lane
def Invocation.indices (v : Invocation) := (List.range v.config.block).map v.offset
def segmentCount (n : Nat) := (n + span - 1) / span
def segmentSize (n k : Nat) := min (n - k * span) span
def programs (n k : Nat) (c : Config) := QuantizeGlobalFinite.programs (segmentSize n k) c
def launch (n k : Nat) (c : Config) : List Invocation :=
  (List.range (programs n k c)).map (fun pid => ⟨k*span,c,pid⟩)
def schedule (n : Nat) (trials : Nat → List Config) : List Invocation :=
  (List.range (segmentCount n)).flatMap (fun k => (trials k).flatMap (launch n k))
def footprint (vs : List Invocation) := vs.flatMap Invocation.indices

end QuantizeGlobalRepair
