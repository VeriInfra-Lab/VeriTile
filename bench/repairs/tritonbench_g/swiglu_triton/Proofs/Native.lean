import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Math
import bench.tritonbench_g.swiglu_backward.Proofs.Observed

namespace VeriTile.Bench.Repairs.SwigluTriton.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term graphAlgebra)
noncomputable section
set_option maxRecDepth 20000

/-- Exact local ports of the compiler-resolved FP32 profile. Sigmoid is the
local exp/denominator/division subgraph, not the SwiGLU output. FMA is ternary.
The reference never assumes that FMA equals separately rounded mul/add. -/
structure Primitives (α : Type) where
  literal : ℝ → α
  cast : α → α
  sigmoid : α → α
  sub : α → α → α
  mul : α → α → α
  fma : α → α → α → α
  unused : α

/-- Site-specific lowering rules for this task's unchanged source graph.
They retain every observed operand order, including silu, db, and da's inner
product. These are graph rules, not commutativity or contraction identities. -/
def interpret (m : Primitives α) : Term α → α
  | .input x => x
  | .literal (some .fp32) .real x => m.literal x
  | .cast (some .fp32) .real .real x => m.cast (interpret m x)
  | .unary (some .fp32) .sigmoid x => m.sigmoid (interpret m x)
  | .binary (some .fp32) .real .add
      (.binary (some .fp32) .real .mul a b) c => m.fma (interpret m a) (interpret m b) (interpret m c)
  | .binary (some .fp32) .real .mul a (.unary (some .fp32) .sigmoid b) =>
      m.mul (m.sigmoid (interpret m b)) (interpret m a)
  | .binary (some .fp32) .real .mul a
      (.binary (some .fp32) .real .mul b (.unary (some .fp32) .sigmoid c)) =>
      m.mul (m.mul (m.sigmoid (interpret m c)) (interpret m b)) (interpret m a)
  | .binary (some .fp32) .real .mul a
      (.binary (some .fp32) .real .add (.binary (some .fp32) .real .mul b c) d) =>
      m.mul (m.fma (interpret m b) (interpret m c) (interpret m d)) (interpret m a)
  | .binary (some .fp32) .real .mul a b => m.mul (interpret m a) (interpret m b)
  | .binary (some .fp32) .real .sub a b => m.sub (interpret m a) (interpret m b)
  | _ => m.unused

/-- Independent fused FP reference read from the resolved native graph.
All raw input words, including NaN payloads and signed zero, retain order. -/
def reference (m : Primitives α) (ch : Channel) (a b dc : α) : α :=
  let a := m.cast a
  let s := m.sigmoid a
  let silu := m.mul s a
  match ch with
  | .c => m.mul silu b
  | .da => m.mul (m.mul (m.fma silu (m.sub (m.literal 1) s) s) dc) b
  | .db => m.mul silu dc

def sourceScalar (m : Primitives α) (ch : Channel) (a b dc : α) : α :=
  interpret m (Tile.scalar graphAlgebra ch (.input a) (.input b) (.input dc))

theorem graph_lowering (m : Primitives α) (ch : Channel) (a b dc : α) :
    sourceScalar m ch a b dc=reference m ch a b dc := by
  cases ch <;> rfl

def realPrimitives : Primitives ℝ where
  literal := id
  cast := id
  sigmoid := Real.sigmoid
  sub := fun a b => a-b
  mul := fun a b => a*b
  fma := fun a b c => a*b+c
  unused := 0

theorem reference_real (ch : Channel) (a b dc : ℝ) :
    reference realPrimitives ch a b dc=Math.formula ch a b dc := by
  cases ch <;> simp only [reference,realPrimitives,Math.formula,Math.forward,Math.backward,
    VeriTile.Bench.TritonBenchG.SwigluTriton.Math.forward,
    VeriTile.Bench.TritonBenchG.SwigluTriton.Math.backward,
    VeriTile.Bench.TritonBenchG.SwigluTriton.Math.gate,Real.sigmoid,id_eq] <;> ring

/-- Optional exact primitive bridge between a native implementation and its
specified FP32 ports. Every clause is scalar and profile-local; no clause
mentions a kernel result, host success, allocation, queue or output array. -/
structure LocalAgreement (native target : Primitives (BitVec 32)) : Prop where
  one : native.literal 1=target.literal 1
  cast : ∀ x,native.cast x=target.cast x
  sigmoid : ∀ x,native.sigmoid x=target.sigmoid x
  sub : ∀ x y,native.sub x y=target.sub x y
  mul : ∀ x y,native.mul x y=target.mul x y
  fma : ∀ x y z,native.fma x y z=target.fma x y z

theorem local_bridge (native target : Primitives (BitVec 32)) (h : LocalAgreement native target)
    (ch : Channel) (a b dc : BitVec 32) :
    sourceScalar native ch a b dc=reference target ch a b dc := by
  rw [graph_lowering]
  cases ch <;> simp only [reference,h.one,h.cast,h.sigmoid,h.sub,h.mul,h.fma]

#specNonCircular reference avoiding [interpret,sourceScalar,Tile.scalar,Tile.sourceRun]
#axiomsClean graph_lowering
#axiomsClean reference_real
#axiomsClean local_bridge
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.Native
