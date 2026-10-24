import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Native
import Mathlib.Data.Nat.Log
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Contract
import bench.repairs.tritonbench_g.square_matrix.Proofs.Geometry

namespace VeriTile.Bench.Repairs.SoftmaxTriton1
open VeriTile.Triton

/-- Allocation/descriptor registry contains metadata only. Distinct aliases may
share a region; arbitrary forged Tensor records receive no laws. -/
structure Layout where
  registered : Tensor → Prop
  buffer : RegionName → Native.Buffer
  legal : ∀ t, registered t → (buffer t.region).Legal ∧ Native.Descriptor (buffer t.region) t
  separate : ∀ t u, registered t → registered u → t.region≠u.region →
    (buffer t.region).device≠(buffer u.region).device ∨
    (buffer t.region).base+(buffer t.region).capacity≤(buffer u.region).base ∨
    (buffer u.region).base+(buffer u.region).capacity≤(buffer t.region).base

structure World (S : Type) where
  program : S
  live : RegionName → Prop

def Valid {S} (layout : Layout) (s : World S) (t : Tensor) : Prop :=
  s.live t.region ∧ layout.registered t

def worldAccess {S C V} (a : Access S C V) : Access (World S) C V :=
  ⟨fun s => a.cells s.program,a.decode⟩

def logical {S C V} (a : Access S C V) (neg : V → V) (s : World S) (t : Tensor) (r c : Nat) : V :=
  let v := a.read s.program t.region (t.index r c)
  if t.negative then neg v else v

structure Preserves {S C V} (a : Access S C V) (s t : World S) : Prop where
  live : ∀ r, s.live r → t.live r
  cells : ∀ r i, s.live r → a.cells t.program r i=a.cells s.program r i

theorem Preserves.refl {S C V} (a : Access S C V) (s : World S) : Preserves a s s :=
  ⟨fun _ h => h,fun _ _ _ => rfl⟩

theorem Preserves.trans {S C V} {a : Access S C V} {s t u : World S}
    (st : Preserves a s t) (tu : Preserves a t u) : Preserves a s u :=
  ⟨fun r h => tu.live r (st.live r h),fun r i h => (tu.cells r i (st.live r h)).trans (st.cells r i h)⟩

theorem Preserves.valid {S C V} {a : Access S C V} {s t : World S}
    (frame : Preserves a s t) (layout : Layout) (x : Tensor) (valid : Valid layout s x) : Valid layout t x :=
  ⟨frame.live _ valid.1,valid.2⟩

structure Resolved {S C V} (a : Access S C V) (neg : V → V) (layout : Layout)
    (x y : Tensor) (s t : World S) : Prop where
  shape : SameShape x y
  valid : Valid layout t y
  normal : y.negative=Bool.false
  frame : Preserves a s t
  coordinates : ∀ r c, r<x.rows → c<x.cols → logical a neg t y r c=logical a neg s x r c

structure Allocated {S C V} (a : Access S C V) (layout : Layout)
    (x y : Tensor) (s t : World S) : Prop where
  shape : SameShape x y
  valid : Valid layout t y
  normal : y.negative=Bool.false
  injective : y.Injective
  frame : Preserves a s t
  fresh : ¬s.live y.region

/-- No success, numerical output or full-grid behavior occurs in this local
library contract. Resolve and allocation are coordinate/metadata contracts;
compile is read-only; successful *individual* calls execute their given source
body. Error states are returned exactly; ordinary failure frames protect caller storage. -/
structure Runtime (S : Type) where
  resolve : Tensor → World S → Result (World S) Tensor
  allocate : Tensor → World S → Result (World S) Tensor
  compile : Tensor → Tensor → Nat → Nat → World S → Result (World S) Unit
  invoke : Tensor → Tensor → Nat → Nat → World S → Result (World S) Unit

/-- A local invocation may change its output even on failure. All other
regions are read-only and existing allocations remain live. -/
structure OutputFrame {S C V} (a : Access S C V) (y : Tensor) (s t : World S) : Prop where
  live : ∀ r, s.live r → t.live r
  cells : ∀ r i, r≠y.region → a.cells t.program r i=a.cells s.program r i

def Result.state {S A} : Result S A → S
  | .ok _ s => s
  | .error _ s => s

structure RuntimeContract {S C V} (a : Access S C V) (neg : V → V) (layout : Layout)
    (evaluate : Tensor → Tensor → Nat → Nat → S → Option S) (runtime : Runtime S) : Prop where
  resolve : ∀ x s y t, Valid layout s x → runtime.resolve x s=.ok y t →
    Resolved a neg layout x y s t
  allocate : ∀ x s y t, Valid layout s x → runtime.allocate x s=.ok y t →
    Allocated a layout x y s t
  compile : ∀ x y B warps s t, Valid layout s x → Valid layout s y →
    runtime.compile x y B warps s=.ok () t → t=s
  invoke : ∀ x y B p s t, Valid layout s x → Valid layout s y →
    x.negative=Bool.false → y.negative=Bool.false → y.Injective →
    p<x.rows → runtime.invoke x y B p s=.ok () t →
    evaluate x y B p s.program=some t.program ∧ t.live=s.live
  local_abi : ∀ x y B p s t, Valid layout s x → Valid layout s y →
    x.negative=Bool.false → y.negative=Bool.false → y.Injective →
    p<x.rows → runtime.invoke x y B p s=.ok () t → p<2^31 ∧ B≤2^31
  resolve_error : ∀ x s e t, Valid layout s x → runtime.resolve x s=.error e t →
    Preserves a s t
  allocate_error : ∀ x s e t, Valid layout s x → runtime.allocate x s=.error e t →
    Preserves a s t
  compile_error : ∀ x y B warps s e t, Valid layout s x → Valid layout s y →
    runtime.compile x y B warps s=.error e t → Preserves a s t
  invoke_error : ∀ x y B p s e t, Valid layout s x → Valid layout s y →
    x.negative=Bool.false → y.negative=Bool.false → y.Injective →
    p<x.rows → runtime.invoke x y B p s=.error e t → OutputFrame a y s t

/-- Actual wrapper sequence, including allocation before the empty fast return.
The local-call grid is explicit. Its relation to a global asynchronous device
launch is a separately named deferred runtime obligation. -/
def wrapper {S} (rt : Runtime S) (x : Tensor) (s : World S) : Result (World S) Tensor :=
  (rt.resolve x s).bind fun v a =>
    let B := blockSize x.cols
    (rt.allocate v a).bind fun y b =>
      if x.rows=0 ∨ x.cols=0 then .ok y b else
        (rt.compile v y B (numWarps B) b).bind fun _ c =>
          (runRows (rt.invoke v y B) (List.range x.rows) c).bind fun _ d => .ok y d

theorem wrapper_resolve_failure {S} (rt : Runtime S) (x : Tensor) (s t : World S) (e : String)
    (failure : rt.resolve x s=.error e t) : wrapper rt x s=.error e t := by
  simp [wrapper,failure,Result.bind]

theorem caller_bytes {S C V} (a : Access S C V) (encode : C → Nat → BitVec 8)
    (layout : Layout) (s t : World S) (frame : Preserves a s t)
    (u : Tensor) (valid : Valid layout s u) (byte : Nat) :
    backingByte (worldAccess a) encode t u.region byte=
      backingByte (worldAccess a) encode s u.region byte := by
  unfold backingByte worldAccess
  rw [frame.cells _ _ valid.1]

#axiomsClean blockSize_bounds
#axiomsClean caller_bytes
end VeriTile.Bench.Repairs.SoftmaxTriton1
