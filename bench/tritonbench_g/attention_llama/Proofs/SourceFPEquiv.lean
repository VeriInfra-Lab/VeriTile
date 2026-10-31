import bench.tritonbench_g.attention_llama.Sources.Kernels
import bench.tritonbench_g.attention_llama.Proofs.NativeLocal
import bench.tritonbench_g.attention_llama.Proofs.HostGrid
import bench.tritonbench_g.attention_llama.Proofs.FPSourceEquiv
import bench.tritonbench_g.attention_llama.Proofs.FP8SourceEquiv
import VeriTile.Meta.StatementAudit

/-! Native precision contract, independent of Correct.lean and its proof stack.

Source: the separately named repair, Q/Out FP16, K/V FP16 or signed int8 E5M2.
The recorded Triton 3.7.1 CUDA SM80 lowering uses 64x64 tiles, four warps for
D ≤ 64 and eight for D=128, with two stages. The supported source dimensions
are 16/32/64/128. QK is an FP16-input/FP32-output zero-seeded tensor dot.
Normalizer update is FMA(old_l, alpha, sum(p)); the separately rounded old_l
product is retained for rescaling. PV is a seeded tensor dot. Probabilities
and outputs use round-to-nearest FP16 casts. tl.exp lowers to FP32 multiply
by log2(e) then ex2.approx.f32; reciprocal uses div.full.f32. The recorded reduction trees are explicit scalar add/max nodes, retaining
lane-specific XOR2/XOR1 results and the D128 shared cross-warp XOR4/2/1
schedule. QK composes D/16 fixed K16 ISA calls from zero; PV composes four
from the separately rescaled accumulator. Each ISA lane port has exactly
16 FP16 values on each side plus its FP32 seed. No internal MMA rounding
order is invented. The E5M2 branch retains its actual kWidth=4 fragment
permutation (including the D16 special case), after E5M2-to-FP16 conversion.
Exponential lowering is separate multiply/ex2 nodes.

The ordinary compiler contract interprets the recorded emitted graph with
these primitives. The theorem below transports that graph under exact LOCAL
correspondence at its reached primitive operands. No hardware-to-reference
local law is registered or discharged by this entry; those numerical laws are
deferred. NativeLocal.Local lists them individually, including FMA, fixed K16 MMA,
both half casts, E5M2 decode, scalar reduction nodes and exp lowering. It contains no
attention result or successful-execution equation. The source-bound carry-first equivalences below also retain the complete
typed sources and failures; their sole guarded rewrite is finite add_commute.
The generic native theorem is not
a claim of GPU conformance or exact equality with real-number attention.
-/
namespace VeriTile.Bench.Repairs.AttentionLlama.FPEquivalence
open VeriTile Triton
open scoped VeriTile.Spec
open HostGrid HostMemory MaskCoverage

def supportedHead (D : Nat) : Prop := D = 16 ∨ D = 32 ∨ D = 64 ∨ D = 128
def numWarps (D : Nat) : Nat := if D ≤ 64 then 4 else 8
def numStages : Nat := 2

/-- Both precision branches point at the complete repaired typed source. -/
def source (fp8 : Bool) (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      heads H N position D : Nat) (causal : Bool) : ComputeKernel :=
  if fp8 then
    TypedSource.repairedFP8 Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son heads H N position 64 64 D causal
  else
    TypedSource.repaired Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son heads H N position 64 64 D causal

def output {α : Type} {B heads H D : Nat} (A : NativeGraph.Ops α)
    (input : Index B heads H D → SourceGraph.Inputs α D) (scale : α)
    (fp8 causal : Bool) (N position : Nat) (i : Index B heads H D) : α :=
  NativeGraph.returnedWord A (input i) scale fp8 causal N position
    (i.query.val / 64) (i.query.val % 64) i.channel

/-- The input coordinate map and launch geometry determine this exact
primitive domain. Masked keys select zero/negative-infinity explicitly. -/
def numericalDomain {α β : Type} {B heads H D : Nat}
    (A : NativeGraph.Ops α) (R : NativeGraph.Ops β) (f : α → β)
    (input : Index B heads H D → SourceGraph.Inputs α D) (scale : α)
    (fp8 causal : Bool) (N position : Nat) : Prop :=
  ∀ i : Index B heads H D, NativeLocal.Local A R f
    (NativeLocal.outputTerm (input i) scale fp8 causal N position
      (i.query.val / 64) (i.query.val % 64) i.channel)

/-- Primitive-level conformance composes through the entire recorded graph,
actual masked grid stores, tensor readback, and each memory complement frame. -/
theorem native_graph_transport {α β : Type} {B heads H D : Nat}
    (A : NativeGraph.Ops α) (R : NativeGraph.Ops β) (f : α → β)
    (input : Index B heads H D → SourceGraph.Inputs α D) (scale : α)
    (fp8 causal : Bool) (N position : Nat)
    (hlocal : numericalDomain A R f input scale fp8 causal N position)
    (address : Index B heads H D → Nat) (hinj : Function.Injective address)
    (memory : Nat → α) (referenceMemory : Nat → β) :
    (∀ i,
      f ((returnTensor address
        (storeAll address (output A input scale fp8 causal N position) memory
          (schedule B heads H D))).read i) =
      (returnTensor address
        (storeAll address (output R (fun i => (input i).map f) (f scale) fp8 causal N position)
          referenceMemory (schedule B heads H D))).read i) ∧
    (∀ location, (∀ i, location ≠ address i) →
      storeAll address (output A input scale fp8 causal N position) memory
        (schedule B heads H D) location = memory location) ∧
    (∀ location, (∀ i, location ≠ address i) →
      storeAll address (output R (fun i => (input i).map f) (f scale) fp8 causal N position)
        referenceMemory (schedule B heads H D) location = referenceMemory location) := by
  obtain ⟨ha,haf⟩ := logical_result_and_frame address
    (output A input scale fp8 causal N position) hinj memory
  obtain ⟨hr,hrf⟩ := logical_result_and_frame address
    (output R (fun i => (input i).map f) (f scale) fp8 causal N position) hinj referenceMemory
  refine ⟨?_,haf,hrf⟩
  intro i
  rw [ha,hr]
  exact NativeLocal.output_transport A R f (input i) scale fp8 causal N position
    (i.query.val / 64) (i.query.val % 64) i.channel (hlocal i)

/-- Logical input views are shared across channels of a query, as in the
source. These are coordinate bindings, not attention-result premises. -/
structure TensorInputs (α : Type) (B heads H D : Nat) where
  query : Fin B → Fin heads → Fin H → Fin D → α
  halfKey : Fin B → Fin heads → Nat → Fin D → α
  halfValue : Fin B → Fin heads → Nat → Fin D → α
  byteKey : Fin B → Fin heads → Nat → Fin D → UInt8
  byteValue : Fin B → Fin heads → Nat → Fin D → UInt8

def TensorInputs.at {α : Type} {B heads H D : Nat} (x : TensorInputs α B heads H D)
    (i : Index B heads H D) : SourceGraph.Inputs α D :=
  ⟨x.query i.batch i.head i.query, x.halfKey i.batch i.head,
    x.halfValue i.batch i.head, x.byteKey i.batch i.head, x.byteValue i.batch i.head⟩

/-- Actual Out offset from the host-provided empty_like strides. -/
def tensorAddress {B heads H D : Nat} (base soz soh som son : Nat)
    (i : Index B heads H D) : Nat :=
  base + (i.batch.val*soz + i.head.val*soh + i.query.val*som + i.channel.val*son)

/-- The compiler interface binds the pinned source above to the audited
SM80 graph. Under its ordinary correctness contract, this is the complete
native returned tensor; numerical correspondence is only required locally.
The compiler contract is external, not an assumed output equation in Lean. -/
theorem attention_native_tensor_equiv {α β : Type} {B heads H D : Nat}
    (A : NativeGraph.Ops α) (R : NativeGraph.Ops β) (f : α → β)
    (input : TensorInputs α B heads H D) (scale : α)
    (_hD : supportedHead D) (fp8 causal : Bool) (N position : Nat)
    (hlocal : numericalDomain A R f input.at scale fp8 causal N position)
    (base soz soh som son : Nat)
    (hinj : Function.Injective (tensorAddress (B:=B) (heads:=heads) (H:=H) (D:=D) base soz soh som son))
    (memory : Nat → α) (referenceMemory : Nat → β) :
    let address := tensorAddress (B:=B) (heads:=heads) (H:=H) (D:=D) base soz soh som son
    (∀ i,
      f ((returnTensor address
        (storeAll address (output A input.at scale fp8 causal N position) memory
          (schedule B heads H D))).read i) =
      (returnTensor address
        (storeAll address (output R (fun i => (input.at i).map f) (f scale) fp8 causal N position)
          referenceMemory (schedule B heads H D))).read i) ∧
    (∀ location, (∀ i, location ≠ address i) →
      storeAll address (output A input.at scale fp8 causal N position) memory
        (schedule B heads H D) location = memory location) ∧
    (∀ location, (∀ i, location ≠ address i) →
      storeAll address (output R (fun i => (input.at i).map f) (f scale) fp8 causal N position)
        referenceMemory (schedule B heads H D) location = referenceMemory location) := by
  exact native_graph_transport A R f input.at scale fp8 causal N position hlocal
    (tensorAddress base soz soh som son) hinj memory referenceMemory

#print axioms attention_native_tensor_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttentionLlama.FPEquivalence
