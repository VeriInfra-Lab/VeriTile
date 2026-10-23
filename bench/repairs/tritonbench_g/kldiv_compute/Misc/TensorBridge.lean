import bench.repairs.tritonbench_g.kldiv_compute.Correct
import bench.repairs.tritonbench_g.kldiv_compute.Misc.NativeCorrect
import bench.repairs.tritonbench_g.kldiv_compute.Proofs.KernelExecution

/- The Tensor application and raw grid interfaces reuse Program.complete_correct:
Program.grid_complete composes its exact memory effects, while Kernel.complete
instantiates the same effects through KernelBinding.physicalState and preserves
all physical pointer, control, byte and exterior-frame observations. -/
namespace VeriTile.Bench.Repairs.KldivCompute.Kernel.TensorBridge.Real
open VeriTile.Triton
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Contiguous
noncomputable section
set_option autoImplicit false

/-- Actual complete typed kernel execution over the same physical Tensor
bindings as the native result. Every logical output has the independent KL
term; the complete raw/observation frame and native address/control facts are
included. Arbitrary Real values require no finite-byte codec. -/
theorem correctness {M β : Type} (ops : Ops M ℝ β) (x y output : Tensor)
    (undefined : Undefined ℝ) (domain : Domain x y output) (schedule : Index.Schedule x.metadata.numel) (initial : M) :
    ∃ final, Post ops Neg.neg Math.algebra undefined x y output schedule
      (fun i => mathematical (logical ops Neg.neg initial x i) (logical ops Neg.neg initial y i))
      initial final := by
  apply complete ops Neg.neg Math.algebra undefined x y output domain schedule initial
  intro i _
  exact Math.scalar_formula _ _

#axiomsClean correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.Kernel.TensorBridge.Real

namespace VeriTile.Bench.Repairs.KldivCompute.Kernel.TensorBridge.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Contiguous
noncomputable section
set_option autoImplicit false

abbrev Word := FP.Value .fp32
abbrev Byte := BitVec 8
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

open Kernel.FPEquiv

/-- Only native words are connected to four finite little-endian bytes. -/
structure Codec {M : Type} (ops : Ops M Word Byte) : Prop where
  bytes : ∀ s k b,b<4 → ops.observe s k.owner (k.address+b) =
    (ops.read s k).bits.extractLsb' (8*b) 8

def OutputBytes {M : Type} (ops : Ops M Word Byte) (output : Tensor) (n : Nat)
    (expected : Nat → Word) (after : M) : Prop :=
  ∀ i,i<n → ∀ byte,byte<4 → ops.observe after (key output i).owner ((key output i).address+byte) =
    (expected i).bits.extractLsb' (8*byte) 8

/-- Complete typed source versus the independent divide/log/multiply graph,
with successful execution, every output word/byte, the full physical memory
frame and actual Tensor/native pointer binding. No numerical transformation,
accuracy claim or deferred primitive experiment is involved. -/
theorem equivalence {M : Type} (ops : Ops M Word Byte) (codec : Codec ops)
    (p : Primitives) (undefined : Undefined Word) (x y output : Tensor) (domain : Domain x y output)
    (schedule : Index.Schedule x.metadata.numel) (initial : M) :
    ∃ final, Post ops (fun w => w.neg) (algebra p) undefined x y output schedule
      (fun i => reference p (logical ops (fun w => w.neg) initial x i)
        (logical ops (fun w => w.neg) initial y i)) initial final ∧
      OutputBytes ops output x.metadata.numel
        (fun i => reference p (logical ops (fun w => w.neg) initial x i)
          (logical ops (fun w => w.neg) initial y i)) final := by
  obtain ⟨final,result⟩ := complete ops (fun w => w.neg) (algebra p) undefined x y output domain schedule initial
    (fun i => reference p (logical ops (fun w => w.neg) initial x i)
      (logical ops (fun w => w.neg) initial y i)) (fun _ _ => scalar_reference p _ _)
  refine ⟨final,result,?_⟩
  intro i hi byte hb
  have value := result.values i hi
  simp only [logical,domain.outputNoNeg,Bool.false_eq_true,if_false] at value
  rw [codec.bytes _ _ _ hb,value]
  rfl

#axiomsClean equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.Kernel.TensorBridge.Native
