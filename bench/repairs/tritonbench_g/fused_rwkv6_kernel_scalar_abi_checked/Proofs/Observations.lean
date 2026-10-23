import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.Grid
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.Bytes
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.Specialization

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxHeartbeats 8000000

namespace Source
variable {α : Type} [Inhabited α]
def regionOf (r : RegionName) : FusedRwkv6Kernel.Region :=
  match r.name with
  | "q" => .q | "k" => .k | "v" => .v | "w" => .w | "u" => .u
  | "h0" => .h0 | "o" => .o | "ht" => .ht | "result" => .result | _ => .outside

def cells (m : Memory α) : Cells α := fun r b => .mk .real (m (regionOf r) b)
@[simp] theorem words_cells (m : Memory α) : words (cells m)=m := by
  funext r b
  cases r <;> rfl

/-- The actual launch branch. When T=0 and final-state output is disabled,
neither the shared evaluator nor the scalar binder is invoked. -/
def launch (M : Algebra α) (c : Shape) (scale : α) (m : Cells α) : Option (Cells α) :=
  if Launches c then runGrid M c scale (programs c) m else some m

theorem launch_memory (M : Algebra α) (c : Shape) (hc : Sizes c) (plan : Plans c)
    (reduction : ReductionMatches M c plan) (scale : α) (m : Cells α) :
    ∃ out,launch M c scale m=some out ∧ ∀ r b,words out r b=
      if Output c r b then target (ops M) c plan (inputs c (words m)) scale r b else words m r b := by
  by_cases live : Launches c
  · obtain ⟨out,ho,hw⟩:=grid_model M c plan reduction (programs c) scale m
    refine ⟨out,by simpa [launch,live] using ho,?_⟩
    intro r b
    rw [hw,complete_grid _ c hc]
  · have ht : c.T=0 := by by_contra h;exact live (Or.inl h)
    have hf : c.final=Bool.false := by cases h:c.final <;> simp_all [Launches]
    exact ⟨m,by simp [launch,live],by intro r b;simp [Output,ht,hf]⟩

theorem no_launch (M : Algebra α) (c : Shape) (scale : α) (m : Cells α)
    (ht : c.T=0) (hf : c.final=Bool.false) : launch M c scale m=some m := by
  simp [launch,Launches,ht,hf]
end Source

/-- Only the Tensor buffers passed to the kernel. The Torch sum result and
saved/autograd objects are outside this kernel contract. -/
structure Buffers (c : Shape) (x : Inputs) (base : FusedRwkv6Kernel.Region → Nat) where
  scratch : Checked.Tensor
  terminal : Checked.Tensor
  scratchMetadata : scratch.Valid ∧ scratch.metadata.shape=Checked.tensorShape c .o ∧
    scratch.metadata.device=(x.tensor .q).metadata.device ∧ scratch.metadata.dataPtr.toNat=base .o
  terminalMetadata : c.final=Bool.true → terminal.Valid ∧ terminal.metadata.shape=Checked.tensorShape c .ht ∧
    terminal.metadata.device=(x.tensor .q).metadata.device ∧ terminal.metadata.dataPtr.toNat=base .ht
  scratchChecked : Checked.PointerGuard c .o scratch
  terminalChecked : c.final=Bool.true → Checked.PointerGuard c .ht terminal

/-- FP32 observation through the actual Tensor data pointer, including its
storage offset. The offset argument is a contiguous logical element index. -/
def tensorWord (x : Checked.Tensor) (bytes : Bytes) (b : Nat) : BitVec 32 :=
  wordOf (fun j => bytes (x.metadata.dataPtr.toNat+4*b+j.val))

theorem tensor_word (c : Shape) (layout : Physical.Layout c) (x : Checked.Tensor)
    (r : FusedRwkv6Kernel.Region) (hp : x.metadata.dataPtr.toNat=layout.base r)
    (bytes : Bytes) (b : Nat) : tensorWord x bytes b=Physical.loadWords c layout bytes r b := by
  simp [tensorWord,Physical.loadWords,hp]

theorem partial_output (c : Shape) (z ik t f : Nat)
    (hz : z<c.Z) (hk : ik<c.NK) (ht : t<c.T) (hf : f<c.V) :
    Output c .o (partialAddress c z ik t f) := by
  left
  refine ⟨rfl,?_⟩
  have a : ik*c.Z+z<c.NK*c.Z := by nlinarith
  have b : (ik*c.Z+z)*c.T+t<c.NK*c.Z*c.T := by nlinarith
  unfold partialAddress
  nlinarith

theorem terminal_output (c : Shape) (z e f : Nat)
    (hs : c.final=Bool.true) (hz : z<c.Z) (he : e<c.K) (hf : f<c.V) :
    Output c .ht (stateAddress c z e f) :=
  Or.inr ⟨hs,rfl,state_address_bound c z e f hz he hf⟩

#axiomsClean Source.launch_memory
#axiomsClean Source.no_launch
#axiomsClean tensor_word
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked
