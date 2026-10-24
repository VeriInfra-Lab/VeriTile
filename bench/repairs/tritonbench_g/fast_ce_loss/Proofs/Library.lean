import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.HostForwardExecution

namespace VeriTile.Bench.Repairs.FastCeLoss.Host
open Source API Grid
open CrossEntropy1.Host (Step)
noncomputable section
set_option maxHeartbeats 4000000

def Descriptor.elementBytes (d : Descriptor) : Nat := if d.dtype=.fp32 then 4 else 8

def Descriptor.bounded (d : Descriptor) : Prop :=
  d.elementBytes*(d.tensor.offset+d.tensor.shape.prod)≤d.allocationBytes ∧
  d.allocationBytes<2^63 ∧ d.allocationBase.toNat+d.allocationBytes≤2^64 ∧
  d.pointer.toNat=d.allocationBase.toNat+d.elementBytes*d.tensor.offset

def Preserves {σ : Type} (live : σ → Nat → Prop) (before after : σ) : Prop :=
  ∀ storage,live before storage → live after storage

theorem Preserves.trans {σ : Type} {live : σ → Nat → Prop} {a b c : σ}
    (ab : Preserves live a b) (bc : Preserves live b c) : Preserves live a c :=
  fun storage h => bc storage (ab storage h)

/-- Ordinary successful allocator/view/dispatch metadata contracts. They contain
no loss, LSE, gradient, numerical output, or caller memory-frame predicate.
The signed address bounds are consequences of representable library allocation
sizes, not numerical gates or additional shape assumptions for this task. -/
structure Library {σ : Type} (runtime : Runtime σ) (live : σ → Nat → Prop) : Prop where
  allocate : ∀ shape dtype device before output after,
    runtime.allocate shape dtype device before=.ok output after →
    output.tensor.shape=shape ∧ output.dtype=dtype ∧ output.device=device ∧
    output.isNeg=Bool.false ∧ output.isConj=Bool.false ∧ output.contiguous=Bool.true ∧ output.bounded ∧
    live after output.tensor.storage ∧ Preserves live before after ∧
    (∀ storage,live before storage → output.tensor.storage≠storage)
  view : ∀ input shape before output after,
    runtime.view input shape before=.ok output after →
    output.tensor.shape=shape ∧ output.tensor.storage=input.tensor.storage ∧ output.tensor.offset=input.tensor.offset ∧
    output.dtype=input.dtype ∧ output.device=input.device ∧ output.isNeg=input.isNeg ∧ output.isConj=input.isConj ∧
    output.bounded ∧ Preserves live before after ∧
    (∀ rows vocab,shape=[rows,vocab] → 0<vocab → input.contiguous=Bool.true → output.strides=[(vocab : Int),1])
  expand : ∀ input rows before output after,
    runtime.expand input rows before=.ok output after →
    output.tensor.shape=[rows] ∧ output.tensor.storage=input.tensor.storage ∧ output.tensor.offset=input.tensor.offset ∧
    output.dtype=input.dtype ∧ output.device=input.device ∧ output.isNeg=input.isNeg ∧ output.isConj=input.isConj ∧
    output.strides=[0] ∧ Preserves live before after
  perform : ∀ call before value after,runtime.perform call before=.ok value after → Preserves live before after

variable {α σ : Type} {runtime : Runtime σ} {live : σ → Nat → Prop}

 theorem Library.allocate_preserves (laws : Library runtime live) {shape dtype device before after output}
    (success : Host.allocate (α:=α) runtime shape dtype device before=.ok output after) :
    Preserves live before.native after.native :=
  (laws.allocate _ _ _ _ _ _ (allocate_memory runtime _ _ _ _ _ _ success).2.2.2).2.2.2.2.2.2.2.2.1

 theorem Library.allocate_live (laws : Library runtime live) {shape dtype device before after output}
    (success : Host.allocate (α:=α) runtime shape dtype device before=.ok output after) :
    live after.native output.tensor.storage :=
  (laws.allocate _ _ _ _ _ _ (allocate_memory runtime _ _ _ _ _ _ success).2.2.2).2.2.2.2.2.2.2.1

 theorem Library.allocate_fresh (laws : Library runtime live) {shape dtype device before after output}
    (success : Host.allocate (α:=α) runtime shape dtype device before=.ok output after)
    (storage : Nat) (present : live before.native storage) : output.tensor.storage≠storage :=
  (laws.allocate _ _ _ _ _ _ (allocate_memory runtime _ _ _ _ _ _ success).2.2.2).2.2.2.2.2.2.2.2.2 storage present

 theorem Library.view_preserves (laws : Library runtime live) {input shape before after output}
    (success : Host.view (α:=α) runtime input shape before=.ok output after) :
    Preserves live before.native after.native :=
  (laws.view _ _ _ _ _ (view_memory runtime _ _ _ _ _ success).2.2.2).2.2.2.2.2.2.2.2.1

 theorem Library.perform_preserves (laws : Library runtime live) {request floats integers before after value}
    (success : Host.perform (α:=α) runtime request floats integers before=.ok value after) :
    Preserves live before.native after.native :=
  laws.perform _ _ _ _ (perform_memory runtime _ _ _ _ _ _ success).2.2.2

 theorem Library.expand_preserves (laws : Library runtime live) {input rows before after output}
    (success : Host.expand (α:=α) runtime input rows before=.ok output after) :
    Preserves live before.native after.native :=
  (laws.expand _ _ _ _ _ (expand_memory runtime _ _ _ _ _ success).2.2.2).2.2.2.2.2.2.2.2

 theorem Library.expand_coordinate (laws : Library runtime live) {input rows before after output}
    (success : Host.expand (α:=α) runtime input rows before=.ok output after) (row : Nat) :
    addr output.tensor (row*(output.strides.headD 0).toNat)=addr input.tensor 0 := by
  have h := laws.expand _ _ _ _ _ (expand_memory runtime _ _ _ _ _ success).2.2.2
  simp only [h.2.2.2.2.2.2.2.1,List.headD_cons,Int.toNat_zero,Nat.mul_zero]
  exact Prod.ext h.2.1 (by simpa only [addr,Nat.add_zero] using h.2.2.1)

 theorem FinishExecution.preserves {a : Arithmetic α} {stream s logits labels losses chunks label before after lse}
    (e : FinishExecution a runtime stream s logits labels losses chunks label before lse after)
    (laws : Library runtime live) : Preserves live before.native after.native := by
  cases e with
  | single same => exact fun _ h => h
  | chunked different lse allocated combined added final allocation combine addition mask =>
    exact (laws.allocate_preserves allocation).trans ((laws.perform_preserves combine).trans
      ((laws.perform_preserves addition).trans (laws.perform_preserves mask)))

 theorem CountExecution.preserves {stream args logits label before after count}
    (e : CountExecution (α:=α) runtime stream args logits label before count after)
    (laws : Library runtime live) : Preserves live before.native after.native := by
  cases e with
  | explicit => exact fun _ h => h
  | counted items count allocated final allocation operation =>
    exact (laws.allocate_preserves allocation).trans (laws.perform_preserves operation)

#axiomsClean Library.expand_coordinate
#axiomsClean FinishExecution.preserves
#axiomsClean CountExecution.preserves
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Host
