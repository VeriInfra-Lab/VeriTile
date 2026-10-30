import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.BF16Physical
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Contract
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionKernelAligned Full Whole
noncomputable section

/-- Observations of Python's sorted output strides and actual Tensor storage.
The list is the reversed ascending-stride order used by the span scan. -/
structure Observations where
  order : List (Fin 4)
  pointer : Buffer → Nat
  storageBase : Buffer → Nat
  storageBytes : Buffer → Nat
  ordinaryCUDA : Bool

/-- Actual checked numerical/layout/address conditions. It contains no source
execution, floating output, native graph or mathematical target. -/
def InputContract (p : Parameters) (a : Observations) : Prop :=
  a.ordinaryCUDA=Bool.true ∧ Selected p ∧ p.length≤64*p.blockN ∧
  (Nonempty p →
    a.order.Perm (List.finRange 4) ∧
    Layout.Guarded (extentAt p) (strideAt (oStrides p)) a.order ∧
    p.batches*p.heads≤65535 ∧
    (∀ s∈[qStrides p,kStrides p,vStrides p,oStrides p,bStrides p],
      0<s.batch ∧ 0<s.head ∧ 0<s.row ∧ 0<s.dim) ∧
    (∀ s∈[qStrides p,kStrides p,vStrides p,oStrides p,bStrides p],
      ∀ v∈[p.batches,p.heads,p.length,p.head,s.batch,s.head,s.row,s.dim],v<2147483648) ∧
    (∀ s∈[qStrides p,kStrides p,vStrides p,oStrides p],strideEnvelope p s<2147483648) ∧
    biasEnvelope p<2147483648 ∧
    (∀ r,a.pointer r%2=0 ∧ a.pointer r+2*formedEnvelope p r+2<2^64) ∧
    (∀ r s,r≠s → a.storageBase r+a.storageBytes r≤a.storageBase s ∨
      a.storageBase s+a.storageBytes s≤a.storageBase r))

def check (p : Parameters) (a : Observations) : Bool := by
  classical
  exact decide (InputContract p a)

theorem check_iff (p : Parameters) (a : Observations) :
    check p a=Bool.true ↔ InputContract p a := by
  classical
  simp [check]

def guard (p : Parameters) (a : Observations) (h : InputContract p a) : Guard p where
  selected := h.2.1
  length := h.2.2.1
  layout := fun ne=>⟨a.order,(h.2.2.2 ne).1,(h.2.2.2 ne).2.1⟩
  launchY := fun ne=>(h.2.2.2 ne).2.2.1
  positive := fun ne=>(h.2.2.2 ne).2.2.2.1
  arguments := fun ne=>(h.2.2.2 ne).2.2.2.2.1
  signed := fun ne=>(h.2.2.2 ne).2.2.2.2.2.1
  biasSigned := fun ne=>(h.2.2.2 ne).2.2.2.2.2.2.1

/-- Ordinary Tensor API facts: the logical data pointer and remaining element
capacity lie in that tensor's actual full storage interval. -/
structure StorageAPI (p : Parameters) (storage : TensorStorage p) (a : Observations) where
  arena : Nat
  arena64 : arena≤2^64
  bounded : ∀ r,a.pointer r+2*extent storage r≤arena
  begins : ∀ r,a.storageBase r≤a.pointer r
  fits : ∀ r,a.pointer r+2*extent storage r≤a.storageBase r+a.storageBytes r

def placement (p : Parameters) (storage : TensorStorage p) (a : Observations)
    (api : StorageAPI p storage a) (h : InputContract p a) : Placement p storage where
  arena := api.arena
  base := a.pointer
  arena64 := api.arena64
  bounded := api.bounded
  formed64 := fun ne r=>((h.2.2.2 ne).2.2.2.2.2.2.2.1 r).2
  apart := by
    intro ne r hr i hi j hj
    have sep := (h.2.2.2 ne).2.2.2.2.2.2.2.2 r .output hr
    have r0:=api.begins r;have r1:=api.fits r
    have o0:=api.begins .output;have o1:=api.fits .output
    change j<2*extent storage .output at hj
    omega

theorem aligned (p : Parameters) (a : Observations) (h : InputContract p a)
    (ne : Nonempty p) (r : Buffer) (off : Nat) : (a.pointer r+2*off)%2=0 := by
  have hp := ((h.2.2.2 ne).2.2.2.2.2.2.2.1 r).1
  omega

#specNonCircular InputContract avoiding [VeriTile.Triton.FP.Structural.exec,Whole.invoke]
#axiomsClean check_iff
#axiomsClean aligned
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Contract
