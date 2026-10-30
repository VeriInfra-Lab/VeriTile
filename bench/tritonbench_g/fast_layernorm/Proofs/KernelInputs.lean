import bench.tritonbench_g.fast_layernorm.Proofs.Contract
import bench.tritonbench_g.fast_layernorm.Proofs.KernelReference
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.FastLayernorm.KernelInputs
open VeriTile.Triton FP.Structural VeriTile.Torch.FP32
noncomputable section
open scoped Classical

/-- A source input view and its original logical values. For direct forward
views the constructor below derives the value equation from Tensor API laws.
For dY, the source's own contiguous() may supply the elementwise API equation.
No execution, output, allocation freshness, or arithmetic law is a field. -/
structure ReadView (α : Type) (read : Nat → α) where
  tensor : Tensor
  laws : TensorLaws tensor
  checked : Contract.TensorContract tensor.metadata Bool.true
  values : Nat → α
  value : ∀ i,i<tensor.metadata.numel →
    read ((dataView tensor laws (checked.dense rfl)).address i)=values i

def direct {α : Type} (read : Nat → α) (t : Tensor) (laws : TensorLaws t)
    (checked : Contract.TensorContract t.metadata Bool.true) : ReadView α read where
  tensor := t
  laws := laws
  checked := checked
  values := fun i => read (t.storage.address (t.physicalIndex i))
  value := by
    intro i hi
    apply congrArg read
    rw [laws.contiguous (checked.dense rfl) i hi]
    simp only [DataView.address,DataView.base,dataView,if_neg (by omega : t.metadata.numel≠0),Storage.address]
    omega

theorem direct_fp32_values (read : Nat → Word) (t : Tensor) (laws : TensorLaws t)
    (checked : Contract.TensorContract t.metadata Bool.true) (i : Nat) :
    (direct read t laws checked).values i=logicalValue read t laws i := by
  simp [direct,logicalValue,checked.noNeg]

/-- Exactly the value guarantee of the existing source normalization of dY.
The original tensor is allowed to have expanded/noncontiguous strides. -/
def normalized {α : Type} (read : Nat → α) (original dense : Tensor)
    (laws : TensorLaws dense) (checked : Contract.TensorContract dense.metadata Bool.true)
    (sameCount : dense.metadata.numel=original.metadata.numel)
    (contiguousValues : ∀ i,i<original.metadata.numel →
      read ((dataView dense laws (checked.dense rfl)).address i)=
        read (original.storage.address (original.physicalIndex i))) : ReadView α read where
  tensor := dense
  laws := laws
  checked := checked
  values := fun i => read (original.storage.address (original.physicalIndex i))
  value := fun i hi => contiguousValues i (by simpa [sameCount] using hi)

def forwardViews {α : Type} (read : Nat → α) (x w : Tensor) (b : Option Tensor)
    (lx : TensorLaws x) (lw : TensorLaws w) (lb : ∀ t,b=some t → TensorLaws t)
    (eps : Contract.Epsilon) (h : Contract.InputContract x.metadata w.metadata (b.map Tensor.metadata) eps) :
    RegionName → Option (ReadView α read) := fun r =>
  if r="x" then some (direct read x lx h.input)
  else if r="w" then some (direct read w lw h.weight)
  else if r="b" then b.pmap (fun t ht => direct read t (lb t ht) (h.biasTensor t.metadata (by simp [ht])))
    (fun t ht => ht)
  else none

variable {α : Type} [Inhabited α] {read : Nat → α}
def physical (views : RegionName → Option (ReadView α read)) (u : State α) : State α :=
  {u with mem:=fun r i => match views r with
    | none => u.mem r i
    | some v => if i<v.tensor.metadata.numel then
        .mk .real (read ((dataView v.tensor v.laws (v.checked.dense rfl)).address i)) else u.mem r i}
def logical (views : RegionName → Option (ReadView α read)) (u : State α) : State α :=
  {u with mem:=fun r i => match views r with
    | none => u.mem r i
    | some v => if i<v.tensor.metadata.numel then .mk .real (v.values i) else u.mem r i}
def value (views : RegionName → Option (ReadView α read)) (u : State α) (r : RegionName) (i : Nat) : α :=
  match views r with
  | none => (u.mem r i).read .real
  | some v => if i<v.tensor.metadata.numel then v.values i else (u.mem r i).read .real

theorem physical_eq_logical (views : RegionName → Option (ReadView α read)) (u : State α) :
    physical views u=logical views u := by
  have hm : (physical views u).mem=(logical views u).mem := by
    funext r i
    cases hv : views r with
    | none => simp [physical,logical,hv]
    | some v =>
      by_cases hi : i<v.tensor.metadata.numel
      · simp [physical,logical,hv,hi,v.value i hi]
      · simp [physical,logical,hv,hi]
  exact congrArg (fun m => ({u with mem:=m} : State α)) hm

theorem logical_read (views : RegionName → Option (ReadView α read)) (u : State α) (r : RegionName) (i : Nat) :
    ((logical views u).mem r i).read .real=value views u r i := by
  change (match views r with
    | none => u.mem r i
    | some v => if i<v.tensor.metadata.numel then Cell.mk .real (v.values i) else u.mem r i).read .real=_
  unfold value
  cases hv : views r with
  | none => rfl
  | some v =>
    by_cases hi : i<v.tensor.metadata.numel <;> simp only [hi,ite_true,ite_false,Cell.read_same]

theorem actual_pointers (views : RegionName → Option (ReadView α read))
    (r : RegionName) (v : ReadView α read) (hv : views r=some v) (nonempty : 0<v.tensor.metadata.numel) :
    BitVec.ofNat 64 ((dataView v.tensor v.laws (v.checked.dense rfl)).address 0)=v.tensor.dataPtr :=
  actual_pointer v.tensor v.laws (v.checked.dense rfl) nonempty

theorem native_address (v : ReadView α read) (i : Nat) (active : i<v.tensor.metadata.numel) :
    (BitVec.ofNat 64 ((dataView v.tensor v.laws (v.checked.dense rfl)).address i)).toNat=
      (dataView v.tensor v.laws (v.checked.dense rfl)).address i := by
  have span := (dataView v.tensor v.laws (v.checked.dense rfl)).spanFits
  simp only [BitVec.toNat_ofNat]
  apply Nat.mod_eq_of_lt
  simp only [DataView.address]
  change (dataView v.tensor v.laws (v.checked.dense rfl)).base+4*v.tensor.metadata.numel≤2^64 at span
  omega

def Contains (views : RegionName → Option (ReadView α read)) (r : RegionName) (i : Nat) : Prop :=
  ∃ v,views r=some v ∧ i<v.tensor.metadata.numel
def ForwardReads (views : RegionName → Option (ReadView α read)) (pid N : Nat) (bias : Bool) : Prop :=
  (∀ j,j<N → Contains views "x" (pid*N+j)) ∧
  (∀ j,j<N → Contains views "w" j) ∧
  (bias=Bool.true → ∀ j,j<N → Contains views "b" j)
def BackwardReads (views : RegionName → Option (ReadView α read)) (pid N : Nat) : Prop :=
  (∀ j,j<N → Contains views "x" (pid*N+j) ∧ Contains views "dy" (pid*N+j) ∧ Contains views "w" j) ∧
  Contains views "r" pid ∧ Contains views "mu" pid
def AffineReads (views : RegionName → Option (ReadView α read)) (rows N pid : Nat) : Prop :=
  ∀ row,row<rows → Contains views "r" row ∧ Contains views "mu" row ∧
    (∀ j : Fin 128,pid*128+j.val<N →
      Contains views "x" (row*N+(pid*128+j.val)) ∧ Contains views "dy" (row*N+(pid*128+j.val)))

theorem active_physical_load (views : RegionName → Option (ReadView α read)) (u : State α)
    (r : RegionName) (i : Nat) (v : ReadView α read) (hv : views r=some v) (hi : i<v.tensor.metadata.numel) :
    ((physical views u).mem r i).read .real=
      read ((dataView v.tensor v.laws (v.checked.dense rfl)).address i) := by
  change (match views r with
    | none => u.mem r i
    | some v => if i<v.tensor.metadata.numel then Cell.mk .real
        (read ((dataView v.tensor v.laws (v.checked.dense rfl)).address i)) else u.mem r i).read .real=_
  rw [hv]
  simp only [hi,ite_true,Cell.read_same]

theorem checked_forward_reads (read : Nat → α) (x w : Tensor) (b : Option Tensor)
    (lx : TensorLaws x) (lw : TensorLaws w) (lb : ∀ t,b=some t → TensorLaws t)
    (eps : Contract.Epsilon) (h : Contract.InputContract x.metadata w.metadata (b.map Tensor.metadata) eps)
    (pid : Nat) (rowBound : pid<x.metadata.numel/Contract.width x.metadata) :
    ForwardReads (forwardViews read x w b lx lw lb eps h) pid (Contract.width x.metadata) b.isSome := by
  have xbound (j : Nat) (hj : j<Contract.width x.metadata) :
      pid*Contract.width x.metadata+j<x.metadata.numel := by
    have cap := Nat.div_mul_le_self x.metadata.numel (Contract.width x.metadata)
    have low := Nat.mul_le_mul_right (Contract.width x.metadata) (Nat.succ_le_of_lt rowBound)
    nlinarith
  refine ⟨?_,?_,?_⟩
  · intro j hj
    exact ⟨direct read x lx h.input,by simp [forwardViews],xbound j hj⟩
  · intro j hj
    exact ⟨direct read w lw h.weight,by simp [forwardViews],lt_of_lt_of_le hj h.weightCount⟩
  · intro hb j hj
    cases b with
    | none => contradiction
    | some t =>
      refine ⟨direct read t (lb t rfl) (h.biasTensor t.metadata rfl),?_,?_⟩
      · simp [forwardViews]
      · exact lt_of_lt_of_le hj (h.biasCount t.metadata rfl)

#axiomsClean physical_eq_logical
#axiomsClean actual_pointers
#axiomsClean native_address
#axiomsClean checked_forward_reads
#axiomsClean active_physical_load
end
end VeriTile.Bench.Repairs.FastLayernorm.KernelInputs
