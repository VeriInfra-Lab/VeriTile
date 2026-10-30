import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.RuntimeModel
import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.InputContract
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.RbeTritonTransform.Binding
open VeriTile Triton FP.Structural Torch.FP32
noncomputable section
open Classical
set_option maxHeartbeats 5000000

/-- One shared Tensor supplies its metadata, logical indices, backing storage
and observed data pointer. No copied/materialized input is introduced. -/
structure Tensor where
  actual : Torch.FP32.Tensor
  laws : TensorLaws actual
  contiguous : actual.metadata.isContiguous=Bool.true
  notNeg : actual.metadata.isNeg=Bool.false

def Tensor.view (t : Tensor) := dataView t.actual t.laws t.contiguous
def Tensor.size (t : Tensor) := t.actual.metadata.numel
def Tensor.read {α : Type} (t : Tensor) (memory : Nat → α) (i : Nat) :=
  memory (t.actual.storage.address (t.actual.physicalIndex i))

theorem Tensor.read_view (t : Tensor) (memory : Nat → α) (i : Nat) (hi : i<t.size) :
    t.read memory i=memory (t.view.address i) := by
  unfold Tensor.read
  rw [t.laws.contiguous t.contiguous i hi]
  congr 1
  have nonempty : t.actual.metadata.numel≠0 := by change t.size≠0; omega
  simp [Tensor.view,dataView,DataView.address,DataView.base,nonempty,Storage.address,Nat.mul_add,Nat.add_assoc]

theorem Tensor.read_logical (t : Tensor) (memory : Nat → Word) (i : Nat) :
    t.read memory i=logicalValue memory t.actual t.laws i := by
  simp [Tensor.read,logicalValue,t.notNeg]

/-- Ordinary shape/argument binding, independent of any execution or result. -/
structure Input (x : Tensor) (r : Request) : Prop where
  shape : x.actual.metadata.shape=[r.batches,r.rows,r.columns]
  admitted : Contract.InputContract x.actual.metadata ⟨Bool.true,r.position⟩

theorem Input.size (input : Input x r) : x.size=r.size := by
  simp [Tensor.size,Torch.TensorMetadata.numel,input.shape,Request.size,Nat.mul_assoc]

/-- Fresh output allocation at kernel entry. Full input backing separation
also protects bytes outside the input view, including a nonzero storage offset. -/
structure Output (x y : Tensor) : Prop where
  shape : y.actual.metadata.shape=x.actual.metadata.shape
  device : y.actual.metadata.device=x.actual.metadata.device
  offset : y.actual.storageOffset=0
  aligned : 0<y.size → y.actual.dataPtr.toNat%4=0
  fresh : y.view.base+4*y.size≤x.actual.storage.base ∨
    x.actual.storage.base+4*x.actual.storage.elements≤y.view.base

theorem Output.size (out : Output x y) : y.size=x.size := by
  simp only [Tensor.size,Torch.TensorMetadata.numel,out.shape]

def initial {α : Type} [Inhabited α] (x y : Tensor) (memory : Nat → α) : State α where
  mem := fun r i => .mk .fp32 (memory ((if r="X" then x.view else y.view).address i))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun d => if d=0 then x.actual.metadata.shape[0]?.getD 0
    else if d=1 then ((x.actual.metadata.shape[1]?.getD 0+1)/2)*
      ((x.actual.metadata.shape[2]?.getD 0+1023)/1024) else 1
  undef := fun d _ _ => defaultValue d

def footprint (y : Tensor) (address : Nat) : Prop :=
  y.view.base≤address ∧ address<y.view.base+4*y.size

def commit {α : Type} [Inhabited α] (y : Tensor) (memory : Nat → α) (s : State α) : Nat → α :=
  fun address => if footprint y address ∧ (address-y.view.base)%4=0 then
    (s.mem "Y" ((address-y.view.base)/4)).read .fp32 else memory address

def reference {α : Type} (y : Tensor) (memory : Nat → α) (values : Nat → α) : Nat → α :=
  fun address => if footprint y address ∧ (address-y.view.base)%4=0 then
    values ((address-y.view.base)/4) else memory address

theorem commit_value {α : Type} [Inhabited α] (y : Tensor) (memory : Nat → α)
    (s : State α) (i : Nat) (hi : i<y.size) :
    commit y memory s (y.view.address i)=(s.mem "Y" i).read .fp32 := by
  simp [commit,footprint,DataView.address,show 4*i<4*y.size by omega]

theorem commit_frame {α : Type} [Inhabited α] (y : Tensor) (memory : Nat → α)
    (s : State α) (a : Nat) (outside : ¬footprint y a) : commit y memory s a=memory a := by
  simp [commit,outside]

theorem commit_reference {α : Type} [Inhabited α] (y : Tensor) (memory : Nat → α)
    (s : State α) (values : Nat → α) (all : ∀ i,i<y.size → (s.mem "Y" i).read .fp32=values i) :
    commit y memory s=reference y memory values := by
  funext address
  simp only [commit,reference]
  split
  · rename_i active
    exact all _ (by have := active.1; unfold footprint at this; omega)
  · rfl

def batchIndex (r : Request) (k : Nat) := k/r.columns/r.rows
def rowIndex (r : Request) (k : Nat) := k/r.columns%r.rows
def columnIndex (r : Request) (k : Nat) := k%r.columns

theorem coordinates (r : Request) (k : Nat) (hk : k<r.size) :
    batchIndex r k<r.batches ∧ rowIndex r k<r.rows ∧ columnIndex r k<r.columns ∧
      r.offset (batchIndex r k) (rowIndex r k) (columnIndex r k)=k := by
  have hc : 0<r.columns := by by_contra h; simp [Request.size,show r.columns=0 by omega] at hk
  have hr : 0<r.rows := by by_contra h; simp [Request.size,show r.rows=0 by omega] at hk
  have hdiv : k/r.columns<r.batches*r.rows :=
    (Nat.div_lt_iff_lt_mul hc).mpr hk
  refine ⟨(Nat.div_lt_iff_lt_mul hr).mpr hdiv,Nat.mod_lt _ hr,Nat.mod_lt _ hc,?_⟩
  unfold Request.offset batchIndex rowIndex columnIndex
  rw [Nat.div_add_mod',Nat.div_add_mod']

theorem scalar_logical {α : Type} [Inhabited α] (ops : Operations α) (r : Request)
    (x y : Tensor) (input : Input x r) (memory : Nat → α)
    (b m c : Nat) (hb : b<r.batches) (hm : m<r.rows) (hc : c<r.columns) :
    scalarResult ops r (fun o => ((initial x y memory).mem "X" o).read .fp32) b m c=
      scalarResult ops r (x.read memory) b m c := by
  have hn : c/2*2<r.columns := lt_of_le_of_lt (Nat.div_mul_le_self c 2) hc
  have real := x.read_view memory (r.offset b m (c/2*2))
    (by rw [input.size]; exact offset_lt r b m _ hb hm hn)
  have imag (h : c/2*2+1<r.columns) := x.read_view memory (r.offset b m (c/2*2+1))
    (by rw [input.size]; exact offset_lt r b m _ hb hm h)
  simp only [scalarResult,scalarPair,initial,Cell.read_same,↓reduceIte]
  rw [←real]
  by_cases h : c/2*2+1<r.columns
  · rw [if_pos h,if_pos h,←imag h]
  · rw [if_neg h,if_neg h]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Binding
