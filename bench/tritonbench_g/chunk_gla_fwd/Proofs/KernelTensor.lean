import bench.tritonbench_g.chunk_gla_fwd.Contract

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.KernelTensor
open VeriTile Triton
noncomputable section

def width (mode : Admission.Mode) (c : Host.Dimensions) : Nat :=
  match mode with | .producer => c.BT | .consumer => c.V
def region (mode : Admission.Mode) : RegionName :=
  match mode with | .producer => "attention" | .consumer => "o"
abbrev Index (mode : Admission.Mode) (c : Host.Dimensions) :=
  Fin c.B × Fin c.H × Fin c.T × Fin (width mode c)

def head (c : Host.Dimensions) (b : Fin c.B) (h : Fin c.H) : Fin c.heads :=
  ⟨b.val*c.H+h.val,by
    have hb := b.isLt
    have hh := h.isLt
    change b.val*c.H+h.val<c.B*c.H
    nlinarith⟩
def offset (mode : Admission.Mode) (c : Host.Dimensions) (i : Index mode c) : Nat :=
  (head c i.1 i.2.1).val*c.T*width mode c+i.2.2.1.val*width mode c+i.2.2.2.val

/-- Observation of the actual returned Tensor. Its fields contain only ordinary
metadata and contiguous-coordinate facts, with the origin at data_ptr. They
contain no execution, numerical output or whole-program correspondence. -/
structure Output (mode : Admission.Mode) (c : Host.Dimensions) (device : Nat) where
  info : Admission.Tensor
  valid : info.Valid device [c.B,c.H,c.T,width mode c]
  physicalIndex : Nat → Nat → Nat → Nat → Nat
  coordinates : ∀ b h t j, b<c.B → h<c.H → t<c.T → j<width mode c →
    physicalIndex b h t j=((b*c.H+h)*c.T+t)*width mode c+j

theorem coordinates (view : Output mode c device) (i : Index mode c) :
    view.physicalIndex i.1.val i.2.1.val i.2.2.1.val i.2.2.2.val=offset mode c i := by
  rw [view.coordinates _ _ _ _ i.1.isLt i.2.1.isLt i.2.2.1.isLt i.2.2.2.isLt]
  simp only [offset,head]
  ring

structure Tensor (α : Type) (mode : Admission.Mode) (c : Host.Dimensions) (device : Nat) where
  view : Output mode c device
  storage : Nat → α
def Tensor.get (t : Tensor α mode c device) (i : Index mode c) : α :=
  t.storage (t.view.physicalIndex i.1.val i.2.1.val i.2.2.1.val i.2.2.2.val)

def realResult (view : Output mode c device) (s : BlockState) : Tensor ℝ mode c device :=
  ⟨view,s.readMem (region mode)⟩
def nativeResult (view : Output mode c device) (m : Nat → BitVec 32) : Tensor (BitVec 32) mode c device :=
  ⟨view,fun i => m (view.info.byteBase/4+i)⟩

theorem metadata (t : Tensor α mode c device) :
    t.view.info.shape=[c.B,c.H,c.T,width mode c] ∧
    t.view.info.fp32=Bool.true ∧ t.view.info.device=device :=
  ⟨t.view.valid.2.2.2.2.2.2.2.2,t.view.valid.2.2.1,t.view.valid.2.2.2.2.2.2.2.1⟩

#axiomsClean coordinates
#axiomsClean metadata
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.KernelTensor
