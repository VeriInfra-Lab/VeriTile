import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.NativeIO
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.Backing
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.AccessBounds
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.ReachedBytes

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.FiniteInputs
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.AttentionLlama HostGrid MaskCoverage
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

def tile (i : Index B NH H D) : TileIndex [B,NH,H,D] :=
  (i.batch,i.head,i.query,i.channel,())

def index (i : TileIndex [B,NH,H,D]) : Index B NH H D :=
  ⟨i.1,i.2.1,i.2.2.1,i.2.2.2.1⟩

@[simp] theorem tile_index (i : TileIndex [B,NH,H,D]) : tile (index i)=i := by
  rcases i with ⟨b,h,q,d,u⟩
  cases u
  rfl

@[simp] theorem index_tile (i : Index B NH H D) : index (tile i)=i := by cases i; rfl

@[simp] theorem output_tile (c : Config) (i : Index B c.NH c.H c.D) :
    IOLayout.output c (tile i)=Tensor.offset c i := rfl

def tensorInputs (A : Target.Ops α) (B : Nat) (c : Config)
    (vals : ChannelValues α (IOLayout.inputs B c)) : Tensor.Inputs α B c.NH c.H c.D where
  query := fun b h q d=>vals 0 (b,h,q,d,())
  halfKey := fun b h k d=>if hk:k<c.N then vals 1 (b,h,⟨k,hk⟩,d,()) else A.zero
  halfValue := fun b h k d=>if hk:k<c.N then vals 2 (b,h,⟨k,hk⟩,d,()) else A.zero
  byteKey := fun b h k d=>if hk:k<c.N then UInt8.ofFin (Float8E5Bits.ofInt (vals 3 (b,h,⟨k,hk⟩,d,()))) else 0
  byteValue := fun b h k d=>if hk:k<c.N then UInt8.ofFin (Float8E5Bits.ofInt (vals 4 (b,h,⟨k,hk⟩,d,()))) else 0

theorem expected (A : Target.Ops α) (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) (vals : ChannelValues α (IOLayout.inputs B c))
    (i : TileIndex [B,c.NH,c.H,c.D]) :
    Tensor.expected A C arg c (tensorInputs A B c vals) (index i)=
      Mathematical.native A C arg B c vals 0 i := rfl

theorem binding (A : Target.Ops α) (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) (vals : ChannelValues α (IOLayout.inputs B c)) (s : State (Term α))
    (hin : (NativeIO.io A C arg B c).InputBinding vals s) :
    Tensor.Binding c s (tensorInputs A B c vals) := by
  constructor
  · intro b h q d
    have hv:=hin 0 (b,h,q,d,()) (by simp [NativeIO.io,IOLayout.memory])
    simpa [NativeIO.io,IOLayout.memory,IOLayout.query,IOLayout.offset,mapValue,tensorInputs] using hv
  · intro hf b h k d hk
    have hv:=hin 1 (b,h,⟨k,hk⟩,d,()) (by simp [NativeIO.io,IOLayout.memory,hf])
    simpa [NativeIO.io,IOLayout.memory,IOLayout.key,IOLayout.offset,mapValue,tensorInputs,hk] using hv
  · intro hf b h k d hk
    have hv:=hin 2 (b,h,⟨k,hk⟩,d,()) (by simp [NativeIO.io,IOLayout.memory,hf])
    simpa [NativeIO.io,IOLayout.memory,IOLayout.value,IOLayout.offset,mapValue,tensorInputs,hk] using hv
  · intro hf b h k d hk
    have hv:=hin 3 (b,h,⟨k,hk⟩,d,()) (by simp [NativeIO.io,IOLayout.memory,hf])
    change (s.mem c.K (b.val*c.skz+h.val*c.skh+k*c.skn+d.val*c.skk)).read .int=vals 3 (b,h,⟨k,hk⟩,d,()) at hv
    simp only [tensorInputs,dif_pos hk,hv]
  · intro hf b h k d hk
    have hv:=hin 4 (b,h,⟨k,hk⟩,d,()) (by simp [NativeIO.io,IOLayout.memory,hf])
    change (s.mem c.V (b.val*c.svz+h.val*c.svh+k*c.svk+d.val*c.svn)).read .int=vals 4 (b,h,⟨k,hk⟩,d,()) at hv
    simp only [tensorInputs,dif_pos hk,hv]

/-- Finite byte roles imply representation at exactly the masked source reads. -/
theorem reached (A : Target.Ops α) (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) (vals : ChannelValues α (IOLayout.inputs B c)) (s : State (Term α))
    (hin : (NativeIO.io A C arg B c).InputBinding vals s)
    (bytes : IOLayout.nativeDomain B c vals) (i : Index B c.NH c.H c.D) :
    ReachedBytes c (Tensor.initialFor c s i) := by
  constructor
  · intro hf n j hj
    have hv:=hin 3 (i.batch,i.head,⟨n*64+j.1.val,hj⟩,j.2.1,())
      (by simp [NativeIO.io,IOLayout.memory,hf])
    change (s.mem c.K (i.batch.val*c.skz+i.head.val*c.skh+(n*64+j.1.val)*c.skn+j.2.1.val*c.skk)).read .int=
      vals 3 (i.batch,i.head,⟨n*64+j.1.val,hj⟩,j.2.1,()) at hv
    rw [pointers,Tensor.base_eq]
    have ha : i.batch.val*c.skz+i.head.val*c.skh+j.1.val*c.skn+j.2.1.val*c.skk+n*64*c.skn=
        i.batch.val*c.skz+i.head.val*c.skh+(n*64+j.1.val)*c.skn+j.2.1.val*c.skk := by simp [Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
    change Float8E5Bits.isInt8 ((s.mem c.K _).read .int)=Bool.true
    rw [ha,hv]
    exact (bytes hf _).1
  · intro hf n j hj
    have hv:=hin 4 (i.batch,i.head,⟨n*64+j.1.val,hj⟩,j.2.1,())
      (by simp [NativeIO.io,IOLayout.memory,hf])
    change (s.mem c.V (i.batch.val*c.svz+i.head.val*c.svh+(n*64+j.1.val)*c.svk+j.2.1.val*c.svn)).read .int=
      vals 4 (i.batch,i.head,⟨n*64+j.1.val,hj⟩,j.2.1,()) at hv
    rw [pointers,Tensor.base_eq]
    have ha : i.batch.val*c.svz+i.head.val*c.svh+j.1.val*c.svk+j.2.1.val*c.svn+n*64*c.svk=
        i.batch.val*c.svz+i.head.val*c.svh+(n*64+j.1.val)*c.svk+j.2.1.val*c.svn := by simp [Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
    change Float8E5Bits.isInt8 ((s.mem c.V _).read .int)=Bool.true
    rw [ha,hv]
    exact (bytes hf _).2

/-- Declared finite capacities cover every actual active source address. -/
theorem bounds (A : Target.Ops α) (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) (vals : ChannelValues α (IOLayout.inputs B c)) (s : State (Term α))
    (capacity : RegionName→Nat) (hb : (NativeIO.io A C arg B c).InBounds capacity vals)
    (i : Index B c.NH c.H c.D) : AccessBounds c (Tensor.initialFor c s i) capacity := by
  constructor
  · intro j hj
    have hq:=hb.2.1 0 (i.batch,i.head,⟨rows (Tensor.initialFor c s i) j.1,hj⟩,j.2.1,())
      (by simp [NativeIO.io,IOLayout.memory])
    rw [Tensor.base_eq]
    exact hq
  · intro n j hj
    have hk : i.batch.val*c.skz+i.head.val*c.skh+(n*64+j.1.val)*c.skn+j.2.1.val*c.skk<capacity c.K := by
      cases hf:c.fp8
      · exact hb.2.1 1 (i.batch,i.head,⟨n*64+j.1.val,hj⟩,j.2.1,())
          (by simp [NativeIO.io,IOLayout.memory,hf])
      · exact hb.2.1 3 (i.batch,i.head,⟨n*64+j.1.val,hj⟩,j.2.1,())
          (by simp [NativeIO.io,IOLayout.memory,hf])
    rw [pointers,Tensor.base_eq]
    simpa only [Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using hk
  · intro n j hj
    have hv : i.batch.val*c.svz+i.head.val*c.svh+(n*64+j.1.val)*c.svk+j.2.1.val*c.svn<capacity c.V := by
      cases hf:c.fp8
      · exact hb.2.1 2 (i.batch,i.head,⟨n*64+j.1.val,hj⟩,j.2.1,())
          (by simp [NativeIO.io,IOLayout.memory,hf])
      · exact hb.2.1 4 (i.batch,i.head,⟨n*64+j.1.val,hj⟩,j.2.1,())
          (by simp [NativeIO.io,IOLayout.memory,hf])
    rw [pointers,Tensor.base_eq]
    simpa only [Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using hv
  · intro j hj
    have ho:=hb.2.2.1 0 (i.batch,i.head,⟨rows (Tensor.initialFor c s i) j.1,hj⟩,j.2.1,()) trivial
    rw [address,Tensor.base_eq]
    exact ho

#axiomsClean binding
#axiomsClean reached
#axiomsClean bounds
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.FiniteInputs
