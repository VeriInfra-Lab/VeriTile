import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelNative
import VeriTile.Torch.FP32
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelTensor
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open KernelPacket (Call)
noncomputable section

inductive Role where
  | history | finalState | output | dhistory | initialGradient | queryGradient | keyGradient | valuePartials
  deriving DecidableEq

def Role.region : Role→RegionName
  | .history=>"h" | .finalState=>"ht" | .output=>"o" | .dhistory=>"dhistory"
  | .initialGradient=>"initialGradient" | .queryGradient=>"queryGradient"
  | .keyGradient=>"keyGradient" | .valuePartials=>"valuePartials"
def Role.shape (c : Shape) : Role→List Nat
  | .history | .dhistory=>[c.B,c.H,c.chunks*c.K,c.V]
  | .finalState | .initialGradient=>[c.B,c.H,c.K,c.V]
  | .output=>[c.B,c.H,c.T,c.V]
  | .queryGradient | .keyGradient=>[c.B,c.H,c.T,c.K]
  | .valuePartials=>[c.keyBlocks,c.B,c.H,c.T,c.V]

/-- Ordinary binding of the actual Tensor object and data pointer. It records
only dtype/layout/shape/contiguity/coordinates, with arbitrary storage offsets.
It assumes no kernel completion, output value, or allocation protocol. -/
structure View (c : Shape) (role : Role) where
  tensor : VeriTile.Torch.FP32.Tensor
  laws : VeriTile.Torch.FP32.TensorLaws tensor
  shape : tensor.metadata.shape=role.shape c
  contiguous : tensor.metadata.isContiguous=Bool.true
  resolved : tensor.metadata.isNeg=Bool.false

def address (v : View c role) (i : Nat) := v.tensor.metadata.dataPtr.toNat+4*i

theorem coordinates (v : View c role) (i : Nat) (hi : i<v.tensor.metadata.numel) :
    v.tensor.storage.address (v.tensor.physicalIndex i)=address v i := by
  rw [v.laws.contiguous v.contiguous i hi]
  have hp:=v.laws.pointer (by omega)
  have hc:=VeriTile.Torch.FP32.contiguous_capacity v.tensor v.laws v.contiguous (by omega)
  have hs:=v.tensor.storage.spanFits
  have hb : v.tensor.storage.address v.tensor.storageOffset<2^64 := by
    unfold VeriTile.Torch.FP32.Storage.address
    omega
  have he:=congrArg BitVec.toNat hp
  simp only [VeriTile.Torch.FP32.Tensor.dataPtr,BitVec.toNat_ofNat,Nat.mod_eq_of_lt hb] at he
  unfold address VeriTile.Torch.FP32.Storage.address
  rw [he]
  simp only [VeriTile.Torch.FP32.Storage.address]
  ring

def logical (read : Nat→β) (v : View c role) (i : Nat) : β :=
  read (v.tensor.storage.address (v.tensor.physicalIndex i))

/-- Exact word addresses written by this program into this observed Tensor.
The rest of the Tensor, offset prefixes/tails, and unrelated memory are framed. -/
def Written (a : Call) (role : Role) (v : View a.shape role) (addr : Nat) : Prop :=
  v.tensor.metadata.dataPtr.toNat≤addr ∧
  (addr-v.tensor.metadata.dataPtr.toNat)%4=0 ∧
  (addr-v.tensor.metadata.dataPtr.toNat)/4<v.tensor.metadata.numel ∧
  KernelPacket.Footprint a role.region ((addr-v.tensor.metadata.dataPtr.toNat)/4)

def project {α β : Type} [Inhabited α] (decode : α→β) (read : Nat→β)
    (a : Call) (role : Role) (v : View a.shape role) (after : State α) (addr : Nat) : β := by
  classical
  exact if Written a role v addr then
    decode ((after.mem role.region ((addr-v.tensor.metadata.dataPtr.toNat)/4)).read .real)
  else read addr

theorem projected {α β : Type} [Inhabited α] (decode : α→β) (read : Nat→β)
    (a : Call) (role : Role) (v : View a.shape role) (after : State α)
    (i : Nat) (hi : i<v.tensor.metadata.numel) (hwrite : KernelPacket.Footprint a role.region i) :
    logical (project decode read a role v after) v i=decode ((after.mem role.region i).read .real) := by
  rw [logical,coordinates v i hi]
  simp [project,Written,address,hi,hwrite]

theorem exterior {α β : Type} [Inhabited α] (decode : α→β) (read : Nat→β)
    (a : Call) (role : Role) (v : View a.shape role) (after : State α)
    (addr : Nat) (h : ¬Written a role v addr) : project decode read a role v after addr=read addr := by
  simp [project,h]

/-- Full backing disjointness is the existing kernel buffer guard. This theorem
preserves every caller input word, including inactive offset prefixes/tails. -/
theorem input_frame {α β : Type} [Inhabited α] (decode : α→β) (read : Nat→β)
    (a : Call) (role : Role) (v : View a.shape role) (after : State α)
    (input : VeriTile.Torch.FP32.Tensor)
    (apart : v.tensor.storage.base+4*v.tensor.storage.elements ≤ input.storage.base ∨
      input.storage.base+4*input.storage.elements ≤ v.tensor.storage.base)
    (addr : Nat) (inside : input.storage.base≤addr ∧ addr < input.storage.base+4*input.storage.elements) :
    project decode read a role v after addr=read addr := by
  apply exterior
  intro hw
  have hlo:=hw.1
  have hmod:=hw.2.1
  let i:=(addr-v.tensor.metadata.dataPtr.toNat)/4
  have hi : i<v.tensor.metadata.numel:=hw.2.2.1
  have he : addr=address v i := by
    dsimp [i,address]
    have hd:=Nat.div_add_mod (addr-v.tensor.metadata.dataPtr.toNat) 4
    omega
  have hc:=coordinates v i hi
  have hb:=v.laws.inBounds i hi
  rw [←hc] at he
  unfold VeriTile.Torch.FP32.Storage.address at he
  omega

theorem native_logical (read : Nat→BitVec 32) (v : View c role) (i : Nat) :
    (VeriTile.Torch.FP32.logicalValue (fun addr=>⟨read addr⟩) v.tensor v.laws i).bits=logical read v i := by
  simp [VeriTile.Torch.FP32.logicalValue,v.resolved,logical]

/-- Shared execution to actual Tensor coordinates, with exact physical and
logical frame. The decode is a numerical interpretation, never an output premise. -/
theorem source_tensor {α β : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (decode : α→β) (read : Nat→β) (a : Call) (role : Role) (v : View a.shape role)
    (m : KernelPacket.Memory α) :
    ∃ t,FP.Structural.exec (engine (withDot M D)) a.kernel (a.state m)=some t ∧
      (∀ i,i<v.tensor.metadata.numel → KernelPacket.Footprint a role.region i →
        logical (project decode read a role v t) v i=
          decode (((KernelPacket.reference M D a m) role.region i).read .real)) ∧
      (∀ addr,¬Written a role v addr → project decode read a role v t addr=read addr) ∧
      (∀ r off,¬KernelPacket.Footprint a r off → t.mem r off=m r off) := by
  obtain ⟨t,ht,hm,hf⟩:=KernelPacket.complete M D a m
  refine ⟨t,ht,?_,?_,hf⟩
  · intro i hi hw
    rw [projected decode read a role v t i hi hw,hm]
  · exact fun addr h=>exterior decode read a role v t addr h

/-- Actual Tensor FP32 logical observations equal the independent native target;
no whole-source/compiler-result equality is a hypothesis. -/
theorem native_tensor (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→BitVec 32) (read : Nat→BitVec 32)
    (a : Call) (role : Role) (v : View a.shape role) (m : RegionName→Nat→BitVec 32) :
    ∃ t,FP.Structural.exec (engine (withDot KernelNative.model (KernelNative.dotModel a)))
        a.kernel (a.state (KernelNative.input m))=some t ∧
      (∀ i,i<v.tensor.metadata.numel → KernelPacket.Footprint a role.region i →
        (VeriTile.Torch.FP32.logicalValue
          (fun addr=>⟨project (KernelNative.S.lower (KernelNative.sharedOps (wordOps (Numeric.nativeISA ports))) literal)
            read a role v t addr⟩) v.tensor v.laws i).bits=
          KernelNative.target (Numeric.referenceISA u) literal a m role.region i) ∧
      (∀ addr,¬Written a role v addr →
        project (KernelNative.S.lower (KernelNative.sharedOps (wordOps (Numeric.nativeISA ports))) literal)
          read a role v t addr=read addr) := by
  obtain ⟨t,ht,hm,_⟩:=KernelNative.correct u ports laws literal a m
  refine ⟨t,ht,?_,?_⟩
  · intro i hi hw
    rw [native_logical]
    exact (projected (KernelNative.S.lower (KernelNative.sharedOps (wordOps (Numeric.nativeISA ports))) literal) read a role v t i hi hw).trans (hm role.region i)
  · exact fun addr h=>exterior _ read a role v t addr h

#axiomsClean coordinates
#axiomsClean input_frame
#axiomsClean native_logical
#axiomsClean source_tensor
#axiomsClean native_tensor
#specNonCircular View avoiding [run,FP.Structural.exec,KernelPacket.reference,KernelNative.target]
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelTensor
