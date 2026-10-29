import bench.repairs.tritonbench_g.flash_attn.KernelOnly.NativeResult
import bench.repairs.tritonbench_g.flash_attn.Proofs.NativeBinding
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.FlashAttn.KernelTensor
open VeriTile Triton FP.Structural ActualSource Checked
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000

/-- Mapping the source's two regions back to the original fresh O/L byte
allocations. Other bytes retain their initial values. -/
def finish (x : Layout) (base : FlashAttn.Region→Nat) (I : KernelNative.Instructions)
    (graph : Mem KernelNative.Expr) (before : ByteMemory) : ByteMemory := fun a=>
  if StoredByte x base .o a.val then
    Physical.encode (KernelNative.observe I x graph "O" ((a.val-base .o)/4)) ⟨(a.val-base .o)%4,Nat.mod_lt _ (by decide)⟩
  else if StoredByte x base .l a.val then
    Physical.encode (KernelNative.observe I x graph "L" ((a.val-base .l)/4)) ⟨(a.val-base .l)%4,Nat.mod_lt _ (by decide)⟩
  else before a

structure Result where
  graph : Mem KernelNative.Expr
  bytes : ByteMemory

def run (x : Layout) (base : FlashAttn.Region→Nat) (I : KernelNative.Instructions)
    (scale : ScalarABI) (before : ByteMemory) : Option Result := do
  let graph←KernelNative.run x scale (Physical.load (callerBytes base before))
  return ⟨graph,finish x base I graph before⟩

def nativeTarget (x : Layout) (base : FlashAttn.Region→Nat) (I : KernelNative.Instructions)
    (scale : ScalarABI) (before : ByteMemory) (r : RegionName) (off : Nat) : Word :=
  KernelNative.target I x scale (Physical.load (callerBytes base before)) r off

theorem output_byte (a : Host.Arguments) (q k v o : Checked.Tensor) (scratch : Scratch)
    (allocation : Allocations a o scratch.allocation) (order : SortedAxes) (ho : tensorChecks o order=Bool.true)
    (I : KernelNative.Instructions) (graph : Mem KernelNative.Expr) (before : ByteMemory)
    (b h row col : Nat) (active : (Work.output b h row col).active (Host.shape a)) (j : Fin 4) :
    extendMemory (finish (Host.layout a o.tensor) (pointers q k v o scratch) I graph before)
      (o.dataPointer+4*FlashAttn.address o.tensor.strides b h row col+j.val)=
      Physical.encode (KernelNative.observe I (Host.layout a o.tensor) graph "O"
        (FlashAttn.address o.tensor.strides b h row col)) j := by
  have ha : b<o.tensor.shape.batches ∧ h<o.tensor.shape.heads ∧ row<o.tensor.shape.length ∧ col<o.tensor.shape.dim := by
    simpa only [Work.active,Host.shape,allocation.outputShape] using active
  have hb:=(formed_tensor o order ho b h row col ha.1 ha.2.1 (by omega) ha.2.2.2 j).2.2
  have hs:=stored_output (Host.layout a o.tensor) (pointers q k v o scratch) b h row col active j
  change StoredByte (Host.layout a o.tensor) (pointers q k v o scratch) .o
    (o.dataPointer+4*FlashAttn.address o.tensor.strides b h row col+j.val) at hs
  rw [extendMemory,dif_pos hb]
  simp only [finish,hs,ite_true,pointers]
  have hsub : o.dataPointer+4*FlashAttn.address o.tensor.strides b h row col+j.val-o.dataPointer=
      4*FlashAttn.address o.tensor.strides b h row col+j.val := by omega
  simp only [hsub]
  have hdiv : (4*FlashAttn.address o.tensor.strides b h row col+j.val)/4=FlashAttn.address o.tensor.strides b h row col := by omega
  have hmod : (4*FlashAttn.address o.tensor.strides b h row col+j.val)%4=j.val := by omega
  simp only [hdiv]
  congr 1
  exact Fin.ext hmod

theorem scratch_byte (a : Host.Arguments) (q k v o : Checked.Tensor) (scratch : Scratch)
    (allocation : Allocations a o scratch.allocation) (order : SortedAxes) (ho : tensorChecks o order=Bool.true)
    (binding : ScratchBinding a scratch) (fresh : Fresh q k v o scratch)
    (hp : scratchPointerChecks (Host.shape a) scratch.dataPointer=Bool.true)
    (I : KernelNative.Instructions) (graph : Mem KernelNative.Expr) (before : ByteMemory)
    (b h row : Nat) (active : (Work.logsum b h row).active (Host.shape a)) (j : Fin 4) :
    extendMemory (finish (Host.layout a o.tensor) (pointers q k v o scratch) I graph before)
      (scratch.dataPointer+4*FlashAttn.lAddress (Host.shape a) b h row+j.val)=
      Physical.encode (KernelNative.observe I (Host.layout a o.tensor) graph "L"
        (FlashAttn.lAddress (Host.shape a) b h row)) j := by
  have hs:=stored_scratch (Host.layout a o.tensor) (pointers q k v o scratch) b h row active j
  change StoredByte (Host.layout a o.tensor) (pointers q k v o scratch) .l
    (scratch.dataPointer+4*FlashAttn.lAddress (Host.shape a) b h row+j.val) at hs
  have alloc:=scratch_allocation a q k v o scratch binding _ hs
  have no : ¬StoredByte (Host.layout a o.tensor) (pointers q k v o scratch) .o
      (scratch.dataPointer+4*FlashAttn.lAddress (Host.shape a) b h row+j.val) := by
    intro h
    have other:=output_allocation a q k v o scratch allocation order ho _ h
    have hf:=fresh.outputScratch
    omega
  have hl:=l_bound (Host.shape a) b h row active.1 active.2.1 active.2.2
  simp only [scratchPointerChecks,decide_eq_true_eq] at hp
  have bounded : scratch.dataPointer+4*FlashAttn.lAddress (Host.shape a) b h row+j.val<2^64 := by omega
  rw [extendMemory,dif_pos bounded]
  simp only [finish,no,ite_false,hs,ite_true,pointers]
  have hsub : scratch.dataPointer+4*FlashAttn.lAddress (Host.shape a) b h row+j.val-scratch.dataPointer=
      4*FlashAttn.lAddress (Host.shape a) b h row+j.val := by omega
  simp only [hsub]
  have hdiv : (4*FlashAttn.lAddress (Host.shape a) b h row+j.val)/4=FlashAttn.lAddress (Host.shape a) b h row := by omega
  have hmod : (4*FlashAttn.lAddress (Host.shape a) b h row+j.val)%4=j.val := by omega
  simp only [hdiv]
  congr 1
  exact Fin.ext hmod

theorem finish_frame (x : Layout) (base : FlashAttn.Region→Nat) (I : KernelNative.Instructions)
    (graph : Mem KernelNative.Expr) (before : ByteMemory) (a : Fin (2^64))
    (ho : ¬StoredByte x base .o a.val) (hl : ¬StoredByte x base .l a.val) :
    finish x base I graph before a=before a := by simp [finish,ho,hl]

theorem input_frame (a : Host.Arguments) (q k v o : Checked.Tensor) (scratch : Scratch)
    (allocation : Allocations a o scratch.allocation) (order : SortedAxes) (ho : tensorChecks o order=Bool.true)
    (binding : ScratchBinding a scratch) (fresh : Fresh q k v o scratch)
    (I : KernelNative.Instructions) (graph : Mem KernelNative.Expr) (before : ByteMemory)
    (t : Checked.Tensor) (ht : t∈[q,k,v]) (location : Fin (2^64))
    (hlo : t.storageBase≤location.val) (hhi : location.val<t.storageBase+t.tensor.storageBytes) :
    finish (Host.layout a o.tensor) (pointers q k v o scratch) I graph before location=before location := by
  apply finish_frame
  · intro h
    have bounds:=output_allocation a q k v o scratch allocation order ho _ h
    have hf:=fresh.outputInputs t ht
    omega
  · intro h
    have bounds:=scratch_allocation a q k v o scratch binding _ h
    have hf:=fresh.scratchInputs t ht
    omega

def flat (s : Shape) (b h row col : Nat) := ((b*s.heads+h)*s.length+row)*s.dim+col

/-- Ordinary Torch binding facts about the same observed descriptor/storage.
The strided index equation supplies no source result or arithmetic hypothesis. -/
structure TensorBinding (observed : Checked.Tensor) (actual : Torch.FP32.Tensor) : Prop where
  laws : Torch.FP32.TensorLaws actual
  allocation : actual.storage.object=observed.tensor.allocation
  base : actual.storage.base=observed.storageBase
  offset : actual.storageOffset=observed.tensor.storageOffset
  pointer : actual.dataPtr.toNat=observed.dataPointer
  notNeg : actual.metadata.isNeg=Bool.false
  elements : actual.metadata.numel=observed.tensor.shape.batches*observed.tensor.shape.heads*
    observed.tensor.shape.length*observed.tensor.shape.dim
  strided : ∀ b h row col,(Work.output b h row col).active observed.tensor.shape →
    actual.physicalIndex (flat observed.tensor.shape b h row col)=
      observed.tensor.storageOffset+FlashAttn.address observed.tensor.strides b h row col

def read (bytes : ByteMemory) (location : Nat) : Torch.FP32.Word := ⟨readWord bytes location⟩

theorem logical_read (t : Checked.Tensor) (actual : Torch.FP32.Tensor)
    (bind : TensorBinding t actual) (bytes : ByteMemory) (b h row col : Nat)
    (active : (Work.output b h row col).active t.tensor.shape) :
    (Torch.FP32.logicalValue (read bytes) actual bind.laws (flat t.tensor.shape b h row col)).bits=
      readWord bytes (t.dataPointer+4*FlashAttn.address t.tensor.strides b h row col) := by
  simp only [Torch.FP32.logicalValue,bind.notNeg,Bool.false_eq_true,ite_false,
    bind.strided b h row col active,Torch.FP32.Storage.address,bind.base,read]
  rw [t.pointer]
  congr 1
  omega

/-- Original Torch logical inputs are exactly the raw leaves loaded by the
shared source. No private tensor, normalization, or output premise is supplied. -/
theorem input_tensor (t : Checked.Tensor) (actual : Torch.FP32.Tensor) (binding : TensorBinding t actual)
    (before : ByteMemory) (base : FlashAttn.Region→Nat) (r : FlashAttn.Region)
    (role : r∈[.q,.k,.v]) (pointer : base r=t.dataPointer) (b h row col : Nat)
    (active : (Work.output b h row col).active t.tensor.shape) (owner : Fin 4) :
    ((KernelNative.initialMemory (Physical.load (callerBytes base before)))
      (regionName r) (FlashAttn.address t.tensor.strides b h row col)).read .real owner=
      KernelNative.Node.word ((Torch.FP32.logicalValue (read before) actual binding.laws
        (flat t.tensor.shape b h row col)).bits) := by
  rw [logical_read t actual binding before b h row col active,←pointer]
  have decoded : sourceRegion (regionName r)=r := by
    cases r <;> simp_all [regionName,sourceRegion]
  simp only [KernelNative.initialMemory,liftMemory,Cell.read,decoded,caller_load,dif_pos True.intro]

#axiomsClean input_tensor

theorem whole_native (q k v o : Checked.Tensor) (qo ko vo oo : SortedAxes)
    (scale : ScalarABI) (causal : Bool) (owner : Nat→Nat→Nat→Fin 4)
    (scratch : Scratch) (checked : inputChecks q k v qo ko vo scale=Bool.true)
    (outputChecked : tensorChecks o oo=Bool.true)
    (allocation : Allocations (arguments q k v causal scale owner) o scratch.allocation)
    (scratchBinding : ScratchBinding (arguments q k v causal scale owner) scratch)
    (fresh : Fresh q k v o scratch)
    (scratchChecked : scratchPointerChecks (Host.shape (arguments q k v causal scale owner)) scratch.dataPointer=Bool.true)
    (actualOutput : Torch.FP32.Tensor) (outputBinding : TensorBinding o actualOutput)
    (I : KernelNative.Instructions) (before : ByteMemory) :
    let a:=arguments q k v causal scale owner
    let x:=Host.layout a o.tensor
    let base:=pointers q k v o scratch
    Host.Contract a ∧ ∃ result,run x base I scale before=some result ∧
      KernelNative.run x scale (Physical.load (callerBytes base before))=some result.graph ∧
      (∀ b h row col,(Work.output b h row col).active (Host.shape a) →
        (Torch.FP32.logicalValue (read result.bytes) actualOutput outputBinding.laws
          (flat o.tensor.shape b h row col)).bits=
          nativeTarget x base I scale before "O" (FlashAttn.address o.tensor.strides b h row col)) ∧
      (∀ b h row,(Work.logsum b h row).active (Host.shape a) →
        readWord result.bytes (scratch.dataPointer+4*FlashAttn.lAddress (Host.shape a) b h row)=
          nativeTarget x base I scale before "L" (FlashAttn.lAddress (Host.shape a) b h row)) ∧
      (∀ location,¬StoredByte x base .o location.val → ¬StoredByte x base .l location.val →
        result.bytes location=before location) ∧
      (∀ t∈[q,k,v],∀ location,t.storageBase≤location.val →
        location.val<t.storageBase+t.tensor.storageBytes → result.bytes location=before location) := by
  dsimp only
  let a:=arguments q k v causal scale owner
  let x:=Host.layout a o.tensor
  let base:=pointers q k v o scratch
  obtain ⟨graph,hgraph,heq,values,frame⟩:=KernelNative.whole I x scale (Physical.load (callerBytes base before))
  refine ⟨admitted_inputs q k v qo ko vo scale checked causal owner,
    ⟨graph,finish x base I graph before⟩,?_,hgraph,?_,?_,?_,?_⟩
  · change run x base I scale before=some ⟨graph,finish x base I graph before⟩
    simp [run,hgraph]
  · intro b h row col active
    have ha : (Work.output b h row col).active o.tensor.shape := by
      simpa only [Work.active,allocation.outputShape,Host.shape] using active
    rw [logical_read o actualOutput outputBinding _ b h row col ha]
    unfold readWord
    have hb:=funext (output_byte a q k v o scratch allocation oo outputChecked I graph before b h row col active)
    rw [hb,Physical.decode_encode]
    exact values _ _
  · intro b h row active
    unfold readWord
    have hb:=funext (scratch_byte a q k v o scratch allocation oo outputChecked scratchBinding fresh scratchChecked
      I graph before b h row active)
    rw [hb,Physical.decode_encode]
    exact values _ _
  · exact fun location ho hl=>finish_frame x base I graph before location ho hl
  · exact fun t ht location hlo hhi=>input_frame a q k v o scratch allocation oo outputChecked
      scratchBinding fresh I graph before t ht location hlo hhi

#axiomsClean whole_native
#axiomsClean output_byte
#axiomsClean scratch_byte
#axiomsClean input_frame
#axiomsClean logical_read
#specNonCircular nativeTarget avoiding [FP.Structural.exec,FP.Structural.run,ActualSource.runGrid,KernelNative.run,run]
end
end VeriTile.Bench.Repairs.FlashAttn.KernelTensor
