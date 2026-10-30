import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Proofs.KernelEndpoints
import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Proofs.KernelSafety

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.KernelReadback
open VeriTile.Triton FP.Structural FlashDecode2 KernelModel KernelTensor
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 8000000

def base (c : Call) : Nat := c.val.output.metadata.dataPtr.toNat

/-- This record supplies only checked geometry to the existing byte-store
interpreter. It carries no host preparation, allocator or execution fact. -/
def geometry (c : Call) (h : Checked.metadataChecks c.val) : Host.Request :=
  ⟨Checked.geometry c.val,h.2.2.2.2.2.2.2.1⟩

def localBytes (c : Call) (heap : Heap) : Physical.Bytes := fun _ i => heap (base c+i)

/-- Relocate the source's sequential four-byte stores to the actual output
pointer. Below that pointer bytes retain their caller values; the source frame
theorem also preserves every byte beyond the last output element. -/
def execute (p : KernelNative.Ops) (S : Native.Schedule) (c : Call) (heap : Heap)
    (other : State Native.Term) (h : Checked.metadataChecks c.val) (order : List Nat) : Heap :=
  fun a => if a  <  base c then heap a else
    Physical.executeBytes (KernelNative.lower p) S "O" (geometry c h) order
      (KernelNative.entry c heap other) (localBytes c heap) "O" (a-base c)

theorem initial_related (p : KernelNative.Ops) (S : Native.Schedule) (c : Call)
    (heap : Heap) (other : State Native.Term) :
    ∀ a,Physical.projectO (KernelNative.lower p) S (KernelNative.entry c heap other) a=
      localBytes c heap "O" a := by
  intro a
  simp only [Physical.projectO,Native.observe,KernelNative.entry,lift,
    show ("O" : RegionName)  ≠  "L" by decide,ite_false,
    show ("O" : RegionName) = "O" by rfl,or_true,ite_true,Cell.read_same,Native.eval]
  change Physical.encode (Physical.decode (fun j => heap (base c+4*(a/4)+j.val)))
    ⟨a%4,Nat.mod_lt _ (by decide)⟩=heap (base c+a)
  rw [congrFun (Physical.encode_decode (fun j => heap (base c+4*(a/4)+j.val)))]
  congr 1
  change base c+4*(a/4)+a%4=base c+a
  omega

theorem source_related (p : KernelNative.Ops) (S : Native.Schedule) (c : Call)
    (heap : Heap) (other : State Native.Term) (h : Checked.metadataChecks c.val)
    (order : List Nat) (after : State Native.Term)
    (ran : run Native.algebra c order (KernelNative.entry c heap other)=some after) :
    ∀ a,Physical.projectO (KernelNative.lower p) S after a=
      execute p S c heap other h order (base c+a) := by
  obtain ⟨s,hs,hm⟩ := Grid.source_complete Native.algebra c.val.masked
    (Checked.batches c.val) (Checked.heads c.val) (Checked.capacity c.val)
    (Checked.width c.val) c.val.blockSeq (Checked.block c.val) order (KernelNative.entry c heap other)
  have same : after=s := Option.some.inj (ran.symm.trans hs)
  subst after
  intro a
  simp only [execute,show ¬base c+a < base c by omega,ite_false,Nat.add_sub_cancel_left]
  unfold Physical.projectO Native.observe
  rw [hm]
  exact Physical.grid_related (KernelNative.lower p) S "O" (geometry c h) order
    (KernelNative.entry c heap other) (localBytes c heap) (initial_related p S c heap other) a

theorem exterior (p : KernelNative.Ops) (S : Native.Schedule) (c : Call)
    (heap : Heap) (other : State Native.Term) (h : Checked.metadataChecks c.val)
    (order : List Nat) (all : order.Perm (List.range (rows c))) (a : Nat)
    (outside : a < base c ∨ base c+4*size c ≤ a) :
    execute p S c heap other h order a=heap a := by
  unfold execute
  split
  · rfl
  · rename_i below
    have upper : 4*(geometry c h).g.size ≤ a-base c := by
      change 4*size c ≤ a-base c
      omega
    rw [Physical.execute_frame (KernelNative.lower p) S "O" (geometry c h) order
      (fun r hr => List.mem_range.mp (all.mem_iff.mp hr)) _ _ "O" _ (.inr upper)]
    simp only [localBytes]
    congr 1
    omega

theorem output_word (p : KernelNative.Ops) (S : Native.Schedule) (c : Call)
    (heap : Heap) (other : State Native.Term) (h : Checked.metadataChecks c.val)
    (order : List Nat) (after : State Native.Term)
    (ran : run Native.algebra c order (KernelNative.entry c heap other)=some after) (i : Nat) :
    load (execute p S c heap other h order) (base c+4*i)=
      Native.observe (KernelNative.lower p) S after "O" i := by
  unfold load
  have related := source_related p S c heap other h order after ran
  have bytes : (fun j : Fin 4 => execute p S c heap other h order (base c+4*i+j.val))=
      Physical.encode (Native.observe (KernelNative.lower p) S after "O" i) := by
    funext j
    rw [Nat.add_assoc,←related]
    unfold Physical.projectO
    have div : (4*i+j.val)/4=i := by omega
    have mod : (4*i+j.val)%4=j.val := by omega
    simp only [div]
    congr 1
    exact Fin.ext mod
  rw [bytes,Physical.decode_encode]

theorem output_pointer (c : Call) (t : Arguments c) (h : Checked.metadataChecks c.val)
    (positive : 0 < size c) : base c=c.val.output.storage.base := by
  have facts := KernelSafety.facts c t h c.val.output (by simp [Checked.tensors])
  have shape := h.2.2.2.2.2.2.1
  have numel : c.val.output.metadata.numel=size c := by
    simp [VeriTile.Torch.TensorMetadata.numel,shape,size,rows,Nat.mul_assoc]
  have pointer := facts.pointer (by omega)
  have cap := facts.capacity
  have span := c.val.output.storage.spanFits
  have bound : c.val.output.storage.base < 2^64 := by omega
  exact (congrArg BitVec.toNat pointer).trans (Nat.mod_eq_of_lt bound)

theorem input_storage (p : KernelNative.Ops) (S : Native.Schedule) (c : Call) (t : Arguments c)
    (heap : Heap) (other : State Native.Term) (h : InputContract c (lengths c heap))
    (order : List Nat) (all : order.Perm (List.range (rows c)))
    (o : Checked.Observation) (member : o∈[c.val.mid,c.val.log,c.val.lengths])
    (a : Nat) (inside : o.storage.base ≤ a ∧ a < o.storage.base+4*o.storage.elements) :
    execute p S c heap other h.metadata order a=heap a := by
  apply exterior p S c heap other h.metadata order all
  by_cases positive : 0 < size c
  · have active : Checked.active c.val := by
      dsimp [size,rows] at positive
      constructor
      · by_contra bad
        have zero : Checked.batches c.val=0 := by omega
        simp [zero] at positive
      · by_contra bad
        have zero : Checked.heads c.val=0 := by omega
        simp [zero] at positive
    have apart := (h.launch active).2.1 o member
    have cap := (KernelSafety.facts c t h.metadata c.val.output (by simp [Checked.tensors])).capacity
    have shape := h.metadata.2.2.2.2.2.2.1
    have numel : c.val.output.metadata.numel=size c := by
      simp [VeriTile.Torch.TensorMetadata.numel,shape,size,rows,Nat.mul_assoc]
    rw [output_pointer c t h.metadata positive]
    unfold Checked.apart at apart
    omega
  · have empty : size c=0 := by omega
    simp only [empty,Nat.mul_zero,Nat.add_zero]
    omega

theorem native (p : KernelNative.Ops) (S : Native.Schedule) (c : Call) (t : Arguments c)
    (heap : Heap) (other : State Native.Term) (h : InputContract c (lengths c heap))
    (order : List Nat) (all : order.Perm (List.range (rows c))) :
    ∃ after,run Native.algebra c order (KernelNative.entry c heap other)=some after ∧
      (∀ b head d,b < Checked.batches c.val → head < Checked.heads c.val → d < Checked.width c.val →
        VeriTile.Torch.FP32.logicalValue (readWord (execute p S c heap other h.metadata order))
          t.output.tensor t.output.laws ((b*Checked.heads c.val+head)*Checked.width c.val+d)=
        (⟨KernelNative.target p S c heap (b*Checked.heads c.val+head) d⟩ : VeriTile.Torch.FP32.Word)) ∧
      (∀ a,a < base c ∨ base c+4*size c ≤ a → execute p S c heap other h.metadata order a=heap a) ∧
      (∀ o∈[c.val.mid,c.val.log,c.val.lengths],∀ a,
        o.storage.base ≤ a ∧ a < o.storage.base+4*o.storage.elements →
        execute p S c heap other h.metadata order a=heap a) ∧
      (∀ r i,r ≠ ("O" : RegionName) ∨ size c ≤ i → after.mem r i=(KernelNative.entry c heap other).mem r i) := by
  obtain ⟨after,ran,values,frame⟩ := KernelEndpoints.native p S c t heap other h order all
  refine ⟨after,ran,?_,exterior p S c heap other h.metadata order all,
    input_storage p S c t heap other h order all,frame⟩
  intro b head d hb hh hd
  have hr := List.mem_range.mp (coverage c b head hb hh).1
  have bound : (b*Checked.heads c.val+head)*Checked.width c.val+d < size c := by unfold size; nlinarith
  have checks := h.metadata.1 c.val.output (by simp [Checked.tensors])
  have shape := h.metadata.2.2.2.2.2.2.1
  have numel : (b*Checked.heads c.val+head)*Checked.width c.val+d < c.val.output.metadata.numel := by
    simpa [VeriTile.Torch.TensorMetadata.numel,shape,size,rows,Nat.mul_assoc] using bound
  rw [←t.outputObserved] at checks numel
  rw [←direct_logical t.output (execute p S c heap other h.metadata order)
    checks.2.2.2.1 checks.2.2.2.2.1 checks.2.2.2.2.2.1 _ numel]
  have pointer : t.output.tensor.metadata.dataPtr.toNat=base c := by
    simp [base,←t.outputObserved,TensorBinding.FP32Argument.observation]
  rw [pointer,output_word p S c heap other h.metadata order after ran]
  have value := values b head d hb hh hd
  rw [output_index c t heap h _ bound] at value
  exact congrArg (fun w => (⟨w⟩ : VeriTile.Torch.FP32.Word)) value

#axiomsClean native
#axiomsClean source_related
#axiomsClean exterior
#axiomsClean input_storage
#specNonCircular execute avoiding [KernelNative.target,KernelEndpoints.native]
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.KernelReadback
