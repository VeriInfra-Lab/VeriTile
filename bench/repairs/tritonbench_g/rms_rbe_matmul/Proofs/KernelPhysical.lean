import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelBinding

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelPhysical
open VeriTile.Triton FP.Structural
open KernelBinding (Bytes OriginalTensor)
open KernelNative (Word Term Instructions Schedule)
abbrev NativeSchedule := KernelNative.Schedule
open Contract (Inputs InputContract)
noncomputable section
local instance (p : Prop) : Decidable p := Classical.propDecidable p
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def storeWidth (i : Inputs) : Nat := if KernelBinding.fp8 i then 1 else 2
def start (i : Inputs) (a : Checked.Allocation i) := a.out.base+storeWidth i*a.out.offset
def address (i : Inputs) (a : Checked.Allocation i) (off : Nat) := start i a+storeWidth i*off
def byteOf (w : Word) (k : Nat) : BitVec 8 := BitVec.ofNat 8 (w.toNat/256^k)
def readOutput (i : Inputs) (s : State Term) (off : Nat) : Term :=
  if KernelBinding.fp8 i then (s.mem "OUT" off).read .f8e5 else (s.mem "OUT" off).read .fp16
def Hit (i : Inputs) (a : Checked.Allocation i) (byte : Nat) : Prop :=
  ∃ off : Fin a.out.metadata.numel, ∃ k : Fin (storeWidth i), byte=address i a off.val+k.val
def Outside (i : Inputs) (a : Checked.Allocation i) (byte : Nat) : Prop :=
  byte<start i a ∨ start i a+storeWidth i*a.out.metadata.numel≤byte

/-- Ordinary little-endian interpretation of the supplied output Tensor.
Only its logical 1-byte E5 / 2-byte FP16 destination span is writable. -/
def outputBytes (I : Instructions) (S : NativeSchedule) (i : Inputs) (a : Checked.Allocation i)
    (before : Bytes) (s : State Term) (byte : Nat) : BitVec 8 :=
  if Hit i a byte then
    byteOf (KernelNative.eval I S (readOutput i s ((byte-start i a)/storeWidth i)))
      ((byte-start i a)%storeWidth i)
  else before byte

theorem width_cases (i : Inputs) : storeWidth i=1 ∨ storeWidth i=2 := by
  unfold storeWidth; split <;> simp
theorem width_positive (i : Inputs) : 0<storeWidth i := by rcases width_cases i with h|h <;> omega
theorem output_width (i : Inputs) (h : InputContract i) (a : Checked.Allocation i) :
    RmsMatmulRbe.Contract.width a.out.metadata.dtype=storeWidth i := by
  have hd := (Checked.facts i h).weightAdmitted.2.2.1
  simp only [RmsMatmulRbe.Contract.weightTypes,List.mem_cons,List.not_mem_nil,or_false] at hd
  rcases hd with hd|hd <;>
    simp [a.outDType,hd,storeWidth,KernelBinding.fp8,RmsMatmulRbe.Contract.width,
      RmsMatmulRbe.Contract.fp32,RmsMatmulRbe.Contract.fp16,RmsMatmulRbe.Contract.int8]

theorem coordinate_bytes (I : Instructions) (S : NativeSchedule) (i : Inputs) (a : Checked.Allocation i)
    (before : Bytes) (s : State Term) (off : Fin a.out.metadata.numel) (k : Fin (storeWidth i)) :
    outputBytes I S i a before s (address i a off.val+k.val)=
      byteOf (KernelNative.eval I S (readOutput i s off.val)) k.val := by
  have hit : Hit i a (address i a off.val+k.val) := ⟨off,k,rfl⟩
  have hk := k.isLt
  have index : (address i a off.val+k.val-start i a)/storeWidth i=off.val := by
    rcases width_cases i with hw|hw <;> simp only [address,hw] at hk ⊢ <;> omega
  have lane : (address i a off.val+k.val-start i a)%storeWidth i=k.val := by
    rcases width_cases i with hw|hw <;> simp only [address,hw] at hk ⊢ <;> omega
  simp only [outputBytes,if_pos hit,index,lane]

theorem read_stored (i : Inputs) (s : State Term) (off j : Nat) (v : Term) :
    readOutput i (s.write "OUT" off
      (.mk (KernelSource.outputDType (KernelBinding.fp8 i)).toTileDType
        (ofFloat (KernelSource.outputDType (KernelBinding.fp8 i)) v))) j=
      if j=off then v else readOutput i s j := by
  by_cases hj : j=off <;> cases hf : KernelBinding.fp8 i <;>
    simp [readOutput,hf,hj,State.write,KernelSource.outputDType,ofFloat]

/-- Every actual typed store refines exactly the corresponding 1/2-byte store.
The theorem assumes no final buffer or kernel-output equality. -/
theorem store_bytes (I : Instructions) (S : NativeSchedule) (i : Inputs) (a : Checked.Allocation i)
    (before : Bytes) (s : State Term) (off : Fin a.out.metadata.numel) (v : Term) (byte : Nat) :
    outputBytes I S i a before (s.write "OUT" off.val
      (.mk (KernelSource.outputDType (KernelBinding.fp8 i)).toTileDType
        (ofFloat (KernelSource.outputDType (KernelBinding.fp8 i)) v))) byte=
      if address i a off.val≤byte ∧ byte<address i a off.val+storeWidth i then
        byteOf (KernelNative.eval I S v) (byte-address i a off.val)
      else outputBytes I S i a before s byte := by
  by_cases hit : Hit i a byte
  · have hit' := hit
    obtain ⟨other,k,hbyte⟩ := hit
    have hk := k.isLt
    have ge : start i a≤byte := by rw [hbyte]; unfold address; omega
    simp only [outputBytes,if_pos hit',read_stored]
    by_cases same : (byte-start i a)/storeWidth i=off.val
    · have inside : address i a off.val≤byte ∧ byte<address i a off.val+storeWidth i := by
        rcases width_cases i with hw|hw <;> simp only [address,hw] at same ge ⊢ <;> omega
      have lane : (byte-start i a)%storeWidth i=byte-address i a off.val := by
        rcases width_cases i with hw|hw <;> simp only [address,hw] at same ge inside ⊢ <;> omega
      simp only [if_pos same,if_pos inside,lane]
    · have miss : ¬(address i a off.val≤byte ∧ byte<address i a off.val+storeWidth i) := by
        rcases width_cases i with hw|hw <;> simp only [address,hw] at same ge ⊢ <;> omega
      simp only [if_neg same,if_neg miss]
  · have miss : ¬(address i a off.val≤byte ∧ byte<address i a off.val+storeWidth i) := by
      intro inside
      apply hit
      exact ⟨off,⟨byte-address i a off.val,by omega⟩,by dsimp; omega⟩
    simp only [outputBytes,if_neg hit,if_neg miss]

theorem frame (I : Instructions) (S : NativeSchedule) (i : Inputs) (a : Checked.Allocation i)
    (before : Bytes) (s : State Term) (byte : Nat) (outside : Outside i a byte) :
    outputBytes I S i a before s byte=before byte := by
  have miss : ¬Hit i a byte := by
    rintro ⟨off,k,hbyte⟩
    have ho := off.isLt
    have hk := k.isLt
    unfold Outside at outside
    unfold address at hbyte
    rcases width_cases i with hw|hw <;> simp only [hw] at ho hk outside hbyte ⊢ <;> omega
  simp only [outputBytes,if_neg miss]

theorem original_input_frame (I : Instructions) (S : NativeSchedule) (i : Inputs) (h : InputContract i)
    (a : Checked.Allocation i) (before : Bytes) (s : State Term)
    (t : OriginalTensor) (ht : t∈i.reads) (j : Nat) (hj : j<t.bytes) :
    outputBytes I S i a before s (t.base+j)=before (t.base+j) := by
  apply frame
  have disjoint := a.fresh t (by simp [ht])
  have span := a.outOrdinary.span
  rw [output_width i h a] at span
  have endBound : start i a+storeWidth i*a.out.metadata.numel≤a.out.base+a.out.bytes := by
    unfold start
    nlinarith
  have low : a.out.base≤start i a := by unfold start; omega
  unfold RmsMatmulRbe.Contract.disjoint RmsMatmulRbe.Contract.overlap at disjoint
  unfold Outside
  omega

theorem pointer_exact (i : Inputs) (h : InputContract i) (a : Checked.Allocation i)
    (off : Fin a.out.metadata.numel) : (RmsMatmulRbe.Checked.pointer64 a.out off.val).toNat=address i a off.val := by
  rw [KernelBinding.pointer_exact a.out a.outOrdinary _ off.isLt]
  simp only [KernelBinding.address,output_width i h a,address,start,Nat.mul_add,Nat.add_assoc]

#axiomsClean coordinate_bytes
#axiomsClean store_bytes
#axiomsClean original_input_frame
#axiomsClean pointer_exact
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelPhysical
