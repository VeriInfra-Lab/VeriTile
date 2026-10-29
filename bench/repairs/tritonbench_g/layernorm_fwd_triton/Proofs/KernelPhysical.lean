import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.KernelABI

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.Physical
open VeriTile.Triton FP.Structural Address InputBinding
noncomputable section
local instance (p : Prop) : Decidable p := Classical.propDecidable p
abbrev Bytes := Nat → BitVec 8

def read32 (heap : Bytes) (address : Nat) : Torch.Word :=
  FP.Value.ofBits .fp32 ((heap address).toNat+256*(heap (address+1)).toNat+
    65536*(heap (address+2)).toNat+16777216*(heap (address+3)).toNat)

def readTerm (read : Nat → Torch.Word) (region : RegionName) (address : Nat) : NativeKernel.Term :=
  .leaf region address (read address).bits

def byteOf (word : NativeKernel.Word) (byte : Nat) : BitVec 8 :=
  BitVec.ofNat 8 (word.toNat/256^byte)

variable {shape : Shape} {st : Strides} {eps : Contract.Epsilon}

/-- Native leaves explicitly carry the original logical FP32 word and the
physical address observed by the same source load. -/
theorem logicalX_words (a : Arguments shape st eps) (read : Nat → Torch.Word) (row : Row shape) (col : Nat) :
    logicalX (readTerm read) a row col=VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.Term.leaf "X"
      (a.x.storage.address (a.x.physicalIndex (index3 row col)))
      (Torch.logicalValue read a.x a.xLaws (index3 row col)).bits := by
  simp [logicalX,readTerm,VeriTile.Torch.FP32.logicalValue,a.admitted.input.noNeg]

theorem logicalW_words (a : Arguments shape st eps) (read : Nat → Torch.Word) (row : Row shape) (col : Nat) :
    logicalW (readTerm read) a row col=VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.Term.leaf "W"
      (a.w.storage.address (a.w.physicalIndex (index2 row col)))
      (Torch.logicalValue read a.w a.wLaws (index2 row col)).bits := by
  simp [logicalW,readTerm,VeriTile.Torch.FP32.logicalValue,a.admitted.weight.noNeg]

def Hit (a : Arguments shape st eps) (byte : Nat) : Prop :=
  ∃ row : Row shape, ∃ col : Fin shape.columns, ∃ b : Fin 4,
    byte=address a.y (yOffset st row col.val)+b.val

def Outside (a : Arguments shape st eps) (byte : Nat) : Prop :=
  ∀ row : Row shape, ∀ col : Fin shape.columns,
    byte<address a.y (yOffset st row col.val) ∨ address a.y (yOffset st row col.val)+4≤byte

/-- Ordinary FP32 byte interpretation of the supplied strided output buffer.
Only source-written coordinates are represented; gaps remain caller bytes. -/
def outputBytes (I : NativeKernel.Instructions) (P : NativeKernel.Profile)
    (a : Arguments shape st eps) (before : Bytes) (s : State NativeKernel.Term) (byte : Nat) : BitVec 8 :=
  if Hit a byte then
    byteOf (NativeKernel.observe I P ((s.mem "Y" ((byte-a.y.dataPtr.toNat)/4)).read .fp32))
      ((byte-a.y.dataPtr.toNat)%4)
  else before byte

theorem coordinate_bytes (I : NativeKernel.Instructions) (P : NativeKernel.Profile)
    (a : Arguments shape st eps) (before : Bytes) (s : State NativeKernel.Term)
    (row : Row shape) (col : Fin shape.columns) (b : Fin 4) :
    outputBytes I P a before s (address a.y (yOffset st row col.val)+b.val)=
      byteOf (NativeKernel.observe I P ((s.mem "Y" (yOffset st row col.val)).read .fp32)) b.val := by
  have hit : Hit a (address a.y (yOffset st row col.val)+b.val) := ⟨row,col,b,rfl⟩
  have index : (address a.y (yOffset st row col.val)+b.val-a.y.dataPtr.toNat)/4=yOffset st row col.val := by
    unfold address; omega
  have byte : (address a.y (yOffset st row col.val)+b.val-a.y.dataPtr.toNat)%4=b.val := by
    unfold address; omega
  simp only [outputBytes,if_pos hit,index,byte]

/-- Every active typed store refines exactly one ordinary four-byte store.
There is no assumed final-buffer or result equality in this interpretation. -/
theorem store_bytes (I : NativeKernel.Instructions) (P : NativeKernel.Profile)
    (a : Arguments shape st eps) (before : Bytes) (s : State NativeKernel.Term)
    (row : Row shape) (col : Fin shape.columns) (v : NativeKernel.Term) (byte : Nat) :
    outputBytes I P a before (s.write "Y" (yOffset st row col.val) (.mk .fp32 v)) byte=
      if address a.y (yOffset st row col.val)≤byte ∧ byte<address a.y (yOffset st row col.val)+4 then
        byteOf (NativeKernel.observe I P v) (byte-address a.y (yOffset st row col.val))
      else outputBytes I P a before s byte := by
  by_cases hit : Hit a byte
  · have hhit := hit
    obtain ⟨r,c,b,hbyte⟩ := hit
    have ge : a.y.dataPtr.toNat≤byte := by rw [hbyte]; unfold address; omega
    simp only [outputBytes,if_pos hhit,State.write]
    by_cases same : (byte-a.y.dataPtr.toNat)/4=yOffset st row col.val
    · have inside : address a.y (yOffset st row col.val)≤byte ∧ byte<address a.y (yOffset st row col.val)+4 := by
        unfold address; omega
      have hb : (byte-a.y.dataPtr.toNat)%4=byte-address a.y (yOffset st row col.val) := by
        unfold address; omega
      simp [same,inside,hb]
    · have miss : ¬(address a.y (yOffset st row col.val)≤byte ∧ byte<address a.y (yOffset st row col.val)+4) := by
        unfold address; omega
      simp [same,miss]
  · have miss : ¬(address a.y (yOffset st row col.val)≤byte ∧ byte<address a.y (yOffset st row col.val)+4) := by
      intro inside
      apply hit
      exact ⟨row,col,⟨byte-address a.y (yOffset st row col.val),by omega⟩,by dsimp; omega⟩
    simp [outputBytes,hit,miss]

theorem frame (I : NativeKernel.Instructions) (P : NativeKernel.Profile)
    (a : Arguments shape st eps) (before : Bytes) (s : State NativeKernel.Term)
    (byte : Nat) (outside : Outside a byte) : outputBytes I P a before s byte=before byte := by
  have miss : ¬Hit a byte := by
    rintro ⟨row,col,b,hbyte⟩
    have h := outside row col
    omega
  simp [outputBytes,miss]

theorem input_frame (I : NativeKernel.Instructions) (P : NativeKernel.Profile)
    (a : Arguments shape st eps) (before : Bytes) (s : State NativeKernel.Term)
    (t : Torch.Tensor) (ht : t=a.x ∨ t=a.w) (i : Nat) (hi : i<t.storage.elements) (b : Fin 4) :
    outputBytes I P a before s (t.storage.address i+b.val)=before (t.storage.address i+b.val) :=
  frame I P a before s _ (fun row col => a.fresh t ht i hi row col b.val b.isLt)

theorem output_pointer (a : Arguments shape st eps) (row : Row shape) (col : Fin shape.columns) (b : Fin 4) :
    (KernelABI.pointer a.y.dataPtr row.1.val row.2.val col.val st.ySeq st.yHead st.yDim+
      BitVec.ofNat 64 b.val).toNat=address a.y (yOffset st row col.val)+b.val := by
  rw [BitVec.toNat_add,(KernelABI.active_pointers a row col).2.2,BitVec.toNat_ofNat,
    Nat.add_mod_mod]
  have h := (y_pointer a row col).2.1
  exact Nat.mod_eq_of_lt (by omega)

#axiomsClean coordinate_bytes
#axiomsClean store_bytes
#axiomsClean frame
#axiomsClean input_frame
#axiomsClean output_pointer
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.Physical
