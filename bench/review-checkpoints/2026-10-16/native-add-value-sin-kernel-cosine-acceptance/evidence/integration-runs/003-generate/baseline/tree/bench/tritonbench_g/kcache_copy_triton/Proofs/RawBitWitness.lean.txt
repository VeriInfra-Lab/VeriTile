import bench.tritonbench_g.kcache_copy_triton.RawBitCorrect
namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.RawBits
open VeriTile.Triton FP.Structural
open TypedCopy (Storage Parameters)
noncomputable section

def overlapParameters : Parameters where
  K := "same"
  KCache := "same"
  blockTables := "table"
  lengths := "lengths"
  splitX := 0
  strideKt := 0
  strideKh := 0
  strideKd := 1
  strideKcb := 0
  strideKch := 0
  strideKcsplit := 0
  strideKcs := 1
  strideBts := 0
  strideBtb := 0
  blockSize := 4
  width := 2

def overlapMemory : Memory (BitVec 32) := fun r o =>
  if r="lengths" then .nat 2
  else if r="table" then .nat 0
  else if o=0 then .fp16 0x7d01
  else if o=1 then .fp16 0x8000
  else .fp16 0x3c00

def overlapSeed : State (Carrier (BitVec 32)) where
  mem := fun _ _ => .mk .nat 55
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 37
  undef := fun d _ _ => defaultValue d

/-- Nonempty overlap: initial source [sNaN-payload, -0] at offsets [0,1] is
copied to [1,2] in the SAME region. Output[2] is the original -0, not the NaN
just written to Output[1]. This checks initial-snapshot rather than live read.
Arbitrary-register/undef independence is the separate universal theorem. -/
theorem overlapping_nonempty_source_success :
    ∃ final, FP.Structural.exec (TypedCopy.copyBackend inert)
      (TypedCopy.source .fp16 overlapParameters) (state overlapMemory overlapSeed)=some final ∧
      decode (final.mem "same" 1)=RawCell.fp16 0x7d01 ∧
      decode (final.mem "same" 2)=RawCell.fp16 0x8000 ∧
      final.mem "same" 0=encode (overlapMemory "same" 0) ∧
      final.mem "lengths" 0=encode (RawCell.nat (R:=BitVec 32) 2) := by
  let final := (TypedCopy.execution inert .fp16 overlapParameters (state overlapMemory overlapSeed)).val
  refine ⟨final,(TypedCopy.execution inert .fp16 overlapParameters (state overlapMemory overlapSeed)).property,
    ?_,?_,?_,?_⟩
  · have h := congrArg decode (execution_readback inert .fp16 overlapParameters overlapMemory overlapSeed
      (⟨0,by decide⟩,PUnit.unit))
    simpa [final,overlapParameters,overlapMemory,overlapSeed,inputOffset,outputOffset,past,blockId,
      readNat,readStorage,Region.cast] using h
  · have h := congrArg decode (execution_readback inert .fp16 overlapParameters overlapMemory overlapSeed
      (⟨1,by decide⟩,PUnit.unit))
    simpa [final,overlapParameters,overlapMemory,overlapSeed,inputOffset,outputOffset,past,blockId,
      readNat,readStorage,Region.cast] using h
  · apply execution_frame
    right
    intro i
    simp [overlapParameters,overlapMemory,overlapSeed,outputOffset,past,blockId,readNat,Region.cast]
    omega
  · have h := execution_frame inert .fp16 overlapParameters overlapMemory overlapSeed "lengths" 0
      (Or.inl (by decide))
    simpa [final,overlapMemory] using h

/-- Empty widths are still in the original contract; the representation
bridge neither adds a positive-width premise nor treats failed runs as equal. -/
theorem zero_width_source_success {R : Type} [Inhabited R] (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (hzero : p.width=0) :
    ∃ final, FP.Structural.exec (TypedCopy.copyBackend inert) (TypedCopy.source d p)
      (state m seed)=some final ∧ final.mem=encodeMemory m := by
  obtain ⟨final,he,hm,_⟩ := complete d p m seed
  refine ⟨final,he,?_⟩
  have hempty : TileShape.allIndices [p.width] = [] := by
    apply List.eq_nil_iff_forall_not_mem.mpr
    intro i _
    have hi := i.1.isLt
    omega
  simpa only [reference,hempty,List.foldl_nil] using hm

/-- Width/tag mismatch is not silently called a bijection. This noncanonical
Structural half cell has extra high bits; raw decoding forgets them. The main
source theorem proves all actual outputs remain canonical, so it never relies
on the false unrestricted reverse-round-trip. -/
theorem unrestricted_reverse_roundtrip_is_false :
    let c : Cell (Carrier (BitVec 32)) := .mk .fp16 (.word 0x10000)
    encode (decode c) ≠ c := by
  dsimp only
  intro h
  have h' := congrArg (fun c : Cell (Carrier (BitVec 32)) => c.read .fp16) h
  change Carrier.word (0 : BitVec 32) = Carrier.word 0x10000 at h'
  have hb : (0 : BitVec 32) = 0x10000 := Carrier.word.inj h'
  exact (by decide : (0 : BitVec 32) ≠ 0x10000) hb

/-- NaN payload bits and zero signs remain distinct in the representation. -/
theorem payloads_not_quotiented :
    encode (RawCell.fp16 (R:=BitVec 32) 0x7d01) ≠ encode (.fp16 0x7e01) ∧
      encode (RawCell.fp16 (R:=BitVec 32) 0x8000) ≠ encode (.fp16 0) := by
  constructor
  · intro h
    have hb := RawCell.fp16.inj (encode_injective h)
    exact (by decide : (0x7d01 : BitVec 16) ≠ 0x7e01) hb
  · intro h
    have hb := RawCell.fp16.inj (encode_injective h)
    exact (by decide : (0x8000 : BitVec 16) ≠ 0) hb

#axiomsClean overlapping_nonempty_source_success
#axiomsClean zero_width_source_success
#axiomsClean unrestricted_reverse_roundtrip_is_false
#axiomsClean payloads_not_quotiented
end
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.RawBits
