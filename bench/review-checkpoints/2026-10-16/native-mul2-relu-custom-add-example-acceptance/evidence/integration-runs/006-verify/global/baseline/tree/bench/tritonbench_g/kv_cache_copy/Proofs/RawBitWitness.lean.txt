import bench.tritonbench_g.kv_cache_copy.RawBitCorrect
namespace VeriTile.Bench.TritonBenchG.KvCacheCopy.RawBits
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.RawBits
open KcacheCopyTriton.TypedCopy (Storage copyBackend)
open TypedCopy (Parameters)
noncomputable section

def crossParameters : Parameters where
  K := "K"
  KCache := "V"
  V := "V"
  VCache := "out"
  blockTables := "table"
  lengths := "lengths"
  splitX := 0
  strideKt := 0
  strideKh := 0
  strideKd := 1
  strideKcb := 0
  strideKch := 0
  strideKcsplit := 0
  strideKcs := 0
  strideBts := 0
  strideBtb := 0
  blockSize := 4
  width := 3
  strideVt := 0
  strideVh := 0
  strideVd := 1
  strideVcb := 0
  strideVch := 0
  strideVcs := 0
  strideVcd := 2
  headDim := 2

def crossMemory : Memory (BitVec 32) := fun r o =>
  if r="lengths" then .nat 2
  else if r="table" then .nat 0
  else if r="K" then if o=0 then .fp16 0x7d01 else if o=1 then .fp16 0x8000 else .fp16 0x3c00
  else if r="V" then if o=0 then .bf16 0x7f81 else if o=1 then .bf16 0x8000 else .bool Bool.true
  else .nat 999

def crossSeed : State (Carrier (BitVec 32)) where
  mem := fun _ _ => .mk .nat 55
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 37
  undef := fun d _ _ => defaultValue d

private theorem cross_injective : ∀ a b, TypedCopy.active crossParameters a →
    TypedCopy.active crossParameters b →
    vOutputOffset crossParameters crossMemory crossSeed.pids a =
      vOutputOffset crossParameters crossMemory crossSeed.pids b → a=b := by
  intro a b _ _ h
  simp [vOutputOffset,crossParameters,crossSeed] at h
  exact Prod.ext (Fin.ext (by omega)) (Subsingleton.elim _ _)

/-- Mixed half/bfloat16, two active lanes and an inactive tail. K overwrites
both V inputs with half cells before V stores. The latter must still write the
original bfloat16 NaN payload and -0 from the initial snapshot. Framing checks
both inactive destinations, including a cell with a non-floating tag. -/
theorem mixed_cross_alias_masked_source_success :
    ∃ final, FP.Structural.exec (copyBackend inert) (TypedCopy.source .fp16 .bf16 crossParameters)
      (state crossMemory crossSeed)=some final ∧
      decode (final.mem "V" 0)=RawCell.fp16 0x7d01 ∧
      decode (final.mem "V" 1)=RawCell.fp16 0x8000 ∧
      decode (final.mem "out" 0)=RawCell.bf16 0x7f81 ∧
      decode (final.mem "out" 2)=RawCell.bf16 0x8000 ∧
      final.mem "V" 2=encode (RawCell.bool (R:=BitVec 32) Bool.true) ∧
      final.mem "out" 4=encode (RawCell.nat (R:=BitVec 32) 999) := by
  let final := (TypedCopy.execution inert .fp16 .bf16 crossParameters (state crossMemory crossSeed)).val
  refine ⟨final,(TypedCopy.execution inert .fp16 .bf16 crossParameters (state crossMemory crossSeed)).property,
    ?_,?_,?_,?_,?_,?_⟩
  · have h := congrArg decode (k_readback inert .fp16 .bf16 crossParameters crossMemory crossSeed
      (⟨0,by decide⟩,PUnit.unit) (by decide) (by decide))
    simpa [final,crossParameters,crossMemory,crossSeed,inputOffset,outputOffset,past,blockId,
      readNat,readStorage,Region.cast] using h
  · have h := congrArg decode (k_readback inert .fp16 .bf16 crossParameters crossMemory crossSeed
      (⟨1,by decide⟩,PUnit.unit) (by decide) (by decide))
    simpa [final,crossParameters,crossMemory,crossSeed,inputOffset,outputOffset,past,blockId,
      readNat,readStorage,Region.cast] using h
  · have h := congrArg decode (v_readback inert .fp16 .bf16 crossParameters crossMemory crossSeed
      (⟨0,by decide⟩,PUnit.unit) (by decide) cross_injective)
    simpa [final,crossParameters,crossMemory,crossSeed,vInputOffset,vOutputOffset,past,blockId,
      readNat,readStorage,Region.cast] using h
  · have h := congrArg decode (v_readback inert .fp16 .bf16 crossParameters crossMemory crossSeed
      (⟨1,by decide⟩,PUnit.unit) (by decide) cross_injective)
    simpa [final,crossParameters,crossMemory,crossSeed,vInputOffset,vOutputOffset,past,blockId,
      readNat,readStorage,Region.cast] using h
  · have h := execution_frame inert .fp16 .bf16 crossParameters crossMemory crossSeed "V" 2
      (Or.inr (by intro i hi; simp [TypedCopy.active,crossParameters] at hi
                  simp [outputOffset,crossParameters]; omega)) (Or.inl (by decide))
    simpa [final,crossMemory] using h
  · have h := execution_frame inert .fp16 .bf16 crossParameters crossMemory crossSeed "out" 4
      (Or.inl (by decide))
      (Or.inr (by intro i hi; simp [TypedCopy.active,crossParameters] at hi
                  simp [vOutputOffset,crossParameters]; omega))
    simpa [final,crossMemory] using h

/-- Whole-memory execution also covers aliased output caches. At address 0,
K first writes a half NaN, then V overwrites it with the initial bfloat16 NaN.
No false simultaneous K readback is claimed when the old cache-separation
premise fails. This is an ordinary witness of the more general memory theorem. -/
theorem same_output_cache_v_writes_last :
    let p := { crossParameters with VCache := "V" }
    ∃ final, FP.Structural.exec (copyBackend inert) (TypedCopy.source .fp16 .bf16 p)
      (state crossMemory crossSeed)=some final ∧
      decode (final.mem "V" 0)=RawCell.bf16 0x7f81 ∧
      decode (final.mem "V" 2)=RawCell.bf16 0x8000 := by
  dsimp only
  let p := { crossParameters with VCache := "V" }
  have hinj : ∀ a b, TypedCopy.active p a → TypedCopy.active p b →
      vOutputOffset p crossMemory crossSeed.pids a = vOutputOffset p crossMemory crossSeed.pids b → a=b := by
    intro a b _ _ h
    simp [p,vOutputOffset,crossParameters,crossSeed] at h
    exact Prod.ext (Fin.ext (by omega)) (Subsingleton.elim _ _)
  let final := (TypedCopy.execution inert .fp16 .bf16 p (state crossMemory crossSeed)).val
  refine ⟨final,(TypedCopy.execution inert .fp16 .bf16 p (state crossMemory crossSeed)).property,?_,?_⟩
  · have h := congrArg decode (v_readback inert .fp16 .bf16 p crossMemory crossSeed
      (⟨0,by decide⟩,PUnit.unit) (by decide) hinj)
    simpa [final,p,crossParameters,crossMemory,crossSeed,vInputOffset,vOutputOffset,past,blockId,
      readNat,readStorage,Region.cast] using h
  · have h := congrArg decode (v_readback inert .fp16 .bf16 p crossMemory crossSeed
      (⟨1,by decide⟩,PUnit.unit) (by decide) hinj)
    simpa [final,p,crossParameters,crossMemory,crossSeed,vInputOffset,vOutputOffset,past,blockId,
      readNat,readStorage,Region.cast] using h

/-- No active lane means actual execution succeeds and every tagged cell is
unchanged. No assumption about a numeric masked-off literal/undef is needed. -/
theorem inactive_source_success {R : Type} [Inhabited R] (kd vd : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (hoff : ∀ i, ¬ TypedCopy.active p i) :
    ∃ final, FP.Structural.exec (copyBackend inert) (TypedCopy.source kd vd p)
      (state m seed)=some final ∧ final.mem=encodeMemory m := by
  refine ⟨(TypedCopy.execution inert kd vd p (state m seed)).val,
    (TypedCopy.execution inert kd vd p (state m seed)).property,?_⟩
  funext r o
  apply execution_frame
  · right; intro i hi; exact (hoff i hi).elim
  · right; intro i hi; exact (hoff i hi).elim

theorem zero_width_source_success {R : Type} [Inhabited R] (kd vd : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (hzero : p.width=0) :
    ∃ final, FP.Structural.exec (copyBackend inert) (TypedCopy.source kd vd p)
      (state m seed)=some final ∧ final.mem=encodeMemory m := by
  apply inactive_source_success
  intro i _
  have hi := i.1.isLt
  omega

/-- With one active lane, a zero V component stride satisfies the exact old
injectivity condition. The public theorem does not strengthen it to stride>0. -/
theorem singleton_active_zero_stride :
    let p := { crossParameters with headDim := 1, strideVcd := 0 }
    p.strideVcd=0 ∧ (∀ a b, TypedCopy.active p a → TypedCopy.active p b →
      vOutputOffset p crossMemory crossSeed.pids a = vOutputOffset p crossMemory crossSeed.pids b → a=b) := by
  dsimp only
  refine ⟨rfl,?_⟩
  intro a b ha hb _
  simp [TypedCopy.active,crossParameters] at ha hb
  exact Prod.ext (Fin.ext (by omega)) (Subsingleton.elim _ _)

#axiomsClean mixed_cross_alias_masked_source_success
#axiomsClean same_output_cache_v_writes_last
#axiomsClean inactive_source_success
#axiomsClean zero_width_source_success
#axiomsClean singleton_active_zero_stride
end
end VeriTile.Bench.TritonBenchG.KvCacheCopy.RawBits
