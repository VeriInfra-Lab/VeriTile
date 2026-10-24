import bench.tritonbench_g.fused_rotary_embedding.Proofs.FPV
namespace VeriTile.Bench.TritonBenchG.FusedRotaryEmbedding.FPV
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy (copyBackend)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

def witnessParameters : Parameters where
  input := "v"
  output := "cache"
  blockTables := ⟨"table"⟩
  lengths := ⟨"lengths"⟩
  groups := 2
  tokenStride := 10
  headStride := 4
  dimensionStride := 1
  cacheBlockStride := 100
  cacheHeadStride := 20
  cacheSlotStride := 4
  cacheDimensionStride := 1
  tableSequenceStride := 3
  tableBlockStride := 2
  blockSize := 4
  width := 4

def payload (i : Fin 4) : BitVec 32 :=
  if i.val = 0 then 0x7f800001 else if i.val = 1 then 0x80000000
  else if i.val = 2 then 0x7f800000 else 1

/-- Nonzero token/head/block/slot addresses and arbitrary raw floating payload,
including signaling-NaN bits, negative zero, infinity and a subnormal. -/
def witnessInitial : State (BitVec 32) where
  mem := fun r o =>
    if r = ("lengths" : RegionName) ∧ o = 1 then .mk .nat 6
    else if r = ("table" : RegionName) ∧ o = 5 then .mk .nat 3
    else if r = ("v" : RegionName) then .mk .fp32
      (if o = 14 then 0x7f800001 else if o = 15 then 0x80000000
       else if o = 16 then 0x7f800000 else 1)
    else .mk .fp32 0xdeadbeef
  regs := fun _ _ _ => none
  pids := fun a => if a = 0 then 2 else 1
  numPids := fun _ => 3
  undef := fun d _ _ => defaultValue d

theorem nonempty_bits (rest : Algebra (BitVec 32)) :
    ∃ final, FP.Structural.exec (copyBackend rest) (source witnessParameters)
      witnessInitial = some final ∧
      (∀ i : Fin 4, final.mem "cache" (324 + i.val) = .mk .fp32 (payload i)) ∧
      final.mem "cache" 323 = witnessInitial.mem "cache" 323 := by
  obtain ⟨final, he, hp⟩ := correctness rest witnessParameters witnessInitial
    (fun _ => output_injective witnessParameters witnessInitial (by decide))
  refine ⟨final, he, ?_, ?_⟩
  · intro i
    have hv := hp.1 (i, PUnit.unit) (by decide)
    fin_cases i <;> simpa [outputOffset, inputOffset, blockId, past, witnessInitial,
      witnessParameters, Region.cast, payload] using hv
  · apply hp.2 "cache" 323
    right
    intro i _
    simp [outputOffset, blockId, past, witnessInitial, witnessParameters, Region.cast]
    omega

/-- A nonleader executes successfully without changing any memory cell. -/
theorem inactive_frame {α : Type} [Inhabited α] (rest : Algebra α)
    (p : Parameters) (s : State α) (hinactive : ¬ active p s) :
    ∃ final, FP.Structural.exec (copyBackend rest) (source p) s = some final ∧ final.mem = s.mem := by
  refine ⟨(execution rest p s).val, (execution rest p s).property, ?_⟩
  rw [execution_memory]
  simp [copied, hinactive]

#axiomsClean nonempty_bits
#axiomsClean inactive_frame
end
end VeriTile.Bench.TritonBenchG.FusedRotaryEmbedding.FPV
