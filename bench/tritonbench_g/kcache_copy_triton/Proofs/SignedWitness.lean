import bench.tritonbench_g.kcache_copy_triton.SignedCorrect
namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.SignedWitness
open VeriTile.Triton FP.Structural
open SignedCopy (Parameters)
open RawBits SignedRawBits
noncomputable section
set_option linter.unusedSimpArgs false

/-- Two tokens per sequence straddle a cache-block boundary, with a nonzero
sequence and split coordinate. Source/cache regions deliberately alias. -/
def parameters : Parameters where
  K := "data"
  KCache := "data"
  blockTables := ⟨"blocks"⟩
  lengths := ⟨"lengths"⟩
  nTokens := 2
  strideKt := 4
  strideKh := 2
  strideKd := 1
  strideKcb := 16
  strideKch := 8
  strideKcsplit := 4
  strideKcs := 2
  strideBts := 2
  strideBtb := 1
  blockSize := 4
  width := 2

def ids (token : Nat) : Nat → Nat := fun dim =>
  if dim = 0 then token else if dim = 2 then 1 else 0

def memory : Memory (BitVec 32) := fun r o =>
  if r = "lengths" then .int 5
  else if r = "blocks" then .int (if o = 2 then 3 else 7)
  else .fp16 (if o % 2 = 0 then 0x7e13 else 0x8000)

def seed (token : Nat) : State (Carrier (BitVec 32)) where
  mem := fun _ _ => .mk .nat 0
  regs := fun _ _ _ => none
  pids := ids token
  numPids := fun _ => 8
  undef := fun d _ _ => defaultValue d

example : SignedCopy.tokenShift parameters (ids 2) = -2 := rfl
example : SignedCopy.tokenShift parameters (ids 3) = -1 := rfl
example : SignedRawBits.past parameters memory (ids 2) = 3 := by
  simp [parameters, memory, ids, SignedRawBits.past, SignedRawBits.outputOffset,
    SignedRawBits.blockId, SignedRawBits.readInt, SignedCopy.sequence,
    SignedCopy.tokenShift, SignedCopy.unsigned64, Region.cast]
example : SignedRawBits.past parameters memory (ids 3) = 4 := by
  simp [parameters, memory, ids, SignedRawBits.past, SignedRawBits.outputOffset,
    SignedRawBits.blockId, SignedRawBits.readInt, SignedCopy.sequence,
    SignedCopy.tokenShift, SignedCopy.unsigned64, Region.cast]
example : SignedRawBits.outputOffset parameters memory (ids 2) (⟨0, by decide⟩, ()) = 58 := by
  simp [parameters, memory, ids, SignedRawBits.past, SignedRawBits.outputOffset,
    SignedRawBits.blockId, SignedRawBits.readInt, SignedCopy.sequence,
    SignedCopy.tokenShift, SignedCopy.unsigned64, Region.cast]
example : SignedRawBits.outputOffset parameters memory (ids 3) (⟨0, by decide⟩, ()) = 116 := by
  simp [parameters, memory, ids, SignedRawBits.past, SignedRawBits.outputOffset,
    SignedRawBits.blockId, SignedRawBits.readInt, SignedCopy.sequence,
    SignedCopy.tokenShift, SignedCopy.unsigned64, Region.cast]

/-- The first token preserves a NaN payload despite source/cache aliasing. -/
theorem first_token_nan_payload :
    (SignedCopy.execution inert .fp16 parameters
      (SignedRawBits.state memory (seed 2))).val.mem "data" 58 =
        encode (RawCell.fp16 0x7e13 : RawCell (BitVec 32)) := by
  have h := SignedRawBits.execution_readback inert .fp16 parameters memory (seed 2)
    (⟨0, by decide⟩, ())
  simpa [parameters, memory, ids, SignedRawBits.past, SignedRawBits.outputOffset,
    SignedRawBits.blockId, SignedRawBits.readInt, SignedCopy.sequence,
    SignedCopy.tokenShift, SignedCopy.unsigned64, Region.cast, seed,
    SignedRawBits.inputOffset, readStorage] using h

/-- The next token moves to the next block-table entry and preserves -0. -/
theorem second_token_negative_zero :
    (SignedCopy.execution inert .fp16 parameters
      (SignedRawBits.state memory (seed 3))).val.mem "data" 117 =
        encode (RawCell.fp16 0x8000 : RawCell (BitVec 32)) := by
  have h := SignedRawBits.execution_readback inert .fp16 parameters memory (seed 3)
    (⟨1, by decide⟩, ())
  simpa [parameters, memory, ids, SignedRawBits.past, SignedRawBits.outputOffset,
    SignedRawBits.blockId, SignedRawBits.readInt, SignedCopy.sequence,
    SignedCopy.tokenShift, SignedCopy.unsigned64, Region.cast, seed,
    SignedRawBits.inputOffset, readStorage] using h

#axiomsClean first_token_nan_payload
#axiomsClean second_token_negative_zero
end
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.SignedWitness
