import bench.tritonbench_g.matrix_transpose.Proofs.FPSoftware

namespace VeriTile.Bench.TritonBenchG.MatrixTranspose.FPBits
open VeriTile.Triton FP.Structural
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

/-- A rectangular 2×4 tile with in-place aliasing and output row gaps. The
early store at offset 1 overwrites an input needed by a later output lane. -/
def witnessParameters : Parameters where
  input := "shared"
  output := "shared"
  sx := 4
  sy := 1
  ox := 3
  oy := 1
  m := 2
  d := 4

def witnessWord : Nat → Word
  | 0 => 0x8000 -- negative zero
  | 1 => 0x7c01 -- signaling NaN payload
  | 2 => 0xfc00 -- negative infinity
  | 3 => 0x0001 -- minimum positive subnormal
  | 4 => 0x7e55 -- quiet NaN payload
  | 5 => 0x3c00 -- one
  | 6 => 0x7bff -- maximum finite
  | 7 => 0x0000 -- positive zero
  | _ => 0x55aa

def witnessState : State Word where
  mem := fun r o => if r = "shared" then .mk .fp16 (witnessWord o) else .mk .int (-37)
  regs := fun _ _ _ => none
  pids := fun _ => 127
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

def witnessAddresses : List Nat := [0,1,3,4,6,7,9,10]
def witnessExpected : List Nat := [0x8000,0x7e55,0x7c01,0x3c00,0xfc00,0x7bff,0x0001,0x0000]

theorem witness_injective : Function.Injective (outputOffset witnessParameters) := by
  intro a b h
  apply Prod.ext
  · apply Fin.ext
    have ha := a.2.1.isLt
    have hb := b.2.1.isLt
    simp only [outputOffset, witnessParameters] at h ha hb
    omega
  · apply Prod.ext
    · apply Fin.ext
      have ha := a.2.1.isLt
      have hb := b.2.1.isLt
      simp only [outputOffset, witnessParameters] at h ha hb
      omega
    · exact Subsingleton.elim _ _

theorem witness_reference_words :
    witnessAddresses.map (fun o =>
      ((referenceMemory witnessParameters witnessState "shared" o).read .fp16).toNat) =
      witnessExpected := by
  decide +kernel

theorem witness_gap_words :
    [2,5,8].map (fun o =>
      ((referenceMemory witnessParameters witnessState "shared" o).read .fp16).toNat) =
      [0xfc00,0x3c00,0x55aa] := by
  decide +kernel

/-- Nonempty actual execution of both complete source versions, with arbitrary
NaN/signed-zero/subnormal payloads, overlap and observable output row gaps. -/
theorem witness_executes (v : Bool) :
    ∃ t, exec unusedPrimitives (source v witnessParameters) witnessState = some t ∧
      witnessAddresses.map (fun o => ((t.mem "shared" o).read .fp16).toNat) =
        witnessExpected ∧
      [2,5,8].map (fun o => ((t.mem "shared" o).read .fp16).toNat) =
        [0xfc00,0x3c00,0x55aa] ∧
      t.mem "other" 42 = .mk .int (-37) := by
  refine ⟨execution witnessParameters witnessState,
    execute unusedPrimitives v witnessParameters witnessState, ?_, ?_, ?_⟩
  · rw [execution_memory]
    exact witness_reference_words
  · rw [execution_memory]
    exact witness_gap_words
  · exact frame witnessParameters witnessState "other" 42 (Or.inl (by decide))

#axiomsClean witness_injective
#axiomsClean witness_reference_words
#axiomsClean witness_gap_words
#axiomsClean witness_executes
end VeriTile.Bench.TritonBenchG.MatrixTranspose.FPBits
