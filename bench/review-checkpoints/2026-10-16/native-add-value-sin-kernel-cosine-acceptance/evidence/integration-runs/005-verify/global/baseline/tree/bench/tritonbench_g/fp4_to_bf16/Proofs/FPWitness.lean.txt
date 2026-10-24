import bench.tritonbench_g.fp4_to_bf16.FPCorrect
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def witnessBytes (o : Nat) : BitVec 8 :=
  if o=0 then 0x80 else if o=1 then 0x21 else if o=2 then 0xff else 0xcc

def byteState (bytes : Nat → BitVec 8) (pid undefNat : Nat) : State Word :=
  { mem := fun _ o => .mk .nat (bytes o).toNat
    regs := fun _ _ _ => none
    pids := fun _ => pid
    numPids := fun _ => pid+1
    undef := fun d _ _ => match d with
      | .nat => undefNat
      | d => defaultValue d }

def aliasParameters : Parameters := ⟨"bytes","bytes",3,4⟩
def sixExpected (i : Nat) : BitVec 16 :=
  if i=0 then 0x8000 else if i=1 then 0 else if i=2 then 0x3f80
  else if i=3 then 0x3f00 else 0xc0c0

theorem byteState_typed (p : Parameters) (bytes : Nat → BitVec 8) (pid u : Nat) :
    ActiveBytes p (byteState bytes pid u) := by
  intro i hi
  exact ⟨bytes (inputOffset p (byteState bytes pid u) i.val), rfl⟩

/-- Actual nonempty source execution, with input/output aliasing, a tail,
negative zero, high-nibble-first order and arbitrary masked undef. -/
theorem alias_tail_execution (rest : Algebra Word) (u : Nat) :
    ∃ final, FP.Structural.exec (backend rest) (source aliasParameters)
        (byteState witnessBytes 0 u) = some final ∧
      (∀ i : Fin 6, final.mem "bytes" i.val = .mk .bf16 (embed ⟨sixExpected i.val⟩)) ∧
      final.mem "bytes" 6 = .mk .nat 204 ∧
      final.mem "bytes" 7 = .mk .nat 204 ∧
      final.mem "unrelated" 0 = .mk .nat 128 := by
  obtain ⟨final,he,hm,hv,hf⟩ := FPCorrect.fp4_to_bf16_unscaled_machine_correctness
    rest aliasParameters (byteState witnessBytes 0 u) (byteState_typed _ _ _ _)
  refine ⟨final,he,?_,?_,?_,?_⟩
  · intro i
    let j : TileIndex [aliasParameters.block*2] := (⟨i.val, by have := i.isLt; dsimp [aliasParameters]; omega⟩, PUnit.unit)
    have hj : active aliasParameters (byteState witnessBytes 0 u) j := by
      dsimp [active, outputOffset, aliasParameters, byteState, j]
      simpa only [Nat.zero_add] using i.isLt
    have h := hv j hj
    fin_cases i <;>
      simpa [j, outputOffset, expectedValue, initialNibble, inputOffset,
        aliasParameters, byteState, witnessBytes, sixExpected, Bits.expectedBits,
        Bits.high, Bits.low] using h
  · have h := hf "bytes" 6 (Or.inr (by
      intro j hj; dsimp [active, outputOffset, aliasParameters, byteState] at *; omega))
    simpa [byteState, witnessBytes] using h
  · have h := hf "bytes" 7 (Or.inr (by
      intro j hj; dsimp [active, outputOffset, aliasParameters, byteState] at *; omega))
    simpa [byteState, witnessBytes] using h
  · have h := hf "unrelated" 0 (Or.inl (by decide))
    simpa [byteState, witnessBytes] using h

def hostParameters : Parameters := ⟨"bytes","bytes",768,512⟩
/-- Host BLOCK=512, nonzero program id and a half-active final block: all
256 possible packed bytes occur in its active reads. The unused input half
may return any natural value, and the final inactive output remains framed. -/
theorem host512_tail_execution (rest : Algebra Word) (u : Nat) :
    ∃ final, FP.Structural.exec (backend rest) (source hostParameters)
        (byteState (BitVec.ofNat 8) 1 u) = some final ∧
      (∀ j : Fin 512, final.mem "bytes" (1024+j.val) = .mk .bf16
        (embed ⟨Bits.expectedBits
          (if j.val%2=0 then Bits.high (BitVec.ofNat 8 (512+j.val/2))
           else Bits.low (BitVec.ofNat 8 (512+j.val/2)))⟩)) ∧
      final.mem "bytes" 1536 = .mk .nat 0 := by
  obtain ⟨final,he,hm,hv,hf⟩ := FPCorrect.fp4_to_bf16_unscaled_machine_correctness
    rest hostParameters (byteState (BitVec.ofNat 8) 1 u) (byteState_typed _ _ _ _)
  refine ⟨final,he,?_,?_⟩
  · intro j
    let i : TileIndex [hostParameters.block*2] := (⟨j.val, by have := j.isLt; dsimp [hostParameters]; omega⟩, PUnit.unit)
    have hi : active hostParameters (byteState (BitVec.ofNat 8) 1 u) i := by
      dsimp [active, outputOffset, hostParameters, byteState, i]
      have := j.isLt; omega
    have h := hv i hi
    have hmod (v : Nat) : BitVec.ofNat 8 (v % 256) = BitVec.ofNat 8 v := by
      apply BitVec.eq_of_toNat_eq
      simp
    simpa [i, outputOffset, expectedValue, initialNibble, inputOffset,
      hostParameters, byteState, hmod] using h
  · have h := hf "bytes" 1536 (Or.inr (by
      intro j hj; dsimp [active, outputOffset, hostParameters, byteState] at *; omega))
    simpa [byteState] using h

#axiomsClean byteState_typed
#axiomsClean alias_tail_execution
#axiomsClean host512_tail_execution
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
