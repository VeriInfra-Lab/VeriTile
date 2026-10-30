import bench.tritonbench_g.fp4_to_bf16.Proofs.FPSource
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def inputOffset (p : Parameters) (s : State Word) (i : Nat) := s.pids 0*p.block+i
def outputOffset (p : Parameters) (s : State Word) (j : TileIndex [p.block*2]) :=
  s.pids 0*(p.block*2)+j.1.val
abbrev active (p : Parameters) (s : State Word) (j : TileIndex [p.block*2]) :=
  outputOffset p s j < p.count*2

/-- Storage typing of actual uint8 reads, not a floating numerical law. No
condition is imposed on masked-off reads or their arbitrary undef values. -/
def ActiveBytes (p : Parameters) (s : State Word) : Prop :=
  ∀ i : Fin p.block, inputOffset p s i.val < p.count →
    ∃ b : BitVec 8, s.mem p.input (inputOffset p s i.val) = .mk .nat b.toNat

/-- Independent initial-byte lookup. Every output uses the original memory,
including when input and output region names alias. -/
def initialNibble (p : Parameters) (s : State Word) (j : TileIndex [p.block*2]) : Bits.Nibble :=
  let b : Bits.Packed := BitVec.ofNat 8 ((s.mem p.input (inputOffset p s (j.1.val/2))).read .nat)
  if j.1.val%2 = 0 then Bits.high b else Bits.low b

def expectedValue (p : Parameters) (s : State Word) (j : TileIndex [p.block*2]) : Word :=
  embed ⟨Bits.expectedBits (initialNibble p s j)⟩

def written (p : Parameters) (s : State Word) : State Word :=
  (TileShape.allIndices [p.block*2]).foldl (fun t j =>
    if active p s j then t.write p.output (outputOffset p s j) (.mk .bf16 (expectedValue p s j)) else t) s

theorem active_parent (p : Parameters) (s : State Word) (j : TileIndex [p.block*2])
    (h : active p s j) : inputOffset p s (j.1.val/2) < p.count := by
  dsimp [active, outputOffset] at h
  dsimp [inputOffset]
  rw [← Nat.mul_assoc] at h
  omega

theorem offsets_injective (p : Parameters) (s : State Word) :
    Function.Injective (outputOffset p s) := by
  intro i j h
  have hv : i.1.val = j.1.val := by simpa [outputOffset] using h
  rcases i with ⟨i,⟨⟩⟩; rcases j with ⟨j,⟨⟩⟩
  have : i=j := Fin.ext hv
  cases this; rfl

theorem readback (p : Parameters) (s : State Word) (j : TileIndex [p.block*2])
    (hj : active p s j) :
    (written p s).mem p.output (outputOffset p s j) = .mk .bf16 (expectedValue p s j) :=
  State.masked_scatter_readback _ _ _ _ _ _ hj (offsets_injective p s)

theorem frame (p : Parameters) (s : State Word) (r : RegionName) (o : Nat)
    (h : r ≠ p.output ∨ ∀ j : TileIndex [p.block*2], active p s j → o ≠ outputOffset p s j) :
    (written p s).mem r o = s.mem r o :=
  State.masked_scatter_frame _ _ _ _ _ _ _ (h.imp id (fun h j _ => h j)) _

def loaded (p : Parameters) (s : State Word) (i : Nat) : Nat :=
  if inputOffset p s i < p.count then (s.mem p.input (inputOffset p s i)).read .nat
  else s.undef .nat p.input (inputOffset p s i)
def laneNibble (p : Parameters) (s : State Word) (j : TileIndex [p.block*2]) : Nat :=
  let b := loaded p s (j.1.val/2)
  if j.1.val%2=0 then b >>> 4 else b &&& 15

#specNonCircular written avoiding [source, FP.Structural.exec]
#axiomsClean active_parent
#axiomsClean readback
#axiomsClean frame
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
