import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.FPSource
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

abbrev inputOffset (p : Parameters) := Fp4ToBf16.FPValue.inputOffset (base p)
abbrev outputOffset (p : Parameters) := Fp4ToBf16.FPValue.outputOffset (base p)
abbrev active (p : Parameters) := Fp4ToBf16.FPValue.active (base p)
abbrev ActiveBytes (p : Parameters) := Fp4ToBf16.FPValue.ActiveBytes (base p)
abbrev initialNibble (p : Parameters) := Fp4ToBf16.FPValue.initialNibble (base p)
abbrev byteValue (p : Parameters) := Fp4ToBf16.FPValue.expectedValue (base p)
abbrev loaded (p : Parameters) := Fp4ToBf16.FPValue.loaded (base p)
abbrev laneNibble (p : Parameters) := Fp4ToBf16.FPValue.laneNibble (base p)

def scaleOffset (p : Parameters) (s : State Word) (i : Nat) := s.pids 0*p.groups+i
def scaleCount (p : Parameters) := p.count*2/32
/-- The source uses floor, not ceil. A partially filled final scale group has
masked-off scale input even though some of its output lanes are active. -/
def loadedScale (p : Parameters) (s : State Word) (i : Nat) :=
  if scaleOffset p s i < scaleCount p then
    (s.mem p.scales (scaleOffset p s i)).read .nat
  else s.undef .nat p.scales (scaleOffset p s i)
def ActiveScaleBytes (p : Parameters) (s : State Word) : Prop :=
  ∀ i : Fin p.groups, scaleOffset p s i.val < scaleCount p →
    ∃ b : BitVec 8, s.mem p.scales (scaleOffset p s i.val) = .mk .nat b.toNat

def signed16 (z : Int) := (BitVec.ofInt 16 z).toInt
def scaleExponent (n : Nat) : Int :=
  (BitVec.ofInt 32 (signed16 (signed16 n - 127))).toInt

def nanScale : Word := embed (Bits.narrow ⟨0x7fc00000⟩)
def scaleValue (P : Primitives) (n : Nat) : Word :=
  if n ≠ 255 then embed (Bits.narrow (P.pow2 (scaleExponent n))) else nanScale

def expectedValue (P : Primitives) (p : Parameters) (s : State Word)
    (j : TileIndex [p.block*2]) : Word :=
  P.mulBF16 (byteValue p s j) (scaleValue P (loadedScale p s (j.1.val/32)))

def written (P : Primitives) (p : Parameters) (s : State Word) : State Word :=
  (TileShape.allIndices [p.block*2]).foldl (fun t j =>
    if active p s j then t.write p.output (outputOffset p s j) (.mk .bf16 (expectedValue P p s j)) else t) s

theorem readback (P : Primitives) (p : Parameters) (s : State Word)
    (j : TileIndex [p.block*2]) (hj : active p s j) :
    (written P p s).mem p.output (outputOffset p s j) = .mk .bf16 (expectedValue P p s j) :=
  State.masked_scatter_readback _ _ _ _ _ _ hj (Fp4ToBf16.FPValue.offsets_injective (base p) s)

theorem frame (P : Primitives) (p : Parameters) (s : State Word) (r : RegionName) (o : Nat)
    (h : r ≠ p.output ∨ ∀ j : TileIndex [p.block*2], active p s j → o ≠ outputOffset p s j) :
    (written P p s).mem r o = s.mem r o :=
  State.masked_scatter_frame _ _ _ _ _ _ _ (h.imp id (fun h j _ => h j)) _

/-- For complete groups, an active output's scale load is active as well.
For arbitrary counts the source does not provide this property. -/
theorem active_scale_of_grouped (p : Parameters) (s : State Word)
    (hgrouped : p.count % 16 = 0) (j : TileIndex [p.block*2]) (hj : active p s j) :
    scaleOffset p s (j.1.val/32) < scaleCount p := by
  change s.pids 0*(p.groups*16*2)+j.1.val < p.count*2 at hj
  have hid : s.pids 0*(p.groups*16*2) = (s.pids 0*p.groups)*32 := by ring
  rw [hid] at hj
  simp only [scaleOffset, scaleCount]
  omega

#specNonCircular written avoiding [source, FP.Structural.exec]
#axiomsClean active_scale_of_grouped
#axiomsClean readback
#axiomsClean frame
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
