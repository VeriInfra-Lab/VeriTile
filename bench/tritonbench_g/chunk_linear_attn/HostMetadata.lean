import bench.tritonbench_g.chunk_linear_attn.GridReadback
import bench.tritonbench_g.chunk_gla_fwd.Proofs.HostMetadata
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Host
open VeriTile Triton
open VeriTile.Bench.Repairs.ChunkGlaFwd.Host (nextPowerOfTwo nextPowerOfTwo_ge)
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (ceilDiv owner_lt)
set_option maxHeartbeats 12000000

structure Dimensions where
  (B H T K V : Nat)
def Dimensions.BK (c : Dimensions) := min 64 (nextPowerOfTwo c.K)
def Dimensions.BV (c : Dimensions) (backward : Bool) :=
  min (if backward then 32 else 64) (nextPowerOfTwo c.V)
def Dimensions.NT (c : Dimensions) := ceilDiv c.T 64
def Dimensions.parameters (c : Dimensions) (backward : Bool) : Parameters :=
  ⟨c.B,c.H,c.T,c.K,c.V,64,c.BK,c.BV backward,c.NT⟩
def Dimensions.warps (c : Dimensions) := if c.BK=64 then 4 else 2

structure Valid (c : Dimensions) : Prop where
  b_pos : 0<c.B
  h_pos : 0<c.H
  t_pos : 0<c.T
  k_min : 16≤c.K
  v_min : 16≤c.V

theorem cap64_legal (K : Nat) (hk : 16≤K) :
    min 64 (nextPowerOfTwo K)∈([16,32,64] : List Nat) := by
  by_cases h : K≤64
  · interval_cases K <;> decide
  · have hp := nextPowerOfTwo_ge K
    have hc : 64≤nextPowerOfTwo K := by omega
    simp [Nat.min_eq_left hc]
theorem cap32_legal (V : Nat) (hv : 16≤V) :
    min 32 (nextPowerOfTwo V)∈([16,32] : List Nat) := by
  by_cases h : V≤32
  · interval_cases V <;> decide
  · have hp := nextPowerOfTwo_ge V
    have hc : 32≤nextPowerOfTwo V := by omega
    simp [Nat.min_eq_left hc]

theorem bk_legal (c : Dimensions) (h : Valid c) : c.BK=16 ∨ c.BK=32 ∨ c.BK=64 := by
  simpa [Dimensions.BK] using cap64_legal c.K h.k_min
theorem bv_legal (c : Dimensions) (h : Valid c) (backward : Bool) :
    c.BV backward=16 ∨ c.BV backward=32 ∨ c.BV backward=64 := by
  cases backward
  · simpa [Dimensions.BV] using cap64_legal c.V h.v_min
  · have hv := cap32_legal c.V h.v_min
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hv
    rcases hv with hv|hv
    · exact Or.inl hv
    · exact Or.inr (Or.inl hv)

theorem shape_valid (c : Dimensions) (h : Valid c) (backward : Bool) :
    Geometry.ShapeValid (c.parameters backward) := by
  have hk := bk_legal c h
  have hv := bv_legal c h backward
  refine ⟨by change 0<64; decide,?_,?_,rfl⟩
  · change 0<c.BK
    rcases hk with hk|hk|hk <;> omega
  · change 0<c.BV backward
    rcases hv with hv|hv|hv <;> omega

theorem ceil_cover (N W : Nat) (hw : 0<W) : N≤ceilDiv N W*W := by
  by_cases hn : N=0
  · simp [hn]
  · have h := owner_lt hw (show N-1<N from by omega)
    have hd := (Nat.div_lt_iff_lt_mul hw).mp h
    omega

theorem covers (c : Dimensions) (h : Valid c) (backward : Bool) :
    c.T≤(c.parameters backward).NT*(c.parameters backward).BT ∧
    c.K≤keyBlocks (c.parameters backward)*(c.parameters backward).BK ∧
    c.V≤valueBlocks (c.parameters backward)*(c.parameters backward).BV := by
  have hs := shape_valid c h backward
  exact ⟨ceil_cover c.T 64 (by decide),ceil_cover c.K c.BK hs.bk_pos,
    ceil_cover c.V (c.BV backward) hs.bv_pos⟩

def historyShape (c : Dimensions) := [c.B,c.H,c.NT*c.K,c.V]
def stateShape (c : Dimensions) := [c.B,c.H,c.K,c.V]
def queryShape (c : Dimensions) := [c.B,c.H,c.T,c.K]
def valueShape (c : Dimensions) := [c.B,c.H,c.T,c.V]
def partialShape (c : Dimensions) := [keyBlocks (c.parameters Bool.true),c.B,c.H,c.T,c.V]

def inputStrides (c : Dimensions) (values : Bool) : List Nat :=
  let d := if values then c.V else c.K
  [c.H*c.T*d,c.T*d,d,1]

def forwardResult (final : Bool) : RegionName × Option RegionName :=
  ("output",if final then some "final" else none)
def backwardResult (seed : Bool) : List (Option RegionName) :=
  [some "queryGradient",some "keyGradient",some "valueGradient",none,
    if seed then some "initialGradient" else none,none]
def wrapperResult (normalize final : Bool) : RegionName × Option RegionName :=
  forwardResult final

theorem normalize_noop (final : Bool) :
    wrapperResult Bool.true final=wrapperResult Bool.false final := rfl
theorem gradient_slots (seed : Bool) :
    (backwardResult seed).length=6 ∧
    (backwardResult seed)[4]?=some (if seed then some ("initialGradient" : RegionName) else none) ∧
    (backwardResult seed)[3]?=some none ∧ (backwardResult seed)[5]?=some none := by
  simp [backwardResult]

#axiomsClean cap64_legal
#axiomsClean cap32_legal
#axiomsClean shape_valid
#axiomsClean covers
#axiomsClean normalize_noop
#axiomsClean gradient_slots
end VeriTile.Bench.Repairs.ChunkLinearAttn.Host
