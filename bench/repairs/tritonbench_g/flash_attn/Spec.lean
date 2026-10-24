import bench.repairs.tritonbench_g.flash_attn.Proofs.Geometry
import bench.repairs.tritonbench_g.flash_attn.Proofs.OnlineMath
namespace VeriTile.Bench.Repairs.FlashAttn.Spec
open VeriTile.Triton
noncomputable section
/-- The independent target only reads the logical Q/K/V tensors and visible keys.
The retained source decimal selects base-2 scores; this is not an error bound
between native TF32/FP16 arithmetic and exact real attention. -/
def score (x : Layout) (m : Memory ℝ) (scale : ℝ) (b h row k : Nat) : ℝ :=
  ((List.range x.shape.dim).map (fun e=>
    (scale*1.44269504*logical m .q x.q b h row e)*logical m .k x.k b h k e)).sum

def keys (s : Shape) (row : Nat) : List Nat :=
  (List.range s.length).filter (visible s row)
def pairs (x : Layout) (m : Memory ℝ) (scale : ℝ) (b h row col : Nat) : List (ℝ × ℝ) :=
  (keys x.shape row).map (fun k=>(score x m scale b h row k,logical m .v x.v b h k col))
def denominator (x : Layout) (m : Memory ℝ) (scale : ℝ) (b h row : Nat) : ℝ :=
  ((keys x.shape row).map (fun k=>pow2 (score x m scale b h row k))).sum
def numerator (x : Layout) (m : Memory ℝ) (scale : ℝ) (b h row col : Nat) : ℝ :=
  ((keys x.shape row).map (fun k=>pow2 (score x m scale b h row k)*logical m .v x.v b h k col)).sum
def output (x : Layout) (m : Memory ℝ) (scale : ℝ) (b h row col : Nat) : ℝ :=
  numerator x m scale b h row col/denominator x m scale b h row
def lse (x : Layout) (m : Memory ℝ) (scale : ℝ) (b h row : Nat) : ℝ :=
  Real.log (denominator x m scale b h row)/Real.log 2

theorem zero_mem (s : Shape) (row : Nat) (hn : 0<s.length) : 0∈keys s row := by
  simp [keys,visible,hn]
theorem pairs_nonempty (x : Layout) (m : Memory ℝ) (scale : ℝ) (b h row col : Nat)
    (hn : 0<x.shape.length) : pairs x m scale b h row col≠[] := by
  have hm:=List.mem_map_of_mem (f:=fun k=>(score x m scale b h row k,logical m .v x.v b h k col))
    (zero_mem x.shape row hn)
  intro he
  change _∈pairs x m scale b h row col at hm
  simpa [he] using hm

theorem denominator_positive (x : Layout) (m : Memory ℝ) (scale : ℝ) (b h row : Nat)
    (hn : 0<x.shape.length) : 0<denominator x m scale b h row := by
  simpa [OnlineMath.denominator,pairs,denominator,List.map_map] using
    OnlineMath.denominator_positive (pairs x m scale b h row 0)
      (pairs_nonempty x m scale b h row 0 hn)
#axiomsClean denominator_positive
#specNonCircular output avoiding [execute,outputValue,accumulated,blockStep]
#specNonCircular lse avoiding [execute,logValue,accumulated,blockStep]
end
end VeriTile.Bench.Repairs.FlashAttn.Spec
