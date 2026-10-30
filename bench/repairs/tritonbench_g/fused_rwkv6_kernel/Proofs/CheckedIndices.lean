import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.Domain
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel.Checked
noncomputable section
set_option maxHeartbeats 8000000

/-- Sum of nonnegative bounds for each setup expression and final increment;
it is computed with Python integers before any Triton scalar specialization. -/
def indexLimit (c : Shape) : Nat :=
  c.Z+c.NK+c.NV+c.BK+c.BV+c.K+c.V+c.H+
  c.Z*c.T*c.K+c.Z*c.T*c.V+c.NK*c.Z*c.T*c.V+c.Z*c.K*c.V+
  (c.NK*c.BK)*c.V+c.NV*c.BV+c.H*c.K+c.NK*c.BK+c.T*c.K+c.T*c.V+c.NK*c.Z+32

structure ShapeContract (c : Shape) : Prop where
  sizes : Sizes c
  keyPitch : c.keyPitch=c.T*c.K
  valuePitch : c.valuePitch=c.T*c.V
  bkSource : c.BK=min (2^Nat.clog 2 c.K) 32
  bvSource : c.BV=min (2^Nat.clog 2 c.V) 32
  gridX : c.NV≤2^31-1
  gridY : c.NK≤65535
  gridZ : c.Z≤65535
  limit : indexLimit c<2^31

theorem setup_bounds (c : Shape) (hc : ShapeContract c) (p : Program) (hp : Valid c p)
    (j l : Nat) (hj : j<c.BK) (hl : l<c.BV) (x : Int) (hx : x∈setupExpressions c p j l) :
    -(indexLimit c : Int)≤x ∧ x ≤ indexLimit c := by
  obtain ⟨hz,hi,hv⟩ := hp
  have zle : p.z≤c.Z := by omega
  have ile : p.ik≤c.NK := by omega
  have vle : p.iv≤c.NV := by omega
  have e : p.ik*c.BK+j≤c.NK*c.BK := by
    have h := Nat.mul_le_mul_right c.BK (show p.ik+1≤c.NK by omega)
    nlinarith
  have f : p.iv*c.BV+l≤c.NV*c.BV := by
    have h := Nat.mul_le_mul_right c.BV (show p.iv+1≤c.NV by omega)
    nlinarith
  have qhead : p.z*c.keyPitch≤c.Z*c.T*c.K := by
    rw [hc.keyPitch]
    simpa [Nat.mul_assoc] using Nat.mul_le_mul_right (c.T*c.K) zle
  have vhead : p.z*c.valuePitch≤c.Z*c.T*c.V := by
    rw [hc.valuePitch]
    simpa [Nat.mul_assoc] using Nat.mul_le_mul_right (c.T*c.V) zle
  have head : p.z%c.H≤c.H := Nat.le_of_lt (Nat.mod_lt _ hc.sizes.H)
  have bias : (p.z%c.H)*c.K≤c.H*c.K := Nat.mul_le_mul_right c.K head
  have ikbh : p.ik*c.B*c.H≤c.NK*c.Z := by
    simpa [Shape.Z,Nat.mul_assoc] using Nat.mul_le_mul_right (c.B*c.H) ile
  have ikb : p.ik*c.B≤c.NK*c.Z := by
    have h : p.ik*c.B≤p.ik*c.B*c.H := by simpa using Nat.mul_le_mul_left (p.ik*c.B) hc.sizes.H
    omega
  have out : (p.z+p.ik*c.B*c.H)*(c.T*c.V)≤c.Z*c.T*c.V+c.NK*c.Z*c.T*c.V := by
    have h := Nat.mul_le_mul_right (c.T*c.V) (show p.z+p.ik*c.B*c.H≤c.Z+c.NK*c.Z by omega)
    simpa [Nat.add_mul,Nat.mul_assoc] using h
  have stateHead : p.z*c.K*c.V≤c.Z*c.K*c.V := by
    simpa [Nat.mul_assoc] using Nat.mul_le_mul_right (c.K*c.V) zle
  have stateK : p.z*c.K≤c.Z*c.K*c.V := by
    have h : p.z*c.K≤p.z*c.K*c.V := by simpa using Nat.mul_le_mul_left (p.z*c.K) hc.sizes.V
    omega
  have stateE : (p.ik*c.BK+j)*c.V≤(c.NK*c.BK)*c.V := Nat.mul_le_mul_right c.V e
  have lowK : -(c.K:Int)≤(if c.reverse then ((c.T:Int)-1)*c.K else 0) := by
    split <;> nlinarith only [Int.natCast_nonneg c.T, Int.natCast_nonneg c.K, Int.natCast_nonneg c.V, mul_nonneg (Int.natCast_nonneg c.T) (Int.natCast_nonneg c.K), mul_nonneg (Int.natCast_nonneg c.T) (Int.natCast_nonneg c.V)]
  have highK : (if c.reverse then ((c.T:Int)-1)*c.K else 0)≤(c.T*c.K:Nat) := by
    split <;> push_cast <;> nlinarith only [Int.natCast_nonneg c.T, Int.natCast_nonneg c.K, Int.natCast_nonneg c.V, mul_nonneg (Int.natCast_nonneg c.T) (Int.natCast_nonneg c.K), mul_nonneg (Int.natCast_nonneg c.T) (Int.natCast_nonneg c.V)]
  have lowV : -(c.V:Int)≤(if c.reverse then ((c.T:Int)-1)*c.V else 0) := by
    split <;> nlinarith only [Int.natCast_nonneg c.T, Int.natCast_nonneg c.K, Int.natCast_nonneg c.V, mul_nonneg (Int.natCast_nonneg c.T) (Int.natCast_nonneg c.K), mul_nonneg (Int.natCast_nonneg c.T) (Int.natCast_nonneg c.V)]
  have highV : (if c.reverse then ((c.T:Int)-1)*c.V else 0)≤(c.T*c.V:Nat) := by
    split <;> push_cast <;> nlinarith only [Int.natCast_nonneg c.T, Int.natCast_nonneg c.K, Int.natCast_nonneg c.V, mul_nonneg (Int.natCast_nonneg c.T) (Int.natCast_nonneg c.K), mul_nonneg (Int.natCast_nonneg c.T) (Int.natCast_nonneg c.V)]
  simp only [setupExpressions,← Nat.cast_mul,← Nat.cast_add,List.mem_cons,List.not_mem_nil,or_false] at hx
  unfold indexLimit
  rcases hx with hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx
  all_goals subst x;constructor <;> omega

theorem capacity_bound (c : Shape) (hc : ShapeContract c) (r : Region) : extent c r ≤ indexLimit c := by
  cases r <;> simp only [extent,hc.keyPitch,hc.valuePitch,indexLimit,Nat.mul_assoc] <;> omega

def ShapeContract.domain (c : Shape) (hc : ShapeContract c) : Domain c where
  sizes := hc.sizes
  keyPitch := by rw [hc.keyPitch]
  valuePitch := by rw [hc.valuePitch]
  bkSource := hc.bkSource
  bvSource := hc.bvSource
  gridX := hc.gridX
  gridY := hc.gridY
  gridZ := hc.gridZ
  i32 := by
    intro p hp j hj l hl x hx
    have h := setup_bounds c hc p hp j l hj hl x hx
    have bound := hc.limit
    dsimp [signed32]
    omega
  capacities := by
    intro r
    have h := capacity_bound c hc r
    have bound := hc.limit
    omega

#axiomsClean setup_bounds
#axiomsClean ShapeContract.domain
end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel.Checked
