import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.HistoricalFPEquiv
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel
noncomputable section
set_option maxHeartbeats 8000000

/-- Integer spelling of the original unsafe output offset. -/
def originalPartialAddress (c : Shape) (z i t f : Nat) :=
  (z+i*c.B*c.H)*c.valuePitch+t*c.V+f

/-- Source spelling after the single repaired pitch expression. -/
def repairedPartialAddress (c : Shape) (z i t f : Nat) :=
  (z+i*c.B*c.H)*(c.T*c.V)+t*c.V+f

theorem repaired_address (c : Shape) (z i t f : Nat) :
    repairedPartialAddress c z i t f=partialAddress c z i t f := by
  unfold repairedPartialAddress partialAddress Shape.Z
  ring

def witnessShape : Shape :=
  ⟨1,2,2,16,16,16,16,32,64,Bool.false,Bool.false,Bool.false⟩

/-- A normal V head slice causes an out-of-allocation write in the original,
while the repair addresses the previously omitted second head. -/
theorem original_counterexample :
    extent witnessShape .o=64 ∧ originalPartialAddress witnessShape 1 0 0 0=64 ∧
    partialAddress witnessShape 1 0 0 0=32 ∧
    inputValue witnessShape 1 1 15<extent witnessShape .v := by
  norm_num [extent,witnessShape,originalPartialAddress,partialAddress,inputValue,Shape.NK,Shape.Z]

/-- This concrete ordinary head-view witness inhabits every supported-domain
condition; the theorem is not vacuous because of hidden source constraints. -/
theorem witness_domain : Domain witnessShape := by
  constructor
  · constructor <;> decide
  · decide
  · decide
  · decide
  · decide
  · decide
  · decide
  · decide
  · intro p hp j hj l hl x hx
    rcases p with ⟨z,i,v⟩
    simp only [Valid,witnessShape,Shape.Z,Shape.NK,Shape.NV] at hp
    norm_num at hp
    obtain ⟨hz,hi,hv⟩ := hp
    have hi0 : i=0 := by omega
    have hv0 : v=0 := by omega
    subst i
    subst v
    simp only [witnessShape] at hj hl
    simp [setupExpressions,witnessShape] at hx
    rcases hx with hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx|hx
    all_goals simp only [signed32]; omega
  · intro r
    cases r <;> norm_num [extent,witnessShape,Shape.Z,Shape.NK]

/-- Initial/current register model reads the same immutable U values as the
source's hoisted bias load, rather than assuming output values. -/
theorem hoisted_bias (a : Ops α) (c : Shape) (plan : Plans c) (p : Program) (scale : α)
    (m : Memory α) (n j : Nat) :
    bias a c p (runLoop a c plan p scale m n).memory j=bias a c p m j := by
  have h := loop_inputs a c plan p scale m n .u (by simp [InputRegion])
  simp [bias,load,h]

#axiomsClean repaired_address
#axiomsClean original_counterexample
#axiomsClean witness_domain
#axiomsClean hoisted_bias
end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel
