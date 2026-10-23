import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.IO
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Geometry
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.IOProof
open VeriTile Triton FP.Structural
noncomputable section
set_option linter.unusedSimpArgs false

def forwardCapacity (d : Host.Dimensions) (r : RegionName) : Nat :=
  if r="query" ∨ r="key" then d.B*d.H*d.N*d.D
  else if r="value" ∨ r="output" then d.B*d.H*d.N*d.E else 0

def backwardCapacity (d : Host.Dimensions) (r : RegionName) : Nat :=
  if r="query" ∨ r="key" ∨ r="queryGradient" ∨ r="keyGradient" then d.B*d.H*d.N*d.D
  else if r="value" ∨ r="incoming" ∨ r="valueGradient" then d.B*d.H*d.N*d.E else 0

theorem dense_lt {heads time width : Nat} (j : TileIndex [heads,time,width]) :
    denseIndex j < heads*time*width := by
  rcases j with ⟨h,t,a,⟨⟩⟩
  simpa [Geometry.dense,denseIndex,Nat.add_mul,Nat.mul_assoc] using
    (Geometry.dense_bound h.isLt t.isLt a.isLt)

/-- The finite IO bounds are witnessed by exactly the original Tensor extents. -/
theorem forward_in_bounds (d : Host.Dimensions) (x : (forwardIO d).Inputs) :
    (forwardIO d).InBounds (forwardCapacity d) x := by
  refine ⟨?_,?_,?_,?_⟩
  · intro r outside
    simp only [forwardIO,List.mem_cons,List.not_mem_nil,or_false,not_or] at outside
    simp [forwardCapacity,outside.1,outside.2.1,outside.2.2.1,outside.2.2.2]
  · intro ⟨i,hi⟩ j active
    have hlt : i < 4 := hi
    interval_cases i
    all_goals first
      | exact False.elim (by simpa [forwardIO] using active)
      | simpa [forwardIO,inputIndex,inputChannels,forwardCapacity] using dense_lt j
  · intro ⟨o,ho⟩ j active
    have eq : o=0 := by change o<1 at ho; omega
    subst o
    simpa [forwardIO,forwardCapacity] using dense_lt j
  · intro k; exact Fin.elim0 k

theorem backward_in_bounds (d : Host.Dimensions) (x : (backwardIO d).Inputs) :
    (backwardIO d).InBounds (backwardCapacity d) x := by
  refine ⟨?_,?_,?_,?_⟩
  · intro r outside
    have hn : r ≠ "query" ∧ r ≠ "key" ∧ r ≠ "value" ∧ r ≠ "incoming" ∧
        r ≠ "queryGradient" ∧ r ≠ "keyGradient" ∧ r ≠ "valueGradient" := by
      simpa [backwardIO] using outside
    simp [backwardCapacity,hn.1,hn.2.1,hn.2.2.1,hn.2.2.2.1,
      hn.2.2.2.2.1,hn.2.2.2.2.2.1,hn.2.2.2.2.2.2]
  · intro ⟨i,hi⟩ j active
    have hlt : i < 4 := hi
    interval_cases i
    all_goals simpa [backwardIO,inputIndex,inputChannels,backwardCapacity] using dense_lt j
  · intro ⟨o,ho⟩ j active
    have hlt : o < 3 := ho
    interval_cases o
    all_goals simpa [backwardIO,backwardIndex,backwardChannels,backwardCapacity] using dense_lt j
  · intro k; exact Fin.elim0 k
#axiomsClean forward_in_bounds
#axiomsClean backward_in_bounds
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.IOProof
