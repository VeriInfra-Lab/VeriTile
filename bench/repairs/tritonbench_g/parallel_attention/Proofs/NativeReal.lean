import bench.repairs.tritonbench_g.parallel_attention.Proofs.NativeGraph
import bench.repairs.tritonbench_g.parallel_attention.Proofs.LoweringReal
import bench.repairs.tritonbench_g.parallel_attention.Proofs.HostReal

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
open scoped BigOperators
set_option maxHeartbeats 800000

theorem nativeQueryLoad_real (c : CompilerLayout) (s : Shape) (x : TensorInputs ℝ)
    (tile : Tile) (L r : Nat) : nativeQueryLoad realOps c s x tile L r=queryLoad realOps s x tile L r := by
  funext e
  simp only [nativeQueryLoad,queryLoad,nativeScale_real,real_mul,real_scale,real_cast]

theorem nativeForwardStep_real (c : CompilerLayout) (s : Shape) (x : TensorInputs ℝ)
    (tile : Tile) (r v : Nat) (diagonal : Bool) (j : Nat) (carry : ℝ×ℝ) :
    nativeForwardStep realOps c s x tile r v diagonal j carry=
      forwardStep realOps c.plans s x tile r v diagonal j carry := by
  simp only [nativeForwardStep,forwardStep,nativeQueryLoad_real,liveDot_real,nativeNorm_real,real_add,real_mul,real_zero,dot_real,total_real]

theorem nativeQueryStep_real (c : CompilerLayout) (s : Shape) (x : TensorInputs ℝ)
    (tile : Tile) (r e : Nat) (diagonal : Bool) (j : Nat) (carry : ℝ) :
    nativeQueryStep realOps c s x tile r e diagonal j carry=
      queryStep realOps c.plans s x tile r e diagonal j carry := by
  simp only [nativeQueryStep,queryStep,nativeQueryLoad_real,liveDot_real,
    nativeCoefficient_real,nativeScaleTwice_real,nativeTwice_real,real_cast,real_mul,real_add,real_two,real_zero,real_scale]
  congr 1
  rw [dot_real,dot_real]
  apply Finset.sum_congr rfl
  intro u hu
  cases diagonal <;> by_cases hv : tile.value=0 <;> by_cases hm : r < j*32+u <;>
    simp [hv,hm,nativeScaleTwice_real,nativeTwice_real,dot_real] <;> try ring <;> simp

theorem nativeKeyValueStep_real (c : CompilerLayout) (s : Shape) (x : TensorInputs ℝ)
    (tile : Tile) (r e v : Nat) (diagonal : Bool) (j : Nat) (carry : ℝ×ℝ) :
    nativeKeyValueStep realOps c s x tile r e v diagonal j carry=
      keyValueStep realOps c.plans s x tile r e v diagonal j carry := by
  simp only [nativeKeyValueStep,keyValueStep,liveDot_real,nativeScale_real,
    nativeCoefficient_real,nativeScaleTwice_real,nativeScaledSum_real,nativeTwice_real,real_cast,real_mul,real_add,real_two,real_zero,real_scale]
  apply Prod.ext <;> dsimp only
  · congr 1
    rw [dot_real,dot_real]
    apply Finset.sum_congr rfl
    intro u hu
    cases diagonal <;> cases hs : s.useScale <;> by_cases hv : tile.value=0 <;>
      by_cases ht : (blocks s.T 32-1-j)*32+u < s.T <;>
      by_cases ht' : tile.chunk*64+j*32+u < s.T <;> by_cases hc : r ≤ j*32+u <;>
      simp [hv,ht,ht',hc,hs,nativeScaleTwice_real,nativeTwice_real,selectedScale,dot_real] <;> try ring <;> simp
  · simp only [dot_real]

/-- Every selected finite scalar layout preserves the independent Real graph;
this proof does not equate different floating-point instruction orders. -/
theorem native_graph_real (c : CompilerLayout) (s : Shape) (x : TensorInputs ℝ)
    (tile : Tile) (r e v : Nat) :
    nativeForwardLane realOps c s x tile r v=forwardLane realOps c.plans s x tile r v ∧
    nativeQueryLane realOps c s x tile r e=queryLane realOps c.plans s x tile r e ∧
    nativeKeyValueLane realOps c s x tile r e v=keyValueLane realOps c.plans s x tile r e v := by
  have hf (d : Bool) : nativeForwardStep realOps c s x tile r v d=forwardStep realOps c.plans s x tile r v d := by
    funext j z
    exact nativeForwardStep_real c s x tile r v d j z
  have hq (d : Bool) : nativeQueryStep realOps c s x tile r e d=queryStep realOps c.plans s x tile r e d := by
    funext j z
    exact nativeQueryStep_real c s x tile r e d j z
  have hk (d : Bool) : nativeKeyValueStep realOps c s x tile r e v d=keyValueStep realOps c.plans s x tile r e v d := by
    funext j z
    exact nativeKeyValueStep_real c s x tile r e v d j z
  constructor
  · simp only [nativeForwardLane,forwardLane,hf]
  constructor
  · simp only [nativeQueryLane,queryLane,hq,nativeScale_real,real_mul,real_scale]
  · simp only [nativeKeyValueLane,keyValueLane,hk]

#axiomsClean nativeForwardStep_real
#axiomsClean nativeQueryStep_real
#axiomsClean nativeKeyValueStep_real
#axiomsClean native_graph_real
end
end VeriTile.Bench.Repairs.ParallelAttention
