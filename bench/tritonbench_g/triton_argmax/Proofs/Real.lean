import bench.tritonbench_g.triton_argmax.Proofs.Native
import bench.tritonbench_g.triton_argmax.Proofs.RealDim

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.RealProjection
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction Memory
noncomputable section

/-- Real interpretation of the same NaN comparison: all Real values and the
padding bottom compare equal to themselves. -/
def flag (x : Math.Value) : Int := if x≠x then 1 else 0

def clean (x : Math.Value) : Math.Value := if x≠x then ⊥ else x

def select {n : Nat} (sites : Native.Sites n) (input : Fin n → Math.Value) : Fin n :=
  let flags := fun i => flag (input i)
  let nanIndex := sites.flags.pick orderedGt orderedEq flags
  let present := flags (sites.present.pick orderedGt orderedEq flags)
  if 0<present then nanIndex else sites.numeric.pick orderedGt orderedEq (fun i => clean (input i))

theorem select_numeric {n : Nat} (sites : Native.Sites n) (input : Fin n → Math.Value) :
    select sites input=sites.numeric.pick orderedGt orderedEq input := by
  simp [select,flag,clean]

def update (next current : Math.Value) : Bool :=
  decide (current=current) && (decide (next≠next) || orderedGt next current)

theorem update_ordered (next current : Math.Value) : update next current=orderedGt next current := by
  simp [update]

def localPair (n cur width : Nat) (input : Nat → ℝ) (sites : Native.Sites width) : Math.Value × Nat :=
  let lane := select sites (Math.padded n cur width input)
  (Math.padded n cur width input lane,cur+lane.val)

def step (n cur width : Nat) (input : Nat → ℝ) (sites : Native.Sites width)
    (current : Math.Value × Nat) : Math.Value × Nat :=
  let next := localPair n cur width input sites
  if update next.1 current.1 then next else current

theorem step_original (n cur width : Nat) (input : Nat → ℝ) (sites : Native.Sites width)
    (current : Math.Value × Nat) :
    step n cur width input sites current=Math.strictStep n cur width input sites.numeric current := by
  simp only [step,localPair,select_numeric,update_ordered,orderedGt,decide_eq_true_eq]
  rfl

def loop (n width : Nat) (positive : 0<width) (input : Nat → ℝ)
    (sites : Nat → Native.Sites width) (cur : Nat) (current : Math.Value × Nat) : Math.Value × Nat :=
  if h : cur<n then loop n width positive input sites (cur+width)
    (step n cur width input (sites cur) current) else current
termination_by n-cur
decreasing_by omega

theorem loop_original (n width : Nat) (positive : 0<width) (input : Nat → ℝ)
    (sites : Nat → Native.Sites width) (cur : Nat) (current : Math.Value × Nat) :
    loop n width positive input sites cur current=
      Math.dimensionLoop n width positive input (fun i => (sites i).numeric) cur current := by
  by_cases active : cur<n
  · rw [loop,dif_pos active,Math.dimensionLoop,dif_pos active,step_original,loop_original]
  · rw [loop,dif_neg active,Math.dimensionLoop,dif_neg active]
termination_by n-cur
decreasing_by omega

theorem dimension_reference (n width : Nat) (nonempty : 0<n) (positive : 0<width)
    (input : Nat → ℝ) (sites : Nat → Native.Sites width) :
    (loop n width positive input sites 0 (⊥,0)).2=
      (Math.argmax n nonempty (fun i : Fin n => input i.val)).val := by
  rw [loop_original]
  obtain ⟨bound,correct⟩ := Math.dimension_mathematical n width nonempty positive input
    (fun i => (sites i).numeric) (fun i _ => (sites i).numericComplete)
  have same := least_maximum_unique _ _ _ correct (Math.argmax_spec n nonempty (fun i => input i.val))
  exact congrArg Fin.val same

/-- Global Real replay retains both stages, scratch indices, gather and stores.
The explicit precision-adapter erasure covers the entire integer clamp/cast
macro and is not a native identity for arbitrary requested scratch dtypes. -/
def global (adapter : RealScratch.Adapter) (c : GlobalConfig) (p : GlobalMemory.Buffers)
    (input : Nat → ℝ) (state : RealScratch.State) : RealScratch.State :=
  RealScratch.run adapter c p input state

theorem global_reference (adapter : RealScratch.Adapter) (c : GlobalConfig)
    (coverage : Math.GlobalCoverage c) (p : GlobalMemory.Buffers)
    (input : Nat → ℝ) (state : RealScratch.State) :
    (global adapter c p input state).indices (address p.output 8 0)=
      (Math.argmax c.size coverage.sizePositive (fun i : Fin c.size => input i.val)).val := by
  rw [global,RealScratch.gathered_index adapter c coverage]
  exact congrArg Fin.val (least_maximum_unique _ _ _
    (Math.global_mathematical c coverage input)
    (Math.argmax_spec c.size coverage.sizePositive (fun i => input i.val)))

#axiomsClean select_numeric
#axiomsClean step_original
#axiomsClean dimension_reference
#axiomsClean global_reference
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.RealProjection
