import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.CompilerProfile

namespace VeriTile.Bench.Repairs.LayerNormFwd.Compiler
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow
noncomputable section
set_option maxHeartbeats 3000000

def StoreOperand.eval (ops : Ops α) (x : Inputs α) (M row : Nat) (eps p : α) : StoreOperand → Cell α
  | .real32 value => .mk .real ((value.expression x M row eps p).eval ops)
  | .boolGreater left right => .mk .bool (ops.greater
      ((left.expression x M row eps p).eval ops) ((right.expression x M row eps p).eval ops))

/-- The exact returned coordinate written by a source store. `none` means the
source does not issue this store (optional branch or masked column). Scalar
stores remain present when N=0. The residual fallback is not an issued store. -/
def storeCoordinate (f : Flags) (result : TypedSource.Result α M N seeds) (row : Fin M) :
    Destination → Option (Cell α)
  | .y col => if h : col<N then some (result.y row ⟨col,h⟩) else none
  | .y1 col => if h : col<N then result.y1.map (fun value => value row ⟨col,h⟩) else none
  | .residualOut col => if f.residualOut then
      if h : col<N then some (result.residual row ⟨col,h⟩) else none else none
  | .mean => result.mean.map (fun value => value row)
  | .rstd => some (result.rstd row)
  | .mask second col => if h : col<N then
      (if second then result.mask1 else result.mask).map (fun value => value row ⟨col,h⟩) else none

/-- This theorem consumes the actual emitted-store field. Its exact source
mask, destination and register operands match each observed output coordinate;
it is not merely a theorem about a separate expression graph. -/
theorem Artifact.observe_store {α : Type} [Inhabited α] (artifact : Artifact f N block) (ops : Ops α)
    (r : Regions) (stride : Strides) (M : Nat) (eps p : α) (s : State α)
    (row : Fin M) (destination : Destination) :
    (artifact.stores destination).map (StoreOperand.eval ops (NativeSchedule.snapshot r stride s) M row.val eps p)=
      storeCoordinate f (artifact.observe ops r stride M eps p s) row destination := by
  rw [artifact.storeOperands]
  cases destination with
  | y col => by_cases h : col<N <;> simp [sourceStore,storeCoordinate,Artifact.observe,h,StoreOperand.eval]
  | y1 col =>
    cases hw : f.w1 <;> by_cases h : col<N
    all_goals simp [sourceStore,storeCoordinate,Artifact.observe,hw,h,StoreOperand.eval]
  | residualOut col =>
    cases hr : f.residualOut <;> by_cases h : col<N
    all_goals simp [sourceStore,storeCoordinate,Artifact.observe,hr,h,StoreOperand.eval]
  | mean => cases hr : f.rms <;> simp [sourceStore,storeCoordinate,Artifact.observe,hr,StoreOperand.eval]
  | rstd => rfl
  | mask second col =>
    cases second <;> cases hd : f.dropout <;> cases hm : f.dropoutMask <;> cases hx : f.x1 <;> by_cases h : col<N
    all_goals simp [sourceStore,storeCoordinate,Artifact.observe,hd,hm,hx,h,StoreOperand.eval]

#axiomsClean Artifact.observe_store
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Compiler
