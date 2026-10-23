import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeReachability
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeAddresses

namespace VeriTile.Bench.Repairs.ContextAttnFwd.Native
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ContextAttnFwd FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def Header (c : Host.Call) (M : Algebra α) (initial : State α) : Prop :=
  ∀ idx : GridIndex c.grid, ∃ t,
    run (M.withDefaultPrecision .fp32) (FPSuccess.parts c.layout c.regions (Host.scale c)).before
      (FPSuccess.atIndex initial idx) = some t ∧
    evalOp (M.withDefaultPrecision .fp32) none (FPSuccess.parts c.layout c.regions (Host.scale c)).start t = some (fun _ => (0 : Int)) ∧
    evalOp (M.withDefaultPrecision .fp32) none (FPSuccess.parts c.layout c.regions (Host.scale c)).stop t = some (fun _ => (stopAt c idx : Int)) ∧
    evalOp (M.withDefaultPrecision .fp32) none (FPSuccess.parts c.layout c.regions (Host.scale c)).stride t = some (fun _ => c.layout.BN)

/-- Actual source header evaluation supplies the range used by loop_bridge. -/
theorem source_header (c : Host.Call) (M : Algebra α) (s : State α)
    (hm : FPSuccess.Metadata c s) : Header c M s := by
  intro idx
  have hb := Host.batch_bound c idx
  have hstart := hm.1 _ hb
  have hlen := hm.2.1 _ hb
  have hprompt := hm.2.2 _ hb
  simp only [batchAt,Formula.batch,Host.Call.launchAt] at hstart hlen hprompt
  simp (config := {maxSteps := 500000}) [FPSuccess.parts,kernel,Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnFwd.original,ComputeKernel.surfaceBody,
    Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,FPSuccess.atIndex,hstart,hlen,hprompt,
    Formula.head,Formula.queryRow,Host.Call.launchAt,batchAt,Formula.batch,bop,Nat.mul_comm]
  funext i
  simp only [stopAt,stopIndex,seqAt,batchAt,Formula.batch,Host.Call.launchAt,endIndex,
    Nat.add_sub_cancel_right,Nat.cast_add,Nat.cast_mul,Nat.cast_min]
  simp only [bop,Broadcast.leftIndex,Broadcast.rightIndex,←Nat.cast_add,←Nat.cast_mul,Int.ofNat_lt]
  split_ifs <;> simp_all [min_def] <;> omega

/-- Source header, every reached memory/mask site, and direct signed64 loop
control use the same metadata. The statement holds after every actual grid prefix. -/
def GridBridge (c : Host.Call) (abi : ABI c) (M : Algebra α) (initial : State α) : Prop :=
  SourceBridge c M initial ∧ Header c M initial ∧
  ∀ order : List (GridIndex c.grid),∃ t,
    FPSuccess.runGrid c M order initial=some t ∧
      SourceBridge c M t ∧ Header c M t

theorem grid_bridge (c : Host.Call) (abi : ABI c) (M : Algebra α)
    (dots : FPSuccess.DotSupport c.layout M) (s : State α)
    (hm : FPSuccess.Metadata c s) (sep : FPSuccess.Separate c) : GridBridge c abi M s := by
  refine ⟨source_bridge c M dots s hm,source_header c M s hm,?_⟩
  intro order
  obtain ⟨t,ht,hmeta,hframe⟩ := FPSuccess.grid_success_frame c M dots sep order s hm
  exact ⟨t,ht,source_bridge c M dots t hmeta,source_header c M t hmeta⟩

#print axioms grid_bridge
end
end VeriTile.Bench.Repairs.ContextAttnFwd.Native
