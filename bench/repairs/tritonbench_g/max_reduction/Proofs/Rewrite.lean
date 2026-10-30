import bench.repairs.tritonbench_g.max_reduction.Proofs.Source
import VeriTile.Triton.Float.GuardedRewrite

namespace VeriTile.Bench.Repairs.MaxReduction.Rewrite
open VeriTile.Triton FP.Structural
noncomputable section
open scoped VeriTile.Spec
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

private theorem scalar_mul (a b : Nat) :
    bop (fun x y : Nat => x * y) Broadcast.nil
      (fun _ : TileIndex [] => a) (fun _ : TileIndex [] => b) =
      (fun _ : TileIndex [] => a * b) := rfl

private theorem lane_lt (N B : Nat) :
    bop (fun a b : Nat => decide (a < b)) Broadcast.scalarR
      (fun i : TileIndex [B] => i.1.val) (fun _ : TileIndex [] => N) =
      (fun i : TileIndex [B] => decide (i.1.val < N)) := rfl

theorem stage1_same_run [Inhabited A] (e : Source.Engine A) (inp out : RegionName)
    (N B : Nat) (s : State A) :
    FP.Structural.exec e.algebra (stage1 inp out N B) s =
      FP.Structural.exec e.algebra (stage1Commuted inp out N B) s := by
  simp [stage1,stage1Commuted,VeriTile.Bench.TritonBenchG.MaxReduction.stage1,FP.Structural.exec,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,bop,natLt,natLe,numericLt,numericLe,Region.cast,scalar_mul,lane_lt,State.setReg,Source.Engine.algebra,Source.reduceWith,Source.indexedWith,Nat.mul_comm]

theorem stage2_candidate_exec [Inhabited A] (e : Source.Engine A)
    (inp out : RegionName) (N B : Nat) (s : State A) (hb : 0 < B) :
    ∃ final,FP.Structural.exec e.algebra (stage2Complement inp out N B) s=some final ∧
      final.mem=(s.write out 0 (.mk .real
        (Source.scalarValue e (fun i => (s.mem inp i).read .real) N B 0))).mem := by
  simp [stage2Complement,VeriTile.Bench.TritonBenchG.MaxReduction.stage2,FP.Structural.exec,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,store,Option.bind,
    ComputeDType.eraseDType,numeric,bop,natLe,numericLe,Region.cast,Source.Engine.algebra,
    Source.reduceWith,Source.scalarValue,TileShape.allIndices,TileShape.axisDim,hb,
    State.write,State.setReg,Nat.not_le]
  rfl

theorem dimension_same_run [Inhabited A] (e : Source.Engine A) (inp ov oi : RegionName)
    (M N K BM BN : Nat) (s : State A) :
    FP.Structural.exec e.algebra (dimension inp ov oi M N K BM BN) s =
      FP.Structural.exec e.algebra (dimensionCommuted inp ov oi M N K BM BN) s := by
  simp [dimension,dimensionCommuted,VeriTile.Bench.TritonBenchG.MaxReduction.dimensionSource,FP.Structural.exec,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,bop,natLt,natLe,numericLt,numericLe,Region.cast,scalar_mul,lane_lt,State.setReg,Source.Engine.algebra,Source.reduceWith,Source.indexedWith,Nat.mul_comm]

#axiomsClean stage1_same_run
#axiomsClean stage2_candidate_exec
#axiomsClean dimension_same_run
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaxReduction.Rewrite
