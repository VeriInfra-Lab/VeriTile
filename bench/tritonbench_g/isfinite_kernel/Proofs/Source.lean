import bench.tritonbench_g.isfinite_kernel.Kernels
import VeriTile.Triton.Float.Structural
import bench.tritonbench_g.isfinite_kernel.Proofs.Mathematical_MemoryProjection

namespace VeriTile.Bench.Repairs.IsfiniteKernel.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

/-- Explicit same-precision casts are bit preserving; the only reached
floating primitive is the selected scalar classifier. -/
def algebra {A : Type} [Inhabited A] (precision : Precision) (classify : A → Bool) : Algebra A where
  literal := fun _ _ _ => default
  negInf := default
  binary := fun _ _ _ _ _ => default
  unary := fun _ _ _ => default
  cast := fun _ _ _ x => x
  fromNat := fun _ _ => default
  fromInt := fun _ _ => default
  fp32Bits := fun _ => default
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => default
  reduceSum := fun _ {_} _ _ _ => fun _ => default
  isFinite := fun p => if p=precision.finite then some classify else none

def stores {A : Type} [Inhabited A] (classify : A → Bool)
    (input output : View) (n B pid : Nat) (snapshot current : State A) : State A :=
  (TileShape.allIndices [B]).foldl (fun s i =>
    if pid*B+i.1.val<n then s.write "y" (output.address (pid*B+i.1.val))
      (.mk .bool (classify ((snapshot.mem "x" (input.address (pid*B+i.1.val))).read .real))) else s) current

/-- Whole direct JIT execution, retaining every active signed byte address,
the originally loaded register values, and arbitrary masked-load padding. -/
theorem executes {A : Type} [Inhabited A] (precision : Precision) (classify : A → Bool)
    (input output : View) (n B : Nat) (s : State A) :
    ∃ final,FP.Structural.exec (algebra precision classify) (source precision input output n B) s=some final ∧
      final.mem=(stores classify input output n B (s.pids 0) s s).mem := by
  cases precision
  all_goals
    simp +contextual [source,source32,source64,View.address,View.signedAddress,
      FP.Structural.exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      algebra,Precision.finite,ComputeDType.eraseDType,numeric,natLt,bop,FP.Structural.store,Region.cast,
      ofFloat,toFloat,stores,Option.bind,
      VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits.fold_writes_mem]

/-- The candidate changes only exact integer index multiplication. No floating
rewrite, finiteness premise, or inactive-lane condition is used. -/
theorem commuted_executes {A : Type} [Inhabited A] (precision : Precision) (classify : A → Bool)
    (input output : View) (n B : Nat) (s : State A) :
    ∃ final,FP.Structural.exec (algebra precision classify) (indexCommuted precision input output n B) s=some final ∧
      final.mem=(stores classify input output n B (s.pids 0) s s).mem := by
  cases precision
  all_goals
    simp +contextual [indexCommuted,candidate32,candidate64,Nat.mul_comm,View.address,View.signedAddress,
      FP.Structural.exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      algebra,Precision.finite,ComputeDType.eraseDType,numeric,natLt,bop,FP.Structural.store,Region.cast,
      ofFloat,toFloat,stores,Option.bind,
      VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits.fold_writes_mem]

#axiomsClean executes
#axiomsClean commuted_executes
end
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Repairs.IsfiniteKernel.Source
