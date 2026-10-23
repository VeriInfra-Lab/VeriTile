import bench.repairs.tritonbench_g.log_softmax.Proofs.Kernel
import bench.tritonbench_g.log_softmax.Proofs.FP

namespace VeriTile.Bench.Repairs.LogSoftmax.Rewrite
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LogSoftmax
set_option maxRecDepth 20000

theorem forward_profiled [Inhabited A] (algebra : Algebra A) (O X : RegionName)
    (M N K BM C : Nat) (s : State A) :
    FP.Structural.exec (engine algebra) (Source.forward O X M N K BM C) s =
      FP.Structural.exec (engine algebra) (Source.forwardCommuted O X M N K BM C) s := by
  change run (engine algebra)
    (FPProof.sourcePrefix Bool.false BM ++ (Source.forward O X M N K BM C).surfaceBody.drop 3) s =
      run (engine algebra)
        (FPProof.sourcePrefix Bool.true BM ++ (Source.forward O X M N K BM C).surfaceBody.drop 3) s
  exact FPProof.prefix_run algebra s BM _

theorem backward_profiled [Inhabited A] (algebra : Algebra A) (Y G O : RegionName)
    (M N K BM C : Nat) (s : State A) :
    FP.Structural.exec (engine algebra) (Source.backward Y G O M N K BM C) s =
      FP.Structural.exec (engine algebra) (Source.backwardCommuted Y G O M N K BM C) s := by
  change run (engine algebra)
    (FPProof.sourcePrefix Bool.false BM ++ (Source.backward Y G O M N K BM C).surfaceBody.drop 3) s =
      run (engine algebra)
        (FPProof.sourcePrefix Bool.true BM ++ (Source.backward Y G O M N K BM C).surfaceBody.drop 3) s
  exact FPProof.prefix_run algebra s BM _

theorem engine_default (e : Kernel.Engine A C) : engine e.algebra=e.algebra := rfl

theorem forward_same_run [Inhabited A] (e : Kernel.Engine A C) (O X : RegionName)
    (M N K BM : Nat) (s : State A) :
    FP.Structural.exec e.algebra (Source.forward O X M N K BM C) s =
      FP.Structural.exec e.algebra (Source.forwardCommuted O X M N K BM C) s := by
  simpa only [engine_default] using forward_profiled e.algebra O X M N K BM C s

theorem backward_same_run [Inhabited A] (e : Kernel.Engine A C) (Y G O : RegionName)
    (M N K BM : Nat) (s : State A) :
    FP.Structural.exec e.algebra (Source.backward Y G O M N K BM C) s =
      FP.Structural.exec e.algebra (Source.backwardCommuted Y G O M N K BM C) s := by
  simpa only [engine_default] using backward_profiled e.algebra Y G O M N K BM C s

#axiomsClean forward_same_run
#axiomsClean backward_same_run
end VeriTile.Bench.Repairs.LogSoftmax.Rewrite
