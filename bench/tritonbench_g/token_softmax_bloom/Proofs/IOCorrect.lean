import bench.tritonbench_g.token_softmax_bloom.Proofs.IOSafety
import bench.tritonbench_g.token_softmax_bloom.Proofs.IOResults
import bench.tritonbench_g.token_softmax_bloom.Proofs.IOScalar

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.IOCorrect
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open SoftmaxIO
noncomputable section

theorem real (p : Parameters) (positive : 0 < p.width)
    (sep : Grid.Separate p) (work : List Grid.PID) :
    (realIO p work).ImplementsOn (realDomain p work)
      (target Real.algebra (fun x : ℝ => some x) p (fun _ => WithBot.unbotD 0) work realRow) := by
  intro vals admitted bounds inside s binding
  exact ⟨IOSafety.safe Real.algebra (fun x : ℝ => some x) p (fun _ => WithBot.unbotD 0)
      positive sep work vals bounds inside s binding,
    IOResults.result Real.algebra (fun x : ℝ => some x) p (fun _ => WithBot.unbotD 0)
      positive sep work realRow vals s binding admitted.1 (IOScalar.real p positive work vals s binding)⟩

/-- The complete native observation is an ordinary supporting theorem. It
retains all FP32 words, exact instruction order, lane ownership and frame. -/
theorem native (I : Native.Instructions) (V : Nat) (p : Parameters) (positive : 0 < p.width)
    (sep : Grid.Separate p) (work : List Grid.PID) :
    (nativeIO I V p work).ImplementsOn (SoftmaxIO.Unique p work)
      (target (SoftmaxTriton1.NativeGraph.algebra I V) SoftmaxTriton1.NativeGraph.atom p
        (fun i value => value (SoftmaxTriton1.NativeGraph.owner V p.width i.val)) work
        (nativeRow I V p.width)) := by
  intro vals admitted bounds inside s binding
  exact ⟨IOSafety.safe (SoftmaxTriton1.NativeGraph.algebra I V) SoftmaxTriton1.NativeGraph.atom p
      (fun i value => value (SoftmaxTriton1.NativeGraph.owner V p.width i.val))
      positive sep work vals bounds inside s binding,
    IOResults.result (SoftmaxTriton1.NativeGraph.algebra I V) SoftmaxTriton1.NativeGraph.atom p
      (fun i value => value (SoftmaxTriton1.NativeGraph.owner V p.width i.val))
      positive sep work (nativeRow I V p.width) vals s binding admitted
      (fun r i _ => IOScalar.native I V p work vals s binding r i)⟩

#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.IOCorrect
