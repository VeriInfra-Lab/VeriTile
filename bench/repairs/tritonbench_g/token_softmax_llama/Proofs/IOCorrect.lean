import bench.repairs.tritonbench_g.token_softmax_llama.IO
import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.IOCorrect

namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.PublicIO
open VeriTile.Triton FP.Structural Strided
open TokenSoftmaxBloom.SoftmaxIO (realDomain Unique target realRow nativeRow)
noncomputable section

theorem real (p : Parameters) (positive : 0<p.width)
    (separate : Grid.Separate p) (work : List Grid.PID) :
    (realIO p work).ImplementsOn (realDomain p work)
      (target Real.algebra (fun x : ℝ => some x) p (fun _ => WithBot.unbotD 0) work realRow) :=
  TokenSoftmaxBloom.IOCorrect.real p positive separate work

/-- Complete native source observation, not another headline equivalence.
All FP32 words, every selected PID/lane, instruction order and frame remain.
Repeated identical launches retain the original Unique event interpretation. -/
theorem native (I : Native.Instructions) (V : Nat) (p : Parameters)
    (positive : 0<p.width) (separate : Grid.Separate p) (work : List Grid.PID) :
    (nativeIO I V p work).ImplementsOn (Unique p work)
      (target (SoftmaxTriton1.NativeGraph.algebra I V) SoftmaxTriton1.NativeGraph.atom p
        (fun i value => value (SoftmaxTriton1.NativeGraph.owner V p.width i.val)) work
        (nativeRow I V p.width)) :=
  TokenSoftmaxBloom.IOCorrect.native I V p positive separate work

#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.PublicIO
