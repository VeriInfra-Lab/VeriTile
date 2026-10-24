import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardSource

/-! Additive constructors over the staged, complete source quotation.
No change to ForwardSource or its original Option String controls is needed.
These four profiles are exactly the staged original-source probes; all flags
and every Option String value remain parameters, including unknown strings.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.Typed
open VeriTile.Triton

inductive CheckedProfile where
  | fp32 | narrow | mixed | wide
  deriving DecidableEq, Repr

def narrowCandidate : Source :=
  forwardSource%[tl.float16, tl.float16, tl.float16, indexCommuted]

def mixedCandidate : Source :=
  forwardSource%[tl.float16, tl.float32, tl.float16, indexCommuted]

def wideCandidate : Source :=
  forwardSource%[tl.float64, tl.float16, tl.float64, indexCommuted]

def checkedKernel (profile : CheckedProfile) (commuted : Bool) : Source :=
  match profile, commuted with
  | .fp32, Bool.false => fp32Original
  | .fp32, Bool.true => fp32Candidate
  | .narrow, Bool.false => narrowOriginal
  | .narrow, Bool.true => narrowCandidate
  | .mixed, Bool.false => mixedOriginal
  | .mixed, Bool.true => mixedCandidate
  | .wide, Bool.false => wideOriginal
  | .wide, Bool.true => wideCandidate

end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.Typed
