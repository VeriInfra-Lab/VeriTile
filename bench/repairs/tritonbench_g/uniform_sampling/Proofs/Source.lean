import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Native

namespace VeriTile.Bench.Repairs.UniformSampling
open VeriTile.Triton VeriTile.Bench.TritonBenchG.UniformSampling
noncomputable section

def sourceOperations (M : Arithmetic α) : Source.Operations α :=
  ⟨M.extract,M.uniform32,M.sub32,M.fma32⟩

theorem source_word (seed offset : Int) (index : Nat) (f : Random.Direct.Field) :
    Source.word seed offset index f=ScalarSpec.word seed offset index f := by
  cases f <;> rfl

/-- Complete source lane registers, including masked-lane arithmetic and the
clamped pointer, are the registers consumed by the reused execution proof. -/
theorem source_lane (M : Arithmetic α) (scalars : Scalars) (seed start n : Nat)
    (x : IndexProof.Location (block n)) :
    let source := Source.lane (sourceOperations M) scalars.lower scalars.upper
      (signed64 seed) (signed64 (start+x.1.done)) (chunkSize n x.1) (block n)
      x.2.1 x.2.2.1.val x.2.2.2.val (field x.2.2.1)
    let native := Native.registers M scalars seed start n x
    source.localIndex=native.localIndex ∧ source.pointerIndex=native.pointerIndex ∧
      source.active=native.active ∧ source.word=native.word ∧
      source.uniform=native.uniform ∧ source.width=native.width ∧ source.value=native.value := by
  simp [Source.lane,Native.registers,sourceOperations,sourceWord,source_word,
    outputIndex,SourceDomain.outputIndex]

#axiomsClean source_word
#axiomsClean source_lane
end
end VeriTile.Bench.Repairs.UniformSampling
