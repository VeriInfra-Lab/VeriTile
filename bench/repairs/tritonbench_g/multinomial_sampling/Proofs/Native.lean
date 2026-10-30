import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Physical

namespace VeriTile.Bench.Repairs.MultinomialSampling.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
open Physical
noncomputable section
set_option maxHeartbeats 3000000

/-- Actual ordinary Tensor descriptors and same-call arange/expand/clone reply.
These contracts concern only library operations, local memory reads, shape,
and allocation separation. The native submission uses scores.device inside
the standard CUDA device context, restoring the ambient caller device afterward.
They do not mention kernel execution or CDF output. -/
structure Preparation {α Store Byte : Type} [Inhabited α] (api : API α Store Byte)
    (p : Parameters) (a : Host.Arguments) (initial : State α) where
  before : Store
  memory : Store
  scores : Tensor
  seeds : Tensor
  offsets : Tensor
  indices : Tensor
  output : Tensor
  caller : Nat → Prop
  /-- Parameters passed by the wrapper's direct `stride` queries. These are
  observation bindings, without size, dtype, contiguity or overflow premises. -/
  scoreStrides : scores.strides = [p.strideScoreBatch,p.strideScoreToken]
  indexStrides : indices.strides = [(Host.params p a).strideIndexBatch,(Host.params p a).strideIndexToken]
  scoreInput : ∀ o j, o < p.batches → j < p.tokens →
    (initial.mem p.scores (o*p.strideScoreBatch+j*p.strideScoreToken)).read .real =
      api.readScore before (scores.base+4*(o*p.strideScoreBatch+j*p.strideScoreToken))
  seedInput : ∀ o, o < p.batches → (initial.mem p.seeds o).read .int = (api.read before (seeds.base+8*o)).toInt
  offsetInput : ∀ o, o < p.batches → (initial.mem p.offsets o).read .int = (api.read before (offsets.base+8*o)).toInt
  labelInput : ∀ o j, o < p.batches → j < p.tokens → a.indexShape.isNone = Bool.false →
    (initial.mem p.indices (o*p.strideIndexBatch+j*p.strideIndexToken)).read .int =
      (api.read before (indices.base+8*(o*p.strideIndexBatch+j*p.strideIndexToken))).toInt
  scorePreserved : ∀ o j, o < p.batches → j < p.tokens →
    api.readScore memory (scores.base+4*(o*p.strideScoreBatch+j*p.strideScoreToken)) =
      api.readScore before (scores.base+4*(o*p.strideScoreBatch+j*p.strideScoreToken))
  seedPreserved : ∀ o, o < p.batches → api.read memory (seeds.base+8*o) = api.read before (seeds.base+8*o)
  offsetPreserved : ∀ o, o < p.batches → api.read memory (offsets.base+8*o) = api.read before (offsets.base+8*o)
  indexPrepared : ∀ o j, o < p.batches → j < p.tokens →
    (api.read memory (indices.base+8*(o*(Host.params p a).strideIndexBatch+j*(Host.params p a).strideIndexToken))).toInt =
      Host.inputLabels p a initial o j
  clone : ∀ o, o < p.batches →
    api.read memory (output.base+8*o) = BitVec.ofInt 64 (Host.inputLabels p a initial o 0)
  labelsRoundtrip : ∀ o j, o < p.batches → j < p.tokens →
    (BitVec.ofInt 64 (Host.inputLabels p a initial o j)).toInt = Host.inputLabels p a initial o j
  outputShape : output.shape = if p.tokens=1 then [p.batches,1] else [p.batches]
  outputAddress : ∀ o, o < p.batches → o*output.strides.headD 1 = o
  outputWidth : output.elementBytes = 8
  outputFits : output.base+8*p.batches ≤ output.allocationEnd
  outputAddressFits : output.allocationEnd ≤ 2^64
  outputDevice : output.device = scores.device
  frame : ∀ b, caller b → api.byte memory b = api.byte before b
  fresh : ∀ b, caller b → b < output.base ∨ output.base+8*p.batches ≤ b
  /-- Complete ordinary Tensor window bounds for every legitimate logical index. -/
  scoreBounds : ∀ o j, o < p.batches → j < p.tokens →
    scores.base+4*(o*p.strideScoreBatch+j*p.strideScoreToken)+4 ≤ scores.allocationEnd ∧ scores.allocationEnd ≤ 2^64
  seedBounds : ∀ o, o < p.batches → seeds.base+8*o+8 ≤ seeds.allocationEnd ∧ seeds.allocationEnd ≤ 2^64
  offsetBounds : ∀ o, o < p.batches → offsets.base+8*o+8 ≤ offsets.allocationEnd ∧ offsets.allocationEnd ≤ 2^64
  indexBounds : ∀ o j, o < p.batches → j < p.tokens →
    indices.base+8*(o*(Host.params p a).strideIndexBatch+j*(Host.params p a).strideIndexToken)+8 ≤ indices.allocationEnd ∧ indices.allocationEnd ≤ 2^64
  /-- Output allocation is disjoint from all live input windows, including
  default arange storage, so actual reads survive every epilogue store. -/
  separateScore : ∀ o j r, o < p.batches → j < p.tokens → r < p.batches →
    output.base+8*r+8 ≤ scores.base+4*(o*p.strideScoreBatch+j*p.strideScoreToken) ∨
      scores.base+4*(o*p.strideScoreBatch+j*p.strideScoreToken)+4 ≤ output.base+8*r
  separateSeed : ∀ o r, o < p.batches → r < p.batches →
    output.base+8*r+8 ≤ seeds.base+8*o ∨ seeds.base+8*o+8 ≤ output.base+8*r
  separateOffset : ∀ o r, o < p.batches → r < p.batches →
    output.base+8*r+8 ≤ offsets.base+8*o ∨ offsets.base+8*o+8 ≤ output.base+8*r
  separateIndex : ∀ o j r, o < p.batches → j < p.tokens → r < p.batches →
    output.base+8*r+8 ≤ indices.base+8*(o*(Host.params p a).strideIndexBatch+j*(Host.params p a).strideIndexToken) ∨
      indices.base+8*(o*(Host.params p a).strideIndexBatch+j*(Host.params p a).strideIndexToken)+8 ≤ output.base+8*r

structure Returned (Store : Type) where
  memory : Store
  tensor : Tensor

def entry {α Store Byte : Type} [Inhabited α] (api : API α Store Byte) (P : Primitives α)
    (p : Parameters) (a : Host.Arguments) (s : State α) (call : Preparation api p a s) : Returned Store :=
  if p.tokens=1 then ⟨call.memory,call.output⟩ else
    ⟨Physical.run api P (Host.params p a) (Host.prepared p a s) call.output call.memory,call.output⟩

theorem returned_descriptor {α Store Byte : Type} [Inhabited α] (api : API α Store Byte) (P : Primitives α)
    (p : Parameters) (a : Host.Arguments) (s : State α) (call : Preparation api p a s) :
    (entry api P p a s call).tensor = call.output ∧
    (entry api P p a s call).tensor.shape = (if p.tokens=1 then [p.batches,1] else [p.batches]) := by
  constructor
  · unfold entry; split <;> rfl
  · have ht : (entry api P p a s call).tensor = call.output := by unfold entry; split <;> rfl
    rw [ht]
    exact call.outputShape

/-- Source input reads are tied to this call's actual prelaunch memory,
including the internally expanded default labels. -/
theorem source_load_binding {α Store Byte : Type} [Inhabited α] (api : API α Store Byte)
    (p : Parameters) (a : Host.Arguments) (s : State α) (call : Preparation api p a s)
    (fresh : Host.Fresh p a) (o j : Nat) (ho : o < p.batches) (hj : j < p.tokens) :
    ((Host.prepared p a s).mem p.scores (o*p.strideScoreBatch+j*p.strideScoreToken)).read .real =
      api.readScore call.memory (call.scores.base+4*(o*p.strideScoreBatch+j*p.strideScoreToken)) ∧
    ((Host.prepared p a s).mem p.seeds o).read .int = (api.read call.memory (call.seeds.base+8*o)).toInt ∧
    ((Host.prepared p a s).mem p.offsets o).read .int = (api.read call.memory (call.offsets.base+8*o)).toInt ∧
    ((Host.prepared p a s).mem p.indices (o*(Host.params p a).strideIndexBatch+j*(Host.params p a).strideIndexToken)).read .int =
      (api.read call.memory (call.indices.base+8*(o*(Host.params p a).strideIndexBatch+j*(Host.params p a).strideIndexToken))).toInt := by
  refine ⟨?_,?_,?_,?_⟩
  · rw [Host.prepared_input p a s p.scores fresh.1.1 (fun h => (fresh.2 h).1),
      call.scorePreserved o j ho hj,call.scoreInput o j ho hj]
  · rw [Host.prepared_input p a s p.seeds fresh.1.2.1 (fun h => (fresh.2 h).2.1),
      call.seedPreserved o ho,call.seedInput o ho]
  · rw [Host.prepared_input p a s p.offsets fresh.1.2.2.1 (fun h => (fresh.2 h).2.2),
      call.offsetPreserved o ho,call.offsetInput o ho]
  · rw [Host.prepared_labels p a s fresh o j ho hj,call.indexPrepared o j ho hj]

/-- Any prefix of epilogues preserves every actual live input load. Combined
with source_load_binding this binds later programs to the same source inputs;
it is proved from allocation separation and primitive memory laws. -/
theorem prefix_inputs {α Store Byte : Type} [Inhabited α] (api : API α Store Byte) (P : Primitives α)
    (p : Parameters) (a : Host.Arguments) (s : State α) (call : Preparation api p a s)
    (xs : List (Physical.Location (Host.params p a))) (o j : Nat) (ho : o<p.batches) (hj : j<p.tokens) :
    let final := Physical.runWrites api (Physical.events P (Host.params p a) (Host.prepared p a s) call.output xs) call.memory
    api.readScore final (call.scores.base+4*(o*p.strideScoreBatch+j*p.strideScoreToken)) =
      api.readScore call.memory (call.scores.base+4*(o*p.strideScoreBatch+j*p.strideScoreToken)) ∧
    api.read final (call.seeds.base+8*o) = api.read call.memory (call.seeds.base+8*o) ∧
    api.read final (call.offsets.base+8*o) = api.read call.memory (call.offsets.base+8*o) ∧
    api.read final (call.indices.base+8*(o*(Host.params p a).strideIndexBatch+j*(Host.params p a).strideIndexToken)) =
      api.read call.memory (call.indices.base+8*(o*(Host.params p a).strideIndexBatch+j*(Host.params p a).strideIndexToken)) := by
  dsimp only
  refine ⟨?_,?_,?_,?_⟩
  · apply Physical.writes_score_frame
    intro e he
    obtain ⟨x,hx,ha,rfl⟩ := (Physical.mem_events P (Host.params p a) (Host.prepared p a s) call.output xs e).mp he
    exact call.separateScore o j (Physical.global (Host.params p a) x) ho hj ha
  · apply Physical.writes_read_frame
    intro e he
    obtain ⟨x,hx,ha,rfl⟩ := (Physical.mem_events P (Host.params p a) (Host.prepared p a s) call.output xs e).mp he
    exact call.separateSeed o (Physical.global (Host.params p a) x) ho ha
  · apply Physical.writes_read_frame
    intro e he
    obtain ⟨x,hx,ha,rfl⟩ := (Physical.mem_events P (Host.params p a) (Host.prepared p a s) call.output xs e).mp he
    exact call.separateOffset o (Physical.global (Host.params p a) x) ho ha
  · apply Physical.writes_read_frame
    intro e he
    obtain ⟨x,hx,ha,rfl⟩ := (Physical.mem_events P (Host.params p a) (Host.prepared p a s) call.output xs e).mp he
    exact call.separateIndex o j (Physical.global (Host.params p a) x) ho hj ha

theorem caller_frame {α Store Byte : Type} [Inhabited α] (api : API α Store Byte) (P : Primitives α)
    (p : Parameters) (a : Host.Arguments) (s : State α) (call : Preparation api p a s)
    (b : Nat) (hb : call.caller b) :
    api.byte (entry api P p a s call).memory b = api.byte call.before b := by
  unfold entry
  split
  · exact call.frame b hb
  · change api.byte (Physical.run api P (Host.params p a) (Host.prepared p a s) call.output call.memory) b = _
    rw [Physical.byte_frame api P (Host.params p a) (Host.prepared p a s) call.output call.memory b (call.fresh b hb)]
    exact call.frame b hb

theorem whole_native (api : API ℝ Store Byte) (p : Parameters) (a : Host.Arguments)
    (s : State ℝ) (call : Preparation api p a s) (fresh : Host.Fresh p a)
    (domain : ∀ o, o < p.batches → Spec.Domain p.tokens (Host.inputWeights p s o)) :
    (entry api Real.primitives p a s call).tensor = call.output ∧
    (entry api Real.primitives p a s call).tensor.shape = (if p.tokens=1 then [p.batches,1] else [p.batches]) ∧
    (∀ o, o < p.batches → Host.Result p a s o
      ((Physical.logicalRead api (entry api Real.primitives p a s call).memory
        (entry api Real.primitives p a s call).tensor o).toInt)) ∧
    (∀ b, call.caller b → api.byte (entry api Real.primitives p a s call).memory b = api.byte call.before b) := by
  refine ⟨(returned_descriptor api Real.primitives p a s call).1,
    (returned_descriptor api Real.primitives p a s call).2,?_,caller_frame api Real.primitives p a s call⟩
  intro o ho
  rw [(returned_descriptor api Real.primitives p a s call).1]
  unfold Physical.logicalRead
  rw [call.outputAddress o ho]
  by_cases ht : p.tokens=1
  · simp only [entry,ht,↓reduceIte]
    rw [call.clone o ho,call.labelsRoundtrip o 0 ho (by omega)]
    unfold Host.Result
    rw [ht]
    apply Spec.singleton
    simpa only [ht] using domain o ho
  · simp only [entry,ht,↓reduceIte]
    rw [Physical.readback api Real.primitives (Host.params p a) (Host.prepared p a s) call.output call.memory o ho]
    obtain ⟨j,hj,hv⟩ := Host.value_result p a s fresh o ho (domain o ho)
    unfold Host.Result Spec.Result
    refine ⟨j,hj,?_⟩
    rw [hv,call.labelsRoundtrip o j ho hj.1]

#axiomsClean returned_descriptor
#axiomsClean source_load_binding
#axiomsClean prefix_inputs
#axiomsClean caller_frame
#axiomsClean whole_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.Native
