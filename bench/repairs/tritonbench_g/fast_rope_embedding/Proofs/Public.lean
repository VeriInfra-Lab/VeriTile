import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.Sequence

namespace VeriTile.Bench.Repairs.FastRopeEmbedding
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs
noncomputable section
set_option maxHeartbeats 5000000

theorem second_inputs {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte) (embed : β→α)
    (project : α→β) (op : RopeEmbedding.Operations α) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api embed rq rk before) (laws : StorageLaws api embed rq rk before p)
    (accepted : inputCheck api embed rq rk before p=Bool.true)
    (b h t c : Nat) (hb : b<rk.batches) (hh : h<rk.heads) (ht : t<rk.sequence) (hc : c<rk.dimension) :
    RopeEmbedding.Native.logicalRead api (secondStage api embed project op rq rk before p).call.before
      (secondStage api embed project op rq rk before p).call.input b h t c=
      RopeEmbedding.Native.logicalRead api before p.inputK b h t c := by
  rw [(secondStage api embed project op rq rk before p).entered]
  change RopeEmbedding.Native.logicalRead api (firstResult api embed project op rq rk before p).memory
    (p.second _).call.input b h t c=_
  rw [p.secondInput]
  exact first_keeps_k api embed project op rq rk before p laws accepted b h t c hb hh ht hc

theorem second_coefficients {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte) (embed : β→α)
    (project : α→β) (op : RopeEmbedding.Operations α) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api embed rq rk before) (t j : Nat) (ht : t<rk.sequence) (hj : j<rk.half) :
    api.read (secondStage api embed project op rq rk before p).call.before
      (RopeEmbedding.Checked.tableAddress rk (secondStage api embed project op rq rk before p).call.cosine t j)=
      api.read before (RopeEmbedding.Checked.tableAddress rk p.first.call.cosine t j) ∧
    api.read (secondStage api embed project op rq rk before p).call.before
      (RopeEmbedding.Checked.tableAddress rk (secondStage api embed project op rq rk before p).call.sine t j)=
      api.read before (RopeEmbedding.Checked.tableAddress rk p.first.call.sine t j) := by
  rw [(secondStage api embed project op rq rk before p).entered]
  have eq := p.secondCoefficients (firstResult api embed project op rq rk before p).memory
  simpa only [secondStage,RopeEmbedding.Checked.tableAddress,eq.1,eq.2] using
    first_keeps_coefficients api embed project op rq rk before p t j ht hj

theorem real_target_transport (api : RopeEmbedding.Memory.API ℝ Store Byte) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api id rq rk before) (laws : StorageLaws api id rq rk before p)
    (accepted : inputCheck api id rq rk before p=Bool.true)
    (b h t c : Nat) (hb : b<rk.batches) (hh : h<rk.heads) (ht : t<rk.sequence) (hc : c<rk.dimension) :
    let k := secondStage api id id RopeEmbedding.ideal rq rk before p
    RopeEmbedding.Checked.mathematicalTarget api rk Bool.false k.call.before k.call.input k.call.cosine k.call.sine b h t c=
      RopeEmbedding.Checked.mathematicalTarget api rk Bool.false before p.inputK p.first.call.cosine p.first.call.sine b h t c := by
  dsimp only
  unfold RopeEmbedding.Checked.mathematicalTarget
  split
  · rename_i active
    let j := if c<rk.half then c else c-rk.half
    have hj : j<rk.half := by dsimp [j]; split <;> omega
    have hd : 2*rk.half≤rk.dimension := by unfold RopeEmbedding.Request.half; omega
    dsimp only
    rw [second_inputs api id id RopeEmbedding.ideal rq rk before p laws accepted b h t j hb hh ht (by omega),
      second_inputs api id id RopeEmbedding.ideal rq rk before p laws accepted b h t (j+rk.half) hb hh ht (by omega),
      (second_coefficients api id id RopeEmbedding.ideal rq rk before p t j ht hj).1,
      (second_coefficients api id id RopeEmbedding.ideal rq rk before p t j ht hj).2]
  · exact second_inputs api id id RopeEmbedding.ideal rq rk before p laws accepted b h t c hb hh ht hc

theorem word_target_transport (api : RopeEmbedding.Memory.API RopeEmbedding.Numerics.Word Store (BitVec 8)) (n : RopeEmbedding.Numerics.Primitives)
    (rq rk : RopeEmbedding.Request) (before : Store) (p : PairPreparation api RopeEmbedding.Numerics.leaf rq rk before)
    (laws : StorageLaws api RopeEmbedding.Numerics.leaf rq rk before p) (accepted : inputCheck api RopeEmbedding.Numerics.leaf rq rk before p=Bool.true)
    (b h t c : Nat) (hb : b<rk.batches) (hh : h<rk.heads) (ht : t<rk.sequence) (hc : c<rk.dimension) :
    let k := secondStage api RopeEmbedding.Numerics.leaf RopeEmbedding.Numerics.Tagged.word (RopeEmbedding.Numerics.operations n) rq rk before p
    RopeEmbedding.Checked.machineTarget api n rk Bool.false k.call.before k.call.input k.call.cosine k.call.sine b h t c=
      RopeEmbedding.Checked.machineTarget api n rk Bool.false before p.inputK p.first.call.cosine p.first.call.sine b h t c := by
  dsimp only
  unfold RopeEmbedding.Checked.machineTarget
  split
  · rename_i active
    let j := if c<rk.half then c else c-rk.half
    have hj : j<rk.half := by dsimp [j]; split <;> omega
    have hd : 2*rk.half≤rk.dimension := by unfold RopeEmbedding.Request.half; omega
    dsimp only
    rw [second_inputs api RopeEmbedding.Numerics.leaf RopeEmbedding.Numerics.Tagged.word (RopeEmbedding.Numerics.operations n) rq rk before p laws accepted b h t j hb hh ht (by omega),
      second_inputs api RopeEmbedding.Numerics.leaf RopeEmbedding.Numerics.Tagged.word (RopeEmbedding.Numerics.operations n) rq rk before p laws accepted b h t (j+rk.half) hb hh ht (by omega),
      (second_coefficients api RopeEmbedding.Numerics.leaf RopeEmbedding.Numerics.Tagged.word (RopeEmbedding.Numerics.operations n) rq rk before p t j ht hj).1,
      (second_coefficients api RopeEmbedding.Numerics.leaf RopeEmbedding.Numerics.Tagged.word (RopeEmbedding.Numerics.operations n) rq rk before p t j ht hj).2]
  · exact second_inputs api RopeEmbedding.Numerics.leaf RopeEmbedding.Numerics.Tagged.word (RopeEmbedding.Numerics.operations n) rq rk before p laws accepted b h t c hb hh ht hc

structure PairOutput {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte) (embed : β→α)
    (project : α→β) (op : RopeEmbedding.Operations α) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api embed rq rk before) (qValue kValue : Nat→Nat→Nat→Nat→β)
    (out : Result Store) : Prop where
  q : out.q=p.first.call.output
  k : out.k=(secondStage api embed project op rq rk before p).call.output
  qShape : out.q.shape=[rq.batches,rq.heads,rq.sequence,rq.dimension]
  kShape : out.k.shape=[rk.batches,rk.heads,rk.sequence,rk.dimension]
  qCache : out.qCache=cache rq p.first.call.cosine p.first.call.sine
  kCache : out.kCache=cache rk (secondStage api embed project op rq rk before p).call.cosine
    (secondStage api embed project op rq rk before p).call.sine
  qWords : ∀ b h t c,b<rq.batches → h<rq.heads → t<rq.sequence → c<rq.dimension →
    RopeEmbedding.Native.logicalRead api out.memory out.q b h t c=qValue b h t c
  kWords : ∀ b h t c,b<rk.batches → h<rk.heads → t<rk.sequence → c<rk.dimension →
    RopeEmbedding.Native.logicalRead api out.memory out.k b h t c=kValue b h t c
  qBytes : ∀ b h t c,b<rq.batches → h<rq.heads → t<rq.sequence → c<rq.dimension → ∀ byte : Fin 4,
    api.byte out.memory (RopeEmbedding.Native.logicalAddress out.q b h t c+byte.val)=api.encode (qValue b h t c) byte
  kBytes : ∀ b h t c,b<rk.batches → h<rk.heads → t<rk.sequence → c<rk.dimension → ∀ byte : Fin 4,
    api.byte out.memory (RopeEmbedding.Native.logicalAddress out.k b h t c+byte.val)=api.encode (kValue b h t c) byte
  frame : ∀ address,Caller api embed rq rk before p address →
    (address<out.q.base ∨ out.q.base+4*rq.size≤address) →
    (address<out.k.base ∨ out.k.base+4*rk.size≤address) → api.byte out.memory address=api.byte before address

theorem compose {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte) (embed : β→α)
    (project : α→β) (op : RopeEmbedding.Operations α) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api embed rq rk before) (laws : StorageLaws api embed rq rk before p)
    (accepted : inputCheck api embed rq rk before p=Bool.true) (qValue kValue : Nat→Nat→Nat→Nat→β)
    (firstValues : ∀ b h t c,b<rq.batches → h<rq.heads → t<rq.sequence → c<rq.dimension →
      RopeEmbedding.Native.logicalRead api (firstResult api embed project op rq rk before p).memory p.first.call.output b h t c=qValue b h t c)
    (secondValues : ∀ b h t c,b<rk.batches → h<rk.heads → t<rk.sequence → c<rk.dimension →
      RopeEmbedding.Native.logicalRead api (secondResult api embed project op rq rk before p).memory
        (secondStage api embed project op rq rk before p).call.output b h t c=kValue b h t c) :
    PairOutput api embed project op rq rk before p qValue kValue (pairResult api embed project op rq rk before p) := by
  have qValues : ∀ b h t c,b<rq.batches → h<rq.heads → t<rq.sequence → c<rq.dimension →
      RopeEmbedding.Native.logicalRead api (pairResult api embed project op rq rk before p).memory p.first.call.output b h t c=qValue b h t c := by
    intro b h t c hb hh ht hc
    rw [show (pairResult api embed project op rq rk before p).memory=(secondResult api embed project op rq rk before p).memory from rfl,
      second_keeps_first api embed project op rq rk before p laws accepted b h t c hb hh ht hc,
      firstValues b h t c hb hh ht hc]
  refine ⟨rfl,rfl,p.first.call.outputShape,(secondStage api embed project op rq rk before p).call.outputShape,
    rfl,rfl,qValues,secondValues,?_,?_,?_⟩
  · intro b h t c hb hh ht hc byte
    rw [api.read_bytes]
    exact congrArg (fun w => api.encode w byte) (qValues b h t c hb hh ht hc)
  · intro b h t c hb hh ht hc byte
    rw [api.read_bytes]
    exact congrArg (fun w => api.encode w byte) (secondValues b h t c hb hh ht hc)
  · exact pair_frame api embed project op rq rk before p laws

def Sources {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte) (embed : β→α)
    (project : α→β) (op : RopeEmbedding.Operations α) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api embed rq rk before) : Prop :=
  let k := secondStage api embed project op rq rk before p
  RopeEmbedding.Native.SourcePrograms api embed project op rq Bool.false p.first.state p.first.call ∧
  RopeEmbedding.Native.SourcePrograms api embed project op rk Bool.false k.state k.call ∧
  RopeEmbedding.Native.IndexABI api embed rq p.first.state p.first.call ∧ RopeEmbedding.Native.IndexABI api embed rk k.state k.call

def RealOutcome (api : RopeEmbedding.Memory.API ℝ Store Byte) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api id rq rk before) : Prop :=
  ∃ out,invoke api id id RopeEmbedding.ideal rq rk before p=some out ∧
    PairOutput api id id RopeEmbedding.ideal rq rk before p
      (RopeEmbedding.Checked.mathematicalTarget api rq Bool.false before p.first.call.input p.first.call.cosine p.first.call.sine)
      (RopeEmbedding.Checked.mathematicalTarget api rk Bool.false before p.inputK p.first.call.cosine p.first.call.sine) out ∧
    Sources api id id RopeEmbedding.ideal rq rk before p

def FP32Outcome (api : RopeEmbedding.Memory.API RopeEmbedding.Numerics.Word Store (BitVec 8)) (n : RopeEmbedding.Numerics.Primitives)
    (rq rk : RopeEmbedding.Request) (before : Store) (p : PairPreparation api RopeEmbedding.Numerics.leaf rq rk before) : Prop :=
  ∃ out,invoke api RopeEmbedding.Numerics.leaf RopeEmbedding.Numerics.Tagged.word (RopeEmbedding.Numerics.operations n) rq rk before p=some out ∧
    PairOutput api RopeEmbedding.Numerics.leaf RopeEmbedding.Numerics.Tagged.word (RopeEmbedding.Numerics.operations n) rq rk before p
      (RopeEmbedding.Checked.machineTarget api n rq Bool.false before p.first.call.input p.first.call.cosine p.first.call.sine)
      (RopeEmbedding.Checked.machineTarget api n rk Bool.false before p.inputK p.first.call.cosine p.first.call.sine) out ∧
    Sources api RopeEmbedding.Numerics.leaf RopeEmbedding.Numerics.Tagged.word (RopeEmbedding.Numerics.operations n) rq rk before p ∧
    (∀ w k,api.encode w k=RopeEmbedding.Numerics.wordByte w k)

theorem whole_real (api : RopeEmbedding.Memory.API ℝ Store Byte) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api id rq rk before) (laws : StorageLaws api id rq rk before p)
    (accepted : check api id id RopeEmbedding.ideal rq rk before p=Bool.true) : RealOutcome api rq rk before p := by
  have admitted := accepted
  simp only [check,Bool.and_eq_true] at admitted
  let k := secondStage api id id RopeEmbedding.ideal rq rk before p
  have qOut := RopeEmbedding.Checked.whole_real api rq Bool.false p.first.state p.first.call Bool.false Bool.true admitted.1.2
  have kOut := RopeEmbedding.Checked.whole_real api rk Bool.false k.state k.call Bool.false Bool.true admitted.2
  refine ⟨pairResult api id id RopeEmbedding.ideal rq rk before p,by simp [invoke,accepted],?_,qOut.2.1,kOut.2.1,qOut.2.2,kOut.2.2⟩
  apply compose api id id RopeEmbedding.ideal rq rk before p laws admitted.1.1
  · intro b h t c hb hh ht hc
    simpa only [firstResult,qOut.1.returned,p.first.entered] using qOut.1.logical b h t c hb hh ht hc
  · intro b h t c hb hh ht hc
    have value := kOut.1.logical b h t c hb hh ht hc
    rw [kOut.1.returned] at value
    exact value.trans (real_target_transport api rq rk before p laws admitted.1.1 b h t c hb hh ht hc)

theorem whole_fp32 (api : RopeEmbedding.Memory.API RopeEmbedding.Numerics.Word Store (BitVec 8)) (n : RopeEmbedding.Numerics.Primitives)
    (rq rk : RopeEmbedding.Request) (before : Store) (p : PairPreparation api RopeEmbedding.Numerics.leaf rq rk before)
    (laws : StorageLaws api RopeEmbedding.Numerics.leaf rq rk before p) (codec : ∀ w k,api.encode w k=RopeEmbedding.Numerics.wordByte w k)
    (accepted : check api RopeEmbedding.Numerics.leaf RopeEmbedding.Numerics.Tagged.word (RopeEmbedding.Numerics.operations n) rq rk before p=Bool.true) :
    FP32Outcome api n rq rk before p := by
  have admitted := accepted
  simp only [check,Bool.and_eq_true] at admitted
  let k := secondStage api RopeEmbedding.Numerics.leaf RopeEmbedding.Numerics.Tagged.word (RopeEmbedding.Numerics.operations n) rq rk before p
  have qOut := RopeEmbedding.Checked.whole_fp32 api n rq Bool.false p.first.state p.first.call Bool.false Bool.true codec admitted.1.2
  have kOut := RopeEmbedding.Checked.whole_fp32 api n rk Bool.false k.state k.call Bool.false Bool.true codec admitted.2
  refine ⟨pairResult api RopeEmbedding.Numerics.leaf RopeEmbedding.Numerics.Tagged.word (RopeEmbedding.Numerics.operations n) rq rk before p,by simp [invoke,accepted],?_,
    ⟨qOut.2.1,kOut.2.1,qOut.2.2.1,kOut.2.2.1⟩,codec⟩
  apply compose api RopeEmbedding.Numerics.leaf RopeEmbedding.Numerics.Tagged.word (RopeEmbedding.Numerics.operations n) rq rk before p laws admitted.1.1
  · intro b h t c hb hh ht hc
    simpa only [firstResult,qOut.1.returned,p.first.entered] using qOut.1.logical b h t c hb hh ht hc
  · intro b h t c hb hh ht hc
    have value := kOut.1.logical b h t c hb hh ht hc
    rw [kOut.1.returned] at value
    exact value.trans (word_target_transport api n rq rk before p laws admitted.1.1 b h t c hb hh ht hc)

#axiomsClean whole_real
#axiomsClean whole_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastRopeEmbedding
