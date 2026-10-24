import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.HostExecution

namespace VeriTile.Bench.Repairs.FastCeLoss.Host
open Source API HostStages
open CrossEntropy1.Host (Step)
noncomputable section
set_option maxHeartbeats 5000000

inductive FinishExecution {α σ : Type} (a : Arithmetic α) (runtime : Runtime σ) (stream : Nat)
    (s : Shape) (logits labels losses chunks : Descriptor) (label : Nat → Int)
    (initial : World α σ) : Descriptor → World α σ → Type where
  | single (same : s.chunks=1) : FinishExecution a runtime stream s logits labels losses chunks label initial chunks initial
  | chunked (different : s.chunks≠1) (lse : Descriptor) (allocated combined added final : World α σ)
      (allocation : allocate runtime [s.rows] .fp32 logits.device initial=.ok lse allocated)
      (combine : perform runtime (call stream .logsumexp [chunks] [lse])
        (lseStage a s (baseBuffers logits labels losses chunks lse)) id allocated=.ok () combined)
      (addition : perform runtime (call stream .addInplace [losses,lse] [losses])
        (addStage a s (baseBuffers logits labels losses chunks lse)) id combined=.ok () added)
      (mask : perform runtime (call stream .maskIgnored [losses,labels] [losses])
        (maskStage a s (baseBuffers logits labels losses chunks lse) label) id added=.ok () final) :
      FinishExecution a runtime stream s logits labels losses chunks label initial lse final

theorem finish_success {α σ : Type} (a : Arithmetic α) (runtime : Runtime σ) (stream : Nat)
    (s : Shape) (logits labels losses chunks : Descriptor) (label : Nat → Int)
    (initial final : World α σ) (lse : Descriptor)
    (success : finishForward a runtime stream s logits labels losses chunks label initial=.ok lse final) :
    Nonempty (FinishExecution a runtime stream s logits labels losses chunks label initial lse final) := by
  by_cases single : s.chunks=1
  · simp only [finishForward,if_pos single,pure,pureAction,Step.ok.injEq] at success
    rcases success with ⟨rfl,rfl⟩
    exact ⟨.single single⟩
  · simp only [finishForward,if_neg single] at success
    obtain ⟨global,allocated,halloc,rest⟩ := bind_success success
    obtain ⟨doneToken,combined,hcombine,rest⟩ := bind_success rest
    cases doneToken
    obtain ⟨doneToken,added,hadd,rest⟩ := bind_success rest
    cases doneToken
    obtain ⟨doneToken,masked,hmask,rest⟩ := bind_success rest
    cases doneToken
    change Step.ok global masked=Step.ok lse final at rest
    rcases Step.ok.inj rest with ⟨rfl,rfl⟩
    exact ⟨.chunked single global allocated combined added masked halloc hcombine hadd hmask⟩

theorem FinishExecution.memory {α σ : Type} {a : Arithmetic α} {runtime : Runtime σ} {stream : Nat}
    {s : Shape} {logits labels losses chunks : Descriptor} {label : Nat → Int}
    {initial final : World α σ} {lse : Descriptor}
    (execution : FinishExecution a runtime stream s logits labels losses chunks label initial lse final) :
    final.memory=HostStages.finish a s (baseBuffers logits labels losses chunks lse) label initial.memory ∧
      final.integer=initial.integer := by
  cases execution with
  | single same => simp [HostStages.finish,same]
  | chunked different lse allocated combined added final allocation combine addition mask =>
    have ha := allocate_memory runtime _ _ _ _ _ _ allocation
    have hc := perform_memory runtime _ _ _ _ _ _ combine
    have hp := perform_memory runtime _ _ _ _ _ _ addition
    have hm := perform_memory runtime _ _ _ _ _ _ mask
    constructor
    · rw [hm.1,hp.1,hc.1,ha.1]
      simp only [HostStages.finish,if_neg different]
    · exact hm.2.1.trans (hp.2.1.trans (hc.2.1.trans ha.2.1))

inductive CountExecution {α σ : Type} (runtime : Runtime σ) (stream : Nat) (args : Arguments)
    (logits : Descriptor) (label : Nat → Int) (initial : World α σ) : Option Descriptor → World α σ → Type where
  | explicit (item : PythonScalar) (items : args.items=some item) :
      CountExecution runtime stream args logits label initial none initial
  | counted (items : args.items=none) (count : Descriptor) (allocated final : World α σ)
      (allocation : allocate runtime [] .int64 logits.device initial=.ok count allocated)
      (operation : perform runtime (call stream .countNonignored [args.labels] [count]) id
        (writeInteger count (Source.count args.shape label)) allocated=.ok () final) :
      CountExecution runtime stream args logits label initial (some count) final

theorem count_success {α σ : Type} (runtime : Runtime σ) (stream : Nat) (args : Arguments)
    (logits : Descriptor) (label : Nat → Int) (initial final : World α σ) (count : Option Descriptor)
    (success : countItems runtime stream args logits label initial=.ok count final) :
    Nonempty (CountExecution runtime stream args logits label initial count final) := by
  cases items : args.items with
  | some item =>
    simp only [countItems,items,pure,pureAction,Step.ok.injEq] at success
    rcases success with ⟨rfl,rfl⟩
    exact ⟨.explicit item items⟩
  | none =>
    simp only [countItems,items] at success
    obtain ⟨tensor,allocated,halloc,rest⟩ := bind_success success
    obtain ⟨doneToken,counted,hcount,rest⟩ := bind_success rest
    cases doneToken
    change Step.ok (some tensor) counted=Step.ok count final at rest
    rcases Step.ok.inj rest with ⟨rfl,rfl⟩
    exact ⟨.counted items tensor allocated counted halloc hcount⟩

def countMemory (s : Shape) (label : Nat → Int) (count : Option Descriptor) (memory : Address → Int) : Address → Int :=
  match count with
  | none => memory
  | some tensor => writeInteger tensor (Source.count s label) memory

theorem CountExecution.memory {α σ : Type} {runtime : Runtime σ} {stream : Nat} {args : Arguments}
    {logits : Descriptor} {label : Nat → Int} {initial final : World α σ} {count : Option Descriptor}
    (execution : CountExecution runtime stream args logits label initial count final) :
    final.memory=initial.memory ∧ final.integer=countMemory args.shape label count initial.integer := by
  cases execution with
  | explicit item items => exact ⟨rfl,rfl⟩
  | counted items count allocated final allocation operation =>
    have ha := allocate_memory runtime _ _ _ _ _ _ allocation
    have ho := perform_memory runtime _ _ _ _ _ _ operation
    constructor
    · exact ho.1.trans ha.1
    · rw [ho.2.1,ha.2.1]
      rfl

structure ForwardExecution {α σ : Type} (a : Arithmetic α) (scalars : ScalarInterpreter α)
    (runtime : Runtime σ) (stream : Nat) (args : Arguments) (initial final : World α σ) (result : ForwardReturn) where
  labelsShape : args.labels.tensor.shape=[args.batch,args.sequence]
  logits : Descriptor
  logitsState : World α σ
  logitsView : view runtime args.logits [args.shape.rows,args.shape.vocab] initial=.ok logits logitsState
  labels : Descriptor
  labelsState : World α σ
  labelsView : view runtime args.labels [args.shape.rows] logitsState=.ok labels labelsState
  losses : Descriptor
  lossesState : World α σ
  lossesAllocation : allocate runtime [args.shape.rows] .fp32 logits.device labelsState=.ok losses lossesState
  chunks : Descriptor
  chunksState : World α σ
  chunksAllocation : allocate runtime (if args.shape.chunks=1 then [args.shape.rows] else [args.shape.rows,args.shape.chunks])
    .fp32 logits.device lossesState=.ok chunks chunksState
  settings : ¬(args.shape.chunks=1 ∧ 65536<args.shape.block)
  launched : World α σ
  kernel : perform runtime (call stream (.launch
    (if args.shape.chunks=1 then Kernel.forward else Kernel.chunked)
    (if args.shape.chunks=1 then [args.shape.rows] else [args.shape.rows,args.shape.chunks])
    args.shape.block (if args.shape.chunks=1 then selectedWarps args.shape.block else 32) args.options)
    [logits,labels] [losses,chunks] [args.capArgument,args.scaleArgument]
    [logits.strides.headD 0,(args.shape.vocab : Int),(args.shape.chunks : Int)])
    (forwardGrid a args.options args.shape (baseBuffers logits labels losses chunks chunks)
      (labelValues labels chunksState)) id chunksState=.ok () launched
  lse : Descriptor
  finished : World α σ
  finish : FinishExecution a runtime stream args.shape logits labels losses chunks (labelValues labels chunksState) launched lse finished
  saved : World α σ
  save : perform runtime (call stream .saveContext [logits,lse,labels] []) id id finished=.ok () saved
  count : Option Descriptor
  counted : World α σ
  counting : CountExecution runtime stream args logits (labelValues labels chunksState) saved count counted
  sum : Descriptor
  sumAllocated : World α σ
  sumAllocation : allocate runtime [] .fp32 logits.device counted=.ok sum sumAllocated
  summed : World α σ
  summation : perform runtime (call stream .sum [losses] [sum])
    (fun memory => writeRows sum.tensor 1 (fun _ =>
      a.hostSum (fun row : Fin args.shape.rows => a.load (memory (addr losses.tensor row.val)))) memory)
    id sumAllocated=.ok () summed
  scalar : Descriptor
  scalarAllocated : World α σ
  scalarAllocation : allocate runtime [] .fp32 logits.device summed=.ok scalar scalarAllocated
  divided : World α σ
  division : perform runtime (call stream .divide ([sum]++count.toList) [scalar] args.items.toList)
    (fun memory => writeRows scalar.tensor 1 (fun _ => a.hostDiv (a.load (memory (addr sum.tensor 0)))
      (denominatorValue a scalars args count scalarAllocated)) memory) id scalarAllocated=.ok () divided
  returned : perform runtime (call stream .returnValue [scalar] []) id id divided=.ok () final
  resultExact : result=⟨scalar,losses,⟨logits,lse,labels,args.options⟩,chunks,sum,count,args⟩

theorem forward_success {α σ : Type} (a : Arithmetic α) (scalars : ScalarInterpreter α)
    (runtime : Runtime σ) (stream : Nat) (args : Arguments) (initial final : World α σ) (result : ForwardReturn)
    (success : forward a scalars runtime stream args initial=.ok result final) :
    Nonempty (ForwardExecution a scalars runtime stream args initial final result) := by
  by_cases invalid : args.labels.tensor.shape≠[args.batch,args.sequence]
  · simp only [forward,if_pos invalid,fail] at success
    cases success
  · simp only [forward,if_neg invalid] at success
    obtain ⟨logits,logitsState,hx,rest⟩ := bind_success success
    obtain ⟨labels,labelsState,hy,rest⟩ := bind_success rest
    obtain ⟨losses,lossesState,hloss,rest⟩ := bind_success rest
    obtain ⟨chunks,chunksState,hchunks,rest⟩ := bind_success rest
    obtain ⟨readChunks,readState,hread,rest⟩ := bind_success (first:=read) rest
    simp only [read,Step.ok.injEq] at hread
    rcases hread with ⟨rfl,rfl⟩
    by_cases blocked : args.shape.chunks=1 ∧ 65536<args.shape.block
    · simp only [if_pos blocked,fail] at rest
      cases rest
    · simp only [if_neg blocked] at rest
      obtain ⟨doneToken,launched,hkernel,rest⟩ := bind_success rest
      cases doneToken
      obtain ⟨lse,finished,hfinish,rest⟩ := bind_success rest
      obtain ⟨doneToken,saved,hsave,rest⟩ := bind_success rest
      cases doneToken
      obtain ⟨count,counted,hcount,rest⟩ := bind_success rest
      obtain ⟨sum,sumAllocated,hsumAlloc,rest⟩ := bind_success rest
      obtain ⟨doneToken,summed,hsum,rest⟩ := bind_success rest
      cases doneToken
      obtain ⟨scalar,scalarAllocated,hscalarAlloc,rest⟩ := bind_success rest
      obtain ⟨readScalar,readState,hread,rest⟩ := bind_success (first:=read) rest
      simp only [read,Step.ok.injEq] at hread
      rcases hread with ⟨rfl,rfl⟩
      obtain ⟨doneToken,divided,hdivide,rest⟩ := bind_success rest
      cases doneToken
      obtain ⟨doneToken,returned,hreturn,rest⟩ := bind_success rest
      cases doneToken
      change Step.ok _ returned=Step.ok result final at rest
      rcases Step.ok.inj rest with ⟨rfl,rfl⟩
      exact ⟨⟨not_ne_iff.mp invalid,logits,logitsState,hx,labels,labelsState,hy,losses,lossesState,hloss,
        chunks,chunksState,hchunks,blocked,launched,hkernel,lse,finished,
        Classical.choice (finish_success a runtime stream args.shape logits labels losses chunks _ _ _ _ hfinish),
        saved,hsave,count,counted,Classical.choice (count_success runtime stream args logits _ _ _ _ hcount),
        sum,sumAllocated,hsumAlloc,summed,hsum,scalar,scalarAllocated,hscalarAlloc,divided,hdivide,hreturn,rfl⟩⟩

#axiomsClean finish_success
#axiomsClean FinishExecution.memory
#axiomsClean count_success
#axiomsClean forward_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Host
