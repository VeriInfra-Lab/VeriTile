import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelOutputs
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelAccess
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelCompiler

namespace VeriTile.Bench.Repairs.FastCeLoss.KernelProfile
open VeriTile.Triton FP.Structural KernelSource KernelModel KernelContract KernelContract.Storage
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
namespace Compiler
abbrev Term := VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.Term
abbrev Reductions := VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.Reductions
abbrev Instructions := VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.Instructions
abbrev Schedule := VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.Schedule
abbrev algebra := VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.algebra
abbrev shape := VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.shape
abbrev observe := KernelCompiler.observe
end Compiler

/-- Every input term leaf contains the actual word of the prepared argument
view. Int64 labels remain exact integers from their own byte-bound region. -/
def entry (p : Parameters) (words : State Word) : State Compiler.Term where
  mem := fun r i => if r=Region.cast p.labels then .mk .int ((words.mem r i).read .int)
    else .mk .real (.leaf r i ((words.mem r i).read .real))
  regs := fun _ _ _ => none
  pids := words.pids
  numPids := words.numPids
  undef := fun d _ _ => defaultValue d

 theorem entry_label (p : Parameters) (words : State Word) : label p (entry p words)=label p words := by
  simp [entry,label]

 theorem entry_word (p : Parameters) (words : State Word) (region : RegionName) (offset : Nat)
    (typed : region≠Region.cast p.labels) :
    ((entry p words).mem region offset).read .real=
      .leaf region offset ((words.mem region offset).read .real) := by simp [entry,typed]

def forwardLSE (R : Compiler.Reductions) (p : Parameters) (words : State Word) (chunk : Nat) : Compiler.Term :=
  KernelModel.cast (Compiler.algebra R) (lseValue (Compiler.algebra R) p (entry p words) chunk)
def fullLoss (R : Compiler.Reductions) (p : Parameters) (words : State Word) : Compiler.Term :=
  KernelModel.cast (Compiler.algebra R) (forwardLoss (Compiler.algebra R) p (entry p words))
def partialLoss (R : Compiler.Reductions) (p : Parameters) (words : State Word) : Compiler.Term :=
  KernelModel.cast (Compiler.algebra R) (chunkLoss (Compiler.algebra R) p (entry p words))
def backwardLane (R : Compiler.Reductions) (p : Parameters) (words : State Word) (lane : Fin p.block) : Compiler.Term :=
  gradientValue (Compiler.algebra R) p (entry p words) lane

 theorem forward (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (views : ForwardViews p 1) (alloc : RegionName → Allocation)
    (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : ForwardInputs p views alloc bytes words)
    (launch : Launch p 1) (pids : LaunchState launch words) :
    let initial := entry p words
    ∃ final, exec (engine (Compiler.algebra R)) (KernelSource.forward p) initial=some final ∧
      Compiler.observe I S (tensorValue final views.lse launch.row)=
        Compiler.observe I S (forwardLSE R p words 0) ∧
      Compiler.observe I S (tensorValue final views.loss launch.row)=
        Compiler.observe I S (fullLoss R p words) ∧
      (∀ region offset, (region≠p.loss ∨ offset≠launch.row) →
        (region≠p.lse ∨ offset≠launch.row) → final.mem region offset=initial.mem region offset) := by
  have valid : label p (entry p words) = -100 ∨ 0≤label p (entry p words) := by
    rw [entry_label]
    exact KernelAccess.forward_label p views alloc bytes words inputs launch.row launch.rowBound pids.1
  obtain ⟨final,execution,lseRead,lossRead,frame⟩ := KernelExecution.forward_outputs
    (Compiler.algebra R) p (entry p words) inputs.parameters.blockPositive valid views.outputsDistinct
  simp only [entry,pids.1] at lseRead lossRead frame
  refine ⟨final,execution,?_,?_,frame⟩
  · apply congrArg (Compiler.observe I S)
    simpa only [tensorValue,views.lseView.index launch.row (by simpa using launch.rowBound),
      views.lseView.regionEq,Nat.add_sub_cancel_left,forwardLSE,entry] using lseRead
  · apply congrArg (Compiler.observe I S)
    simpa only [tensorValue,views.lossView.index launch.row launch.rowBound,
      views.lossView.regionEq,Nat.add_sub_cancel_left,fullLoss,entry] using lossRead

 theorem chunked (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (views : ForwardViews p p.chunks) (alloc : RegionName → Allocation)
    (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : ForwardInputs p views alloc bytes words)
    (launch : Launch p p.chunks) (pids : LaunchState launch words) :
    let initial := entry p words
    ∃ final, exec (engine (Compiler.algebra R)) (KernelSource.chunked p) initial=some final ∧
      Compiler.observe I S (tensorValue final views.lse (launch.row*p.chunks+launch.block))=
        Compiler.observe I S (forwardLSE R p words launch.block) ∧
      (launch.block=0 → Compiler.observe I S (tensorValue final views.loss launch.row)=
        Compiler.observe I S (partialLoss R p words)) ∧
      (∀ region offset, (launch.block=0 → region≠p.loss ∨ offset≠launch.row) →
        (region≠p.lse ∨ offset≠launch.row*p.chunks+launch.block) →
        final.mem region offset=initial.mem region offset) := by
  have valid : label p (entry p words) = -100 ∨ 0≤label p (entry p words) := by
    rw [entry_label]
    exact KernelAccess.forward_label p views alloc bytes words inputs launch.row launch.rowBound pids.1
  obtain ⟨final,execution,lseRead,lossRead,frame⟩ := KernelExecution.chunked_outputs
    (Compiler.algebra R) p (entry p words) inputs.parameters.blockPositive valid views.outputsDistinct
  simp only [entry,pids.1,pids.2] at lseRead lossRead frame
  refine ⟨final,execution,?_,?_,frame⟩
  · have hi : launch.row*p.chunks+launch.block<p.rows*p.chunks := by
      have bound := Nat.mul_le_mul_right p.chunks (Nat.succ_le_of_lt launch.rowBound)
      have hb := launch.blockBound
      simp only [Nat.succ_mul] at bound
      omega
    apply congrArg (Compiler.observe I S)
    simpa only [tensorValue,views.lseView.index _ hi,views.lseView.regionEq,
      Nat.add_sub_cancel_left,forwardLSE,entry] using lseRead
  · intro first
    apply congrArg (Compiler.observe I S)
    simpa only [tensorValue,views.lossView.index launch.row launch.rowBound,
      views.lossView.regionEq,Nat.add_sub_cancel_left,partialLoss,entry] using lossRead first

 theorem backward (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (views : BackwardViews p) (alloc : RegionName → Allocation)
    (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : BackwardInputs p views alloc bytes words)
    (launch : Launch p blocks) (pids : LaunchState launch words) :
    let initial := entry p words
    ∃ final, exec (engine (Compiler.algebra R)) (KernelSource.backward p) initial=some final ∧
      (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.vocab →
        Compiler.observe I S (tensorValue final views.logits
          (launch.row*p.vocab+(launch.block*p.block+lane.val)))=
        Compiler.observe I S (backwardLane R p words lane)) ∧
      (∀ region offset,
        (∀ lane : Fin p.block, launch.block*p.block+lane.val<p.vocab →
          region≠p.logits ∨ offset≠launch.row*p.logitsStride+(launch.block*p.block+lane.val)) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,reads,frame⟩ := KernelExecution.backward_outputs (Compiler.algebra R) p (entry p words)
  simp only [KernelExecution.gradientOffset,entry,pids.1,pids.2] at reads frame
  refine ⟨final,execution,?_,frame⟩
  intro lane active
  apply congrArg (Compiler.observe I S)
  simpa only [tensorValue,matrix_index _ _ _ _ _ _ _ views.logitsView launch.rowBound active,
    views.logitsView.regionEq,Nat.add_assoc,Nat.add_sub_cancel_left,backwardLane,entry] using reads lane active

 theorem literal_softcap (p : Parameters) (o : KernelABI.Options p) (C : KernelABI.Conversions)
    (I : Compiler.Instructions) (S : Compiler.Schedule) (abi : KernelABI.LiteralProfile p o C I)
    (enabled : p.hasSoftcap=Bool.true) :
    Compiler.observe I S (.literal (some .fp32) .real p.softcap)=KernelABI.convert C o.softcap :=
  abi.softcap enabled

 theorem literal_scaling (p : Parameters) (o : KernelABI.Options p) (C : KernelABI.Conversions)
    (I : Compiler.Instructions) (S : Compiler.Schedule) (abi : KernelABI.LiteralProfile p o C I)
    (enabled : p.hasScaling=Bool.true) :
    Compiler.observe I S (.literal (some .fp32) .real p.scaling)=KernelABI.convert C o.scaling :=
  abi.scaling enabled

#axiomsClean forward
#axiomsClean chunked
#axiomsClean backward
#axiomsClean literal_softcap
#axiomsClean literal_scaling
#specNonCircular forwardLSE avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,FP.Structural.exec,FP.Structural.run]
#specNonCircular fullLoss avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,FP.Structural.exec,FP.Structural.run]
#specNonCircular partialLoss avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,FP.Structural.exec,FP.Structural.run]
#specNonCircular backwardLane avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,FP.Structural.exec,FP.Structural.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.KernelProfile
