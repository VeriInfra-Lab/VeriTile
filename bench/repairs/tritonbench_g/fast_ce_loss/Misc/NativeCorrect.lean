import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.PublicNativeIO
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.FastCeLoss.NativeCorrect
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.Repairs.SoftmaxFlaggems.Native
noncomputable section

/- The .real channel tag is the existing Structural cell representation;
FP32 storage/arithmetic contexts remain fixed by each unchanged typed source. -/

/-- Actual FP32 source and signed-int64 label. Symbolic inputs remain arbitrary
terms; active tile and separately backed target gather retain the old domain. -/
def forwardIO (R : Reductions) (p : Parameters) (row : Nat) : ProgramIO Term Term where
  inputs := [⟨.int,[],p.labels.cast⟩,⟨.real,[p.block],p.logits⟩,⟨.real,[],p.logits⟩]
  outputs := [⟨.real,[],p.lse⟩,⟨.real,[],p.loss⟩]
  scratch := []
  buffers := [p.labels.cast,p.logits,p.lse,p.loss]
  read := PublicIO.forwardRead p row 0 Bool.false
  inputMask := PublicIO.forwardInputMask p 0 Bool.false Bool.false
  write := PublicIO.forwardWrite p row 0 Bool.false Bool.false
  outputMask := PublicIO.forwardOutputMask p 0 Bool.false Bool.false
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [{ (PublicIO.forwardInvocation (algebra R) p row 0 Bool.false) with
    kernel := KernelSource.forward p }]
  encode := id
  observe := fun _ _ => id

/-- Chunk-local logit tile, every chunk's LSE and the masked first-chunk loss. -/
def chunkedIO (R : Reductions) (p : Parameters) (row chunk : Nat) : ProgramIO Term Term where
  inputs := [⟨.int,[],p.labels.cast⟩,⟨.real,[p.block],p.logits⟩,⟨.real,[],p.logits⟩]
  outputs := [⟨.real,[],p.lse⟩,⟨.real,[],p.loss⟩]
  scratch := []
  buffers := [p.labels.cast,p.logits,p.lse,p.loss]
  read := PublicIO.forwardRead p row chunk Bool.false
  inputMask := PublicIO.forwardInputMask p chunk Bool.true Bool.false
  write := PublicIO.forwardWrite p row chunk Bool.true Bool.false
  outputMask := PublicIO.forwardOutputMask p chunk Bool.true Bool.false
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [{ (PublicIO.forwardInvocation (algebra R) p row chunk Bool.true) with
    kernel := KernelSource.chunked p }]
  encode := id
  observe := fun _ _ => id

/-- Original in-place gradient, saved LSE and conditional upstream load. -/
def backwardIO (R : Reductions) (p : Parameters) (row chunk : Nat) : ProgramIO Term Term where
  inputs := [⟨.int,[],p.labels.cast⟩,⟨.real,[p.block],p.logits⟩,
    ⟨.real,[],p.lse⟩,⟨.real,[],p.upstream⟩]
  outputs := [⟨.real,[p.block],p.logits⟩]
  scratch := []
  buffers := [p.labels.cast,p.logits,p.lse,p.upstream]
  read := PublicIO.backwardRead p row chunk Bool.false
  inputMask := PublicIO.backwardInputMask p chunk Bool.false
  write := Fin.cases (fun _ j => row*p.logitsStride+(chunk*p.block+j.1.val)) (fun k => k.elim0)
  outputMask := Fin.cases (fun _ j => chunk*p.block+j.1.val<p.vocab) (fun k => k.elim0)
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [{ (PublicIO.backwardInvocation (algebra R) p row chunk) with
    kernel := KernelSource.backward p }]
  encode := id
  observe := fun _ _ => id

/-- Preserve the original native label range, including backed gathers beyond
the vocabulary. Bounds are proved by the shared IO judgment, not assumed here. -/
def inputDomain (p : Parameters) (xs : ChannelValues Term (PublicIO.forwardInputs p Bool.false)) : Prop :=
  PublicIO.forwardLabel p Bool.false xs = -100 ∨ 0≤PublicIO.forwardLabel p Bool.false xs

/-- Observe the independent ordered FP32 scalar/reduction expression using
the existing instruction and source-local compiler schedules. -/
def forwardFormula (R : Reductions) (I : Instructions) (S : Schedule) (p : Parameters)
    (xs : ChannelValues Term (PublicIO.forwardInputs p Bool.false)) : ChannelValues Word (PublicIO.forwardOutputs p) :=
  fun o j => mapValue (KernelCompiler.observe I S) (PublicIO.forwardOutputs p)[o].dtype
    (PublicIO.forwardFormula (algebra R) p Bool.false xs o j)
def chunkedFormula (R : Reductions) (I : Instructions) (S : Schedule) (p : Parameters) (chunk : Nat)
    (xs : ChannelValues Term (PublicIO.forwardInputs p Bool.false)) : ChannelValues Word (PublicIO.forwardOutputs p) :=
  fun o j => mapValue (KernelCompiler.observe I S) (PublicIO.forwardOutputs p)[o].dtype
    (PublicIO.chunkedFormula (algebra R) p chunk Bool.false xs o j)
def backwardFormula (R : Reductions) (I : Instructions) (S : Schedule) (p : Parameters) (chunk : Nat)
    (xs : ChannelValues Term (PublicIO.backwardInputs p Bool.false)) : ChannelValues Word (PublicIO.backwardOutputs p) :=
  fun o j => mapValue (KernelCompiler.observe I S) (PublicIO.backwardOutputs p)[o].dtype
    (PublicIO.backwardFormula (algebra R) p chunk Bool.false xs o j)

/-- Source-to-native observation, preserving arbitrary input terms and the
original reduction/instruction/compiler primitive parameters R, I and S.
This task does not claim a second optimized kernel or native=Real arithmetic. -/
theorem forward_kernel (R : Reductions) (I : Instructions) (S : Schedule)
    (p : Parameters) (row : Nat) (positive : 0<p.block) (distinct : p.loss≠p.lse) :
    (forwardIO R p row).ObservesOn (fun _ _ => KernelCompiler.observe I S)
      (inputDomain p) (forwardFormula R I S p) :=
  NativeIO.forward_observes R I S p row positive distinct

theorem chunked_kernel (R : Reductions) (I : Instructions) (S : Schedule)
    (p : Parameters) (row chunk : Nat) (positive : 0<p.block) (distinct : p.loss≠p.lse) :
    (chunkedIO R p row chunk).ObservesOn (fun _ _ => KernelCompiler.observe I S)
      (inputDomain p) (chunkedFormula R I S p chunk) :=
  NativeIO.chunked_observes R I S p row chunk positive distinct

theorem backward_kernel (R : Reductions) (I : Instructions) (S : Schedule)
    (p : Parameters) (row chunk : Nat) :
    (backwardIO R p row chunk).Observes (fun _ _ => KernelCompiler.observe I S)
      (backwardFormula R I S p chunk) :=
  NativeIO.backward_observes R I S p row chunk

/- Numerical boundaries: R/I/S are the existing primitive and lowering models.
There is no newly assumed FP rewrite and therefore no transformation-assumption
list to print. Scalar FMA/ignored-zero derivations stay in Proofs/KernelFMA and
Proofs/KernelOutputs. No new numerical experiment is required by this format change. -/
#stmtSurfaceSubset forward_kernel ⊆
  [Reductions,Instructions,Schedule,Parameters,Parameters.block,Parameters.loss,Parameters.lse,
   RegionName,forwardIO,ProgramIO.ObservesOn,KernelCompiler.observe,inputDomain,forwardFormula,
   Term,Word,instInhabitedTerm,IOChannel,MemoryIO.outputs,ProgramIO.toMemoryIO,TileIndex,
   IOChannel.shape,ProgramIO.mapOutputs._proof_1]
#stmtSurfaceSubset chunked_kernel ⊆
  [Reductions,Instructions,Schedule,Parameters,Parameters.block,Parameters.loss,Parameters.lse,
   RegionName,chunkedIO,ProgramIO.ObservesOn,KernelCompiler.observe,inputDomain,chunkedFormula,
   Term,Word,instInhabitedTerm,IOChannel,MemoryIO.outputs,ProgramIO.toMemoryIO,TileIndex,
   IOChannel.shape,ProgramIO.mapOutputs._proof_1]
#stmtSurfaceSubset backward_kernel ⊆
  [Reductions,Instructions,Schedule,Parameters,backwardIO,ProgramIO.Observes,KernelCompiler.observe,backwardFormula,
   Term,Word,instInhabitedTerm,IOChannel,MemoryIO.outputs,ProgramIO.toMemoryIO,TileIndex,
   IOChannel.shape,ProgramIO.mapOutputs._proof_1]
#specNonCircular forwardFormula avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,FP.Structural.exec,FP.Structural.run]
#specNonCircular chunkedFormula avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,FP.Structural.exec,FP.Structural.run]
#specNonCircular backwardFormula avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,FP.Structural.exec,FP.Structural.run]
#axiomsClean forward_kernel
#axiomsClean chunked_kernel
#axiomsClean backward_kernel
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.NativeCorrect
