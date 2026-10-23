import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.Input

namespace VeriTile.Bench.Repairs.FastCeLoss.Host
open Source API Complete
noncomputable section
set_option maxHeartbeats 6000000

/-- Only the logits member is read by these input-only graph formulas. Other
members are placeholders; this record is not an execution buffer layout. -/
def graphBuffers (args : Arguments) : Buffers :=
  baseBuffers args.logits args.labels args.logits args.logits args.logits

def entryScalar {α σ : Type} (a : Arithmetic α) (scalars : ScalarInterpreter α)
    (args : Arguments) (initial : World α σ) : α :=
  scalarGraph a args.options args.shape (graphBuffers args) (labelValues args.labels initial)
    (args.items.map scalars.apply) initial.memory

def entryGradient {α σ : Type} (a : Arithmetic α) (scalars : ScalarInterpreter α)
    (args : Arguments) (upstream : Descriptor) (initial : World α σ) (row col : Nat) : α :=
  gradientGraph a args.options args.shape (graphBuffers args) (labelValues args.labels initial)
    (args.items.map scalars.apply) (initial.memory (addr upstream.tensor 0)) initial.memory row col

variable {α σ : Type} {a : Arithmetic α} {scalars : ScalarInterpreter α}
  {runtime : Runtime σ} {live : σ → Nat → Prop} {stream : Nat} {args : Arguments}
  {upstream : Descriptor} {initial final : World α σ} {result : WholeReturn}

 theorem WholeExecution.tile_entry (e : WholeExecution a scalars runtime live stream args upstream initial final result)
    (q : Task) : Source.tile a args.options args.shape (e.forward.buffers e.backward.expanded) initial.memory q=
      Source.tile a args.options args.shape (graphBuffers args) initial.memory q := by
  funext lane
  simp only [Source.tile,ForwardExecution.buffers,graphBuffers,baseBuffers,addr,
    e.forwardFacts.logitsStorage,e.forwardFacts.logitsOffset]

 theorem WholeExecution.scalar_entry (e : WholeExecution a scalars runtime live stream args upstream initial final result) :
    final.memory (addr result.forward.scalar.tensor 0)=entryScalar a scalars args initial := by
  rw [e.scalar_descriptor,e.scalar_output]
  simp only [entryScalar,scalarGraph,lossGraph,lseGraph,forwardLoss,forwardLSE,e.tile_entry,e.label_values]
  simp only [target,graphBuffers,baseBuffers,ForwardExecution.buffers,addr,e.forwardFacts.logitsStorage,
    e.forwardFacts.logitsOffset]

 theorem WholeExecution.gradient_entry (e : WholeExecution a scalars runtime live stream args upstream initial final result)
    (upstreamLive : live initial.native upstream.tensor.storage)
    (row col : Nat) (hr : row<args.shape.rows) (hc : col<args.shape.vocab) :
    final.memory (addr args.logits.tensor (row*args.shape.vocab+col))=
      entryGradient a scalars args upstream initial row col := by
  have h := e.gradient_output upstreamLive row col hr hc
  rw [e.logits_address] at h
  rw [h]
  simp only [entryGradient,gradientGraph,lseGraph,forwardLSE,e.tile_entry,e.label_values]
  simp only [graphBuffers,baseBuffers,ForwardExecution.buffers,addr,e.forwardFacts.logitsStorage,
    e.forwardFacts.logitsOffset]

 theorem scalar_computed {σ : Type} (args : Arguments) (initial : World Floating.Expr σ) (word : Floating.Word) :
    entryScalar Floating.arithmetic fpScalar args initial≠.raw word := by
  intro impossible
  cases impossible

 theorem gradient_computed {σ : Type} (args : Arguments) (upstream : Descriptor)
    (initial : World Floating.Expr σ) (row col : Nat) (word : Floating.Word) :
    entryGradient Floating.arithmetic fpScalar args upstream initial row col≠.raw word := by
  intro impossible
  cases impossible

#axiomsClean WholeExecution.scalar_entry
#axiomsClean WholeExecution.gradient_entry
#axiomsClean scalar_computed
#axiomsClean gradient_computed
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Host
