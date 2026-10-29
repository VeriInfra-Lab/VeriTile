import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.HistoricalCorrect
namespace VeriTile.Bench.Repairs.RopeBackwardTransform
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding.Memory (API)
open Numerics (Word Tagged leaf)
noncomputable section
set_option maxHeartbeats 3000000

def machineTarget (api : API Word Store Byte) (p : Numerics.Primitives) (r : Request)
    (before : Store) (input : Bool→Native.Tensor) (cosine sine : Native.Tensor)
    (key : Bool) (b h time col : Nat) : Word :=
  if col<2*r.half then
    let j := if col<r.half then col else col-r.half
    Numerics.pair p (decide (r.half≤col))
      (Native.logicalRead api before (input key) b h time j)
      (Native.logicalRead api before (input key) b h time (j+r.half))
      (api.read before (Native.tableAddress cosine time j))
      (api.read before (Native.tableAddress sine time j))
  else Native.logicalRead api before (input key) b h time col

/-- Exact word profile for the observed default-fusion instruction graph.
The primitive implementation parameter is local; native conformance is deferred. -/
theorem default_profile_words (api : API Word Store Byte) (p : Numerics.Primitives)
    (r : Request) (m : Source.Memory Tagged) (call : Native.Preparation api leaf r m)
    (_abi : Native.IndexABI api leaf r m call)
    (key : Bool) (b h time col : Nat) (hb : b<r.batches) (hh : h<r.heads key)
    (ht : time<r.sequence) (hc : col<r.dimension) :
    Native.logicalRead api (Native.entry api leaf Tagged.word (Numerics.operations p) r m call).memory
      (Native.returned (Native.entry api leaf Tagged.word (Numerics.operations p) r m call) key) b h time col=
      machineTarget api p r call.before call.input call.cosine call.sine key b h time col := by
  rw [(Native.whole api leaf Tagged.word (Numerics.operations p) r m call (fun _ => rfl)).2.2.1
    key b h time col hb hh ht hc]
  rw [scalar_binding api leaf (Numerics.operations p) r m call key b h time col hb hh ht hc]
  unfold logicalScalar machineTarget
  split
  · exact Numerics.source_pair p _ _ _ _ _
  · rfl

/-- Whole-call numerical theorem: explicit source execution, returned views,
word/byte profile, independent real adjoint and caller frame. Only reached scalar
mul/neg/FMA exactness is assumed; no final output, pair or reduction oracle. -/
theorem fp_equiv (api : API Word Store (BitVec 8)) (p : Numerics.Primitives)
    (r : Request) (m : Source.Memory Tagged) (call : Native.Preparation api leaf r m)
    (abi : Native.IndexABI api leaf r m call)
    (codec : ∀ w k,api.encode w k=Numerics.wordByte w k)
    (nodes : LocalNodes api p r call.before call.input call.cosine call.sine) :
    GridExec (Numerics.operations p).backend r (Native.launchRows r) m
      (grid (Numerics.operations p).backend r m (Native.launchRows r)) ∧
    (∀ k,Native.returned (Native.entry api leaf Tagged.word (Numerics.operations p) r m call) k=call.output k) ∧
    (∀ k,(Native.returned (Native.entry api leaf Tagged.word (Numerics.operations p) r m call) k).shape=
      [r.batches,r.heads k,r.sequence,r.dimension]) ∧
    (∀ k b h time c,b<r.batches → h<r.heads k → time<r.sequence → c<r.dimension →
      let out := Native.entry api leaf Tagged.word (Numerics.operations p) r m call
      let word := machineTarget api p r call.before call.input call.cosine call.sine k b h time c
      (Native.logicalRead api out.memory (Native.returned out k) b h time c).bits=word.bits ∧
      (∀ byte : Fin 4,api.byte out.memory (Native.logicalAddress (Native.returned out k) b h time c+byte.val)=
        Numerics.wordByte word byte) ∧
      Numerics.decode (Native.logicalRead api out.memory (Native.returned out k) b h time c)=
        mathematicalTarget api Numerics.decode r call.before call.input call.cosine call.sine k b h time c) ∧
    (∀ b,call.caller b → Memory.Outside r (fun k => (call.output k).base) (Native.launchRows r) b →
      api.byte (Native.entry api leaf Tagged.word (Numerics.operations p) r m call).memory b=api.byte call.before b) := by
  obtain ⟨obj,shape,read,frame⟩ := Native.whole api leaf Tagged.word (Numerics.operations p) r m call (fun _ => rfl)
  refine ⟨grid_execution _ _ _ _,obj,shape,?_,frame⟩
  intro k b h time c hb hh ht hc
  dsimp only
  have word := default_profile_words api p r m call abi k b h time c hb hh ht hc
  refine ⟨congrArg (fun w : Word => w.bits) word,?_,?_⟩
  · intro byte
    rw [api.read_bytes,codec]
    exact congrArg (fun w => Numerics.wordByte w byte) word
  · rw [read k b h time c hb hh ht hc,
      scalar_binding api leaf (Numerics.operations p) r m call k b h time c hb hh ht hc]
    exact fp_logical api p r call.before call.input call.cosine call.sine nodes k b h time c hb hh ht

#specNonCircular machineTarget avoiding [Source.kernel,scalarPair,scalarResult,Native.entry,Memory.run]
#axiomsClean default_profile_words
#axiomsClean fp_equiv
end
end VeriTile.Bench.Repairs.RopeBackwardTransform
