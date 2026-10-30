import bench.tritonbench_g.chunk_gla_fwd.Proofs.NativeHost

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSource
open VeriTile Triton FP.Structural
open NativeStaging NativeInstructions
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
noncomputable section

def inputMemory (m : Nat → BitVec 32) : Nat → Expr := fun i => .bits (m i)
def materialize (m : Nat → Expr) : Nat → Graph := fun i => lower (m i)

/-- Independent five-stage primitive/store reference, instantiated with the
explicit staging algebra. Native graph construction contains no interpreter. -/
def reference (w : Fin 4) (abi : Physical.ABI c) (scale : ℝ) (m : Nat → BitVec 32) : Nat → Graph :=
  materialize (Physical.consumerReference (algebra w) dot abi scale
    (Physical.producerReference (algebra w) dot abi scale (inputMemory m)))

def execute (w : Fin 4) (abi : Physical.ABI c) (scale : ℝ) (m : Nat → BitVec 32) : Option (Nat → Graph) :=
  ((Physical.producerExecute (algebra w) dot abi scale (inputMemory m)).bind
    (Physical.consumerExecute (algebra w) dot abi scale)).map materialize

/-- All actual source stages successfully construct the same independent
native graph. The only new instantiation has no opaque dot/reduction port. -/
theorem source_to_graph (w : Fin 4) (abi : Physical.ABI c) (scale : ℝ) (m : Nat → BitVec 32) :
    execute w abi scale m=some (reference w abi scale m) := by
  unfold execute reference
  rw [Physical.calls_compose]
  rfl

def outputGraph (w : Fin 4) (abi : Physical.ABI c) (scale : ℝ) (m : Nat → BitVec 32)
    (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) : Graph :=
  lower (FPReturned.value (algebra w) dot c.consumer scale
    (FPStorage.project abi.alloc
      (Physical.producerReference (algebra w) dot abi scale (inputMemory m))) h t v)

/-- Every returned FP32 word is the fully lowered native graph at its unique
source owner/lane. In particular no pending dot/reduction survives a store. -/
theorem returned_graph (w : Fin 4) (abi : Physical.ABI c) (scale : ℝ) (m : Nat → BitVec 32)
    (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    reference w abi scale m ((abi.alloc "o").base+h.val*(c.T*c.V)+t.val*c.V+v.val)=
      outputGraph w abi scale m h t v :=
  congrArg lower (Physical.consumer_tensor (algebra w) dot abi scale
    (Physical.producerReference (algebra w) dot abi scale (inputMemory m)) h t v)

theorem preserves_input (w : Fin 4) (abi : Physical.ABI c)
    (hp : Physical.ProducerFresh abi) (hc : Physical.ConsumerFresh abi)
    (scale : ℝ) (m : Nat → BitVec 32) (r : RegionName) (hr : Physical.tracked r)
    (hra : r≠"attention") (hrs : r≠"scratch") (hro : r≠"o")
    (i : Nat) (hi : i<(abi.alloc r).length) :
    reference w abi scale m ((abi.alloc r).base+i)=.bits (m ((abi.alloc r).base+i)) := by
  unfold reference materialize
  rw [Physical.consumer_preserves_input (algebra w) dot abi hc scale _ r hr hro i hi,
    Physical.producer_preserves_input (algebra w) dot abi hp scale _ r hr hra hrs i hi]
  rfl

/-- The compiler graph is evaluated only by fixed local numerical ports.
Whole-task numerical transport is *derived* by graph induction, rather than
assuming the complete tl.dot, reduceSum, kernel or tensor result. -/
theorem numerical_transport {α β : Type} (f : α → β) (A : ISA α) (B : ISA β)
    (ports : LocalTransport f A B) (w : Fin 4) (abi : Physical.ABI c)
    (scale : ℝ) (m : Nat → BitVec 32) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    f ((outputGraph w abi scale m h t v).eval A)=(outputGraph w abi scale m h t v).eval B :=
  NativeInstructions.transport f A B ports _

theorem returned_numeric {α : Type} (I : ISA α) (w : Fin 4) (abi : Physical.ABI c)
    (scale : ℝ) (m : Nat → BitVec 32) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    (reference w abi scale m ((abi.alloc "o").base+h.val*(c.T*c.V)+t.val*c.V+v.val)).eval I=
      (outputGraph w abi scale m h t v).eval I := by rw [returned_graph]

#specNonCircular outputGraph avoiding [execute,FP.Structural.exec,Physical.producerExecute,Physical.consumerExecute]
#axiomsClean source_to_graph
#axiomsClean returned_graph
#axiomsClean numerical_transport
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSource
