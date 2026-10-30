import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.SourceBridge
import VeriTile.Triton.Concurrent.EventIR

/-! Reification of the existing source decoder into closed shared event data.
The numerical expressions are obtained by interpreting the protected quotation
over symbolic scalar primitives, not by maintaining another kernel AST. -/
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.ClosedSource
open Whole001 Source
open VeriTile.Triton
noncomputable section

def symbolic : Instructions Concurrent.Expr :=
  ⟨.zero,.mma8,.halfRoundTrip,.add⟩

def backend (ops : Instructions α) : Concurrent.Backend α :=
  ⟨ops.zero,ops.atomicAdd,ops.mma8,ops.halfRoundTrip⟩

def inputA (p : Parameters) : A p → Concurrent.Expr := fun i=>.input 0 i.val
def inputB (p : Parameters) : B p → Concurrent.Expr := fun i=>.input 1 i.val

/-- All store payload callbacks disappear here. Only closed Expr data enters
the shared executor; atomics still read current C and scratch at execution. -/
def lowerEvent (p : Parameters) : Source.Event Concurrent.Expr p → Concurrent.Event
  | .storeOutput i value=>.store 0 i.val (value (inputA p) (inputB p))
  | .storeScratch i value=>.store 1 i.val (value (inputA p) (inputB p))
  | .atomic i j=>.atomicAdd 0 i.val (.load 1 j.val)

/-- The mutation tape decodes the actual six quotation bodies before lowering.
Separate read-only events will retain source accesses without output events. -/
def mutationTape (p : Parameters) : Option (List Concurrent.Event) := do
  let program ← Reader.compile kernels
  if p.h.m=0 ∨ p.h.n=0 then some [] else
    let events ← Reader.events p program symbolic
    some ((events.first.map (lowerEvent p)) ++ (events.reduce.map (lowerEvent p)) ++
      (events.full.map (lowerEvent p)))

theorem mutationTape_eq (p : Parameters) : mutationTape p=some
    (if p.h.m=0 ∨ p.h.n=0 then [] else
      (firstWave p symbolic).map (lowerEvent p) ++ (reduceWave p).map (lowerEvent p) ++
        (fullWave p symbolic).map (lowerEvent p)) := by
  unfold mutationTape
  rw [Reader.source_compiles]
  simp only [Option.bind_eq_bind,Option.bind_some,Option.pure_def]
  split
  · rfl
  · rw [Reader.all_events]
    rfl

#axiomsClean mutationTape_eq
#specNonCircular mutationTape avoiding [Source.call,Source.firstWave,Source.reduceWave,Source.fullWave,Memory.run,Memory.step]
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.ClosedSource
