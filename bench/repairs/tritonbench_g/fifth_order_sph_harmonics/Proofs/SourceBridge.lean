import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Grid
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Storage

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.SourceBridge
open VeriTile.Triton FP.Structural Packing Storage
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 64000
namespace Old
abbrev written := @VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.written
abbrev expected := @VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.expected
abbrev offset := @VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.combinedOffset
abbrev execution_memory := @VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.execution_memory
abbrev written_scatter := @VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.written_scatter
end Old

def extent (p : Parameters) : Grid.Direction → Nat
  | .forward=>p.outputWidth*p.rows | .backward=>3*p.rows

def result (M : Algebra Word) (d : Grid.Direction) (p : Parameters) (s : State Word) : State Word :=
  match d with | .forward=>(Execution.forward M p s).val | .backward=>(Execution.backward M p s).val

theorem executes (M : Algebra Word) (d : Grid.Direction) (p : Parameters) (s : State Word) :
    FP.Structural.exec M (Grid.source d p) s=some (result M d p s) := by
  cases d
  · exact (Execution.forward M p s).property
  · exact (Execution.backward M p s).property

/-- Channel-major source store order, with values captured before the first store. -/
def byteStores (M : Algebra Word) (d : Grid.Direction) (p : Parameters) (base : Nat)
    (s : State Word) (heap : Heap) : Heap :=
  match d with
  | .forward=>
    (TileShape.allIndices [11,p.blockSize]).foldl (fun m q=>
      if Old.offset (forwardParameters p) s q<p.outputWidth*p.rows then
        write m (base+4*Old.offset (forwardParameters p) s q)
          (Old.expected M (forwardParameters p) s q.1 q.2) else m) heap
  | .backward=>
    (TileShape.allIndices [3,p.blockSize]).foldl (fun m q=>
      if Readback.combinedOffset p s q<3*p.rows then
        write m (base+4*Readback.combinedOffset p s q)
          (Execution.backwardValue M p s q.1 q.2) else m) heap

theorem program_projection (M : Algebra Word) (d : Grid.Direction) (p : Parameters)
    (base : Nat) (before : Heap) (s : State Word) :
    image (Grid.output d) base (extent p d) before (result M d p s).mem=
      byteStores M d p base s (image (Grid.output d) base (extent p d) before s.mem) := by
  cases d with
  | forward=>
    change image outputRegion base (p.outputWidth*p.rows) before (Execution.forward M p s).val.mem=_
    rw [Execution.forward_memory,Old.written_scatter]
    exact fold_projection _ _ _ _ _ _ _ (fun _ _ h=>h) _ _
  | backward=>
    change image gradientRegion base (3*p.rows) before (Execution.backward M p s).val.mem=_
    rw [Execution.backward_memory,Readback.written_scatter]
    exact fold_projection _ _ _ _ _ _ _ (fun _ _ h=>h) _ _

structure Machine where
  logical : State Word
  bytes : Heap

/-- The machine advances the complete source interpreter and replays only its
literal emitted stores on the evolving byte heap. -/
def run (M : Algebra Word) (d : Grid.Direction) (p : Parameters) (base : Nat) :
    List Nat → Machine → Machine
  | [],s=>s
  | pid::rest,s=>
    let entry:=Grid.atProgram s.logical pid
    run M d p base rest ⟨result M d p entry,byteStores M d p base entry s.bytes⟩

theorem run_projection (M : Algebra Word) (d : Grid.Direction) (p : Parameters) (base : Nat)
    (order : List Nat) (before : Heap) (s : Machine)
    (related : s.bytes=image (Grid.output d) base (extent p d) before s.logical.mem) :
    Grid.execute M d p order s.logical=some (run M d p base order s).logical ∧
    (run M d p base order s).bytes=
      image (Grid.output d) base (extent p d) before (run M d p base order s).logical.mem := by
  induction order generalizing s with
  | nil=>exact ⟨rfl,related⟩
  | cons pid rest ih=>
    let entry:=Grid.atProgram s.logical pid
    let next:Machine:=⟨result M d p entry,byteStores M d p base entry s.bytes⟩
    have nextRelated : next.bytes=image (Grid.output d) base (extent p d) before next.logical.mem := by
      dsimp [next]
      rw [program_projection,related]
      rfl
    obtain ⟨logical,physical⟩:=ih next nextRelated
    refine ⟨?_,physical⟩
    simp only [Grid.execute,executes,Option.bind_some]
    exact logical

#axiomsClean executes
#axiomsClean program_projection
#axiomsClean run_projection
end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.SourceBridge
