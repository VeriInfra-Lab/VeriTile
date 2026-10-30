import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPGrid
import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPInterComplete
import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPDiagonalComplete
import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPDiagonalReadback
import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPMergeComplete
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPStages
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open FPStorage (Memory state exportMemory)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000

inductive Stage where
  | inter (c : ProducerGeometry.Shape) (scale : ℝ)
  | intra (c : ProducerGeometry.Shape) (BK : Nat) (scale : ℝ)
  | split (c : ProducerGeometry.Shape) (scale : ℝ)
  | merge (c : ProducerGeometry.Shape)
  | output (c : OutputGrid.Shape) (scale : ℝ)

def diagonalLayout (c : ProducerGeometry.Shape) (BK : Nat) : FPDiagonal.Layout :=
  ⟨c.T,c.K,c.BT,16,BK,c.NC⟩
def mergeLayout (c : ProducerGeometry.Shape) : FPMerge.Layout :=
  ⟨c.T,c.BT,16,Geometry.ceilDiv c.K c.BK⟩
def Stage.grid : Stage → Grid
  | .inter c _ => ProducerGeometry.grid c
  | .intra c _ _ => DiagonalGeometry.grid c
  | .split c _ => SplitGeometry.grid c
  | .merge c => DiagonalGeometry.grid c
  | .output c _ => OutputGrid.grid c
def Stage.kernel : Stage → ComputeKernel
  | .inter c scale => FPInter.kernel c scale
  | .intra c BK scale => FPDiagonal.kernel Bool.false (diagonalLayout c BK) scale
  | .split c scale => FPDiagonal.kernel Bool.true (diagonalLayout c c.BK) scale
  | .merge c => FPMerge.kernel (mergeLayout c)
  | .output c scale => FPOutput.kernel c scale
def Stage.region : Stage → RegionName
  | .split _ _ => "scratch"
  | .output _ _ => "o"
  | _ => "attention"
def Stage.algebra (M : Algebra α) (D : FPOutput.DotPrimitive α) : Stage → Algebra α
  | .inter _ _ | .output _ _ => FPOutput.withDot M D
  | _ => M

def program (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage) (s : State α) : State α :=
  match stage with
  | .inter c scale => if FPInter.enabled c s then FPInter.written M D c scale s else s
  | .intra c BK scale => if FPDiagonal.enabled Bool.false (diagonalLayout c BK) s then
      FPDiagonal.written M Bool.false (diagonalLayout c BK) scale s else s
  | .split c scale => if FPDiagonal.enabled Bool.true (diagonalLayout c c.BK) s then
      FPDiagonal.written M Bool.true (diagonalLayout c c.BK) scale s else s
  | .merge c => if FPMerge.enabled (mergeLayout c) s then FPMerge.written M (mergeLayout c) s else s
  | .output c scale => FPOutput.written M D c scale s

theorem program_correct (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage) (s : State α) :
    ∃ t, FP.Structural.exec (engine (stage.algebra M D)) stage.kernel s=some t ∧
      t.mem=(program M D stage s).mem := by
  cases stage with
  | inter c scale => exact FPInter.source_memory M D c scale s
  | intra c BK scale => exact FPDiagonal.source_memory M Bool.false (diagonalLayout c BK) scale s
  | split c scale => exact FPDiagonal.source_memory M Bool.true (diagonalLayout c c.BK) scale s
  | merge c => exact FPMerge.source_memory M (mergeLayout c) s
  | output c scale => exact FPOutput.source_memory M D c scale s

def referenceStep (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage)
    (idx : GridIndex stage.grid) (m : Memory α) : Memory α :=
  exportMemory (program M D stage (state m idx.toPids stage.grid.toNumPids))
def reference (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage) (m : Memory α) : Memory α :=
  FPGrid.referenceJobs (referenceStep M D stage) (FPGrid.jobs stage.grid) m
def execute (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage) (m : Memory α) : Option (Memory α) :=
  FPGrid.runJobs (stage.algebra M D) stage.kernel stage.grid (FPGrid.jobs stage.grid) m

theorem invoke_eq (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage)
    (idx : GridIndex stage.grid) (m : Memory α) :
    FPGrid.invoke (stage.algebra M D) stage.kernel stage.grid idx m=some (referenceStep M D stage idx m) := by
  obtain ⟨t,ht,hm⟩ := program_correct M D stage (state m idx.toPids stage.grid.toNumPids)
  unfold FPGrid.invoke
  rw [ht]
  exact congrArg some (FPStorage.export_eq hm)

theorem execute_eq (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage) (m : Memory α) :
    execute M D stage m=some (reference M D stage m) :=
  FPGrid.runJobs_eq _ _ _ _ (invoke_eq M D stage) _ _

theorem program_frame (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage) (s : State α)
    (r : RegionName) (off : Nat) (hr : r≠stage.region) :
    (program M D stage s).mem r off=s.mem r off := by
  cases stage with
  | inter c scale =>
    simp only [program]
    split
    · exact FPInter.frame M D c scale s r off (Or.inl hr)
    · rfl
  | intra c BK scale =>
    simp only [program]
    split
    · exact FPDiagonal.frame M Bool.false (diagonalLayout c BK) scale s r off (Or.inl hr)
    · rfl
  | split c scale =>
    simp only [program]
    split
    · exact FPDiagonal.frame M Bool.true (diagonalLayout c c.BK) scale s r off (Or.inl hr)
    · rfl
  | merge c =>
    simp only [program]
    split
    · exact FPMerge.frame M (mergeLayout c) s r off (Or.inl hr)
    · rfl
  | output c scale => exact FPOutput.frame M D c scale s r off (Or.inl hr)

theorem referenceStep_frame (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage)
    (idx : GridIndex stage.grid) (m : Memory α) (r : RegionName) (off : Nat) (hr : r≠stage.region) :
    referenceStep M D stage idx m r off=m r off := by
  unfold referenceStep exportMemory
  rw [program_frame M D stage _ r off hr]
  exact FPStorage.payload_cell _ _

theorem reference_frame (M : Algebra α) (D : FPOutput.DotPrimitive α) (stage : Stage)
    (m : Memory α) (r : RegionName) (off : Nat) (hr : r≠stage.region) :
    reference M D stage m r off=m r off :=
  FPGrid.reference_frame _ _ _ _ _ (fun idx _ mm => referenceStep_frame M D stage idx mm r off hr)

#specNonCircular reference avoiding [execute,FPGrid.runJobs,FPGrid.invoke,FP.Structural.exec]
#axiomsClean execute_eq
#axiomsClean reference_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPStages
