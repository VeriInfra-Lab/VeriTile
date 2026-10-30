import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeFinite
import VeriTile.Meta.Specification

namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholePublic
open VeriTile VeriTile.Triton FP.Structural WholeSource WholeGrid WholeHost WholeFinite
noncomputable section
set_option maxHeartbeats 6000000

variable {α : Type} {x : Inputs} {entry : Nat → α} {preexisting : Nat → Prop}

/-- Exhaustive original host outcomes on the FP32 rank2/rank1 CUDA branch.
The dimension assertion precedes allocation. Runtime failures are observable;
a successful setup cannot encounter a source-model execution failure. -/
def Outcomes (formula : Allocated α x entry preexisting → Fin x.rows → α)
    (rt : Runtime α x entry preexisting) : HostOutcome α → Prop
  | .incompatibleDimensions => x.columns≠x.vectorLength
  | .allocationFailure => x.columns=x.vectorLength ∧ rt.allocate=none
  | .setupFailure => x.columns=x.vectorLength ∧
      ∃ a,rt.allocate=some a ∧ rt.launchReady a=Bool.false
  | .sourceFailure => False
  | .returned out => x.columns=x.vectorLength ∧
      ∃ a,rt.allocate=some a ∧ rt.launchReady a=Bool.true ∧
        out.length=x.rows ∧ out.dtype=.fp32 ∧ out.device=x.device ∧ out.stride=1 ∧ out.base=a.base ∧
        (∀ i : Fin x.rows,out.memory (out.base+i.val)=formula a i) ∧
        (∀ addr,(∀ i,i<x.rows → addr≠out.base+i) → out.memory addr=a.memory addr) ∧
        (∀ addr,preexisting addr → out.memory addr=entry addr)

/-- Public finite-address correspondence covers every legal autotune config,
all padded lanes and final pointer/backedge increments, exact source masks,
active allocation bounds, and byte-level output separation from old memory. -/
def IndexCorrespondence (a : Allocated α x entry preexisting) : Prop :=
  (∀ c pid step lane col,pid<gridSize (x.parameters c) → step ≤ chunks x c →
    lane<c.blockRows → col<c.blockColumns →
    (matrixPointer x c pid lane col step).toNat=
      4*(x.matrixBase+(pid*c.blockRows+lane)*x.strideRow+col*x.strideColumn+step*c.blockColumns*x.strideColumn) ∧
    (vectorPointer x c col step).toNat=4*(x.vectorBase+col*x.strideVector+step*c.blockColumns*x.strideVector) ∧
    (bytePointer a.base (pid*c.blockRows+lane)).toNat=4*(a.base+(pid*c.blockRows+lane))) ∧
  (∀ c pid step lane col,pid<gridSize (x.parameters c) → step<chunks x c →
    lane<c.blockRows → col<c.blockColumns →
    (rowBits c pid lane).toInt=((pid*c.blockRows+lane:Nat):Int) ∧
    (columnBits c step col).toInt=((step*c.blockColumns+col:Nat):Int) ∧
    (productBits (pid*c.blockRows+lane) x.strideRow).toInt=(((pid*c.blockRows+lane)*x.strideRow:Nat):Int) ∧
    (productBits col x.strideColumn).toInt=((col*x.strideColumn:Nat):Int) ∧
    (productBits col x.strideVector).toInt=((col*x.strideVector:Nat):Int) ∧
    (productBits c.blockColumns x.strideColumn).toInt=((c.blockColumns*x.strideColumn:Nat):Int) ∧
    (productBits c.blockColumns x.strideVector).toInt=((c.blockColumns*x.strideVector:Nat):Int) ∧
    (columnBits c step c.blockColumns).toInt=((step*c.blockColumns+c.blockColumns:Nat):Int) ∧
    ((rowBits c pid lane).slt (i32 x.rows)=Bool.true ↔ pid*c.blockRows+lane<x.rows) ∧
    ((columnBits c step col).slt (i32 x.columns)=Bool.true ↔ step*c.blockColumns+col<x.columns)) ∧
  (x.columns=x.vectorLength → ∀ i j,i<x.rows → j<x.columns →
    4*(x.matrixAddress i j)+4 ≤ 2^64 ∧ 4*(x.vectorAddress j)+4 ≤ 2^64 ∧
    4*(a.base+i)+4 ≤ 2^64 ∧ preexisting (x.matrixAddress i j) ∧ preexisting (x.vectorAddress j) ∧
    4 ∣ 4*(x.matrixAddress i j) ∧ 4 ∣ 4*(x.vectorAddress j) ∧ 4 ∣ 4*(a.base+i)) ∧
  (∀ i addr b₁ b₂,i<x.rows → preexisting addr → b₁<4 → b₂<4 → 4*(a.base+i)+b₁≠4*addr+b₂)

theorem index_correspondence (a : Allocated α x entry preexisting)
    (f : FiniteAdmission x entry preexisting a) : IndexCorrespondence a :=
  ⟨source_pointer_correspondence a f,source_integer_correspondence a f,
    active_accesses a f,output_bytes_separate a⟩


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholePublic
