import bench.repairs.tritonbench_g.square_matrix.Proofs.Memory
import bench.repairs.tritonbench_g.square_matrix.Proofs.FP
import bench.tritonbench_g.square_matrix.Proofs.Source

namespace VeriTile.Bench.Repairs.SquareMatrix.SourceBinding
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.AddExample.Finite
open Native
noncomputable section
set_option maxHeartbeats 6000000

/-- The source quotation retains FP32 product/store and the original nine
statements; no multiplication formula is supplied by an execution hypothesis. -/
theorem precision (n B : Nat) :
    (fp32Kernel n B).surfaceBody[5]? = some (VeriTile.Bench.TritonBenchG.SquareMatrix.expectedProduct .fp32 B) ∧
    (fp32Kernel n B).surfaceBody.getLast? = some (VeriTile.Bench.TritonBenchG.SquareMatrix.expectedStore .fp32 n B) :=
  ⟨VeriTile.Bench.TritonBenchG.SquareMatrix.multiplication_precision .fp32 Bool.false _ _ _ _ _ _,
   VeriTile.Bench.TritonBenchG.SquareMatrix.store_precision .fp32 Bool.false _ _ _ _ _ _⟩

/-- Active output observations of the pointer-bound row interpreter equal the
actual complete typed DSL execution on the same input row. Inactive lanes are
masked in both interpretations, and the physical frame is independently proved. -/
theorem row_refines {M A : Type} [Inhabited A] (ops : MemoryPrimitives M A) (algebra : Algebra A)
    (x : TensorMetadata) (h : InputContract x) (L : Layout x) (memory : LocalMemory ops L)
    (r : Nat) (hr : r<rows x) (nonempty : 0<x.numel) (before : M) (s : State A)
    (pid : s.pids 0=r)
    (input : ∀ i,i<x.numel → (s.mem "x" i).read .real=ops.read before .fp32 (address L.input i)) :
    ∃ t, FP.Structural.exec (FPProof.backend algebra) (fp32Kernel (cols x) (block (cols x))) s=some t ∧
      (∀ c,c<cols x →
        (t.mem "out" (r*cols x+c)).read .real=
          ops.read (sourceRow ops (algebra.binary (some .fp32) .real .mul) L r before)
            .fp32 (address L.output (r*cols x+c))) ∧
      Preserved ops (view L.output)
        (sourceRow ops (algebra.binary (some .fp32) .real .mul) L r before) before := by
  obtain ⟨t,he,hm⟩ := FPProof.complete_correct algebra (cols x) (block (cols x)) s
  refine ⟨t,he,?_,row_frame _ _ x h L r hr nonempty before⟩
  intro c hc
  have hi : r*cols x+c<x.numel := by rw [numel_eq x h]; exact flat_lt _ _ _ _ hr hc
  have member : r*cols x+c∈indices (cols x) r := (indices_mem _ _ _).mpr ⟨by omega,by omega⟩
  rw [hm,FPProof.expected_cell _ _ _ _ (block_covers _)]
  simp only [pid,FPProof.resultCell,FPProof.reference]
  rw [source_row _ _ x h L r hr nonempty before,writes_read ops x L memory,if_pos member]
  unfold result
  rw [input _ hi]
  simp [hc]


/-- Registers begin unbound, and the two region readers use the actual physical
prestate of this row. This is an observation map, not a fabricated input copy. -/
def sourceState {Memory A : Type} [Inhabited A] (ops : MemoryPrimitives Memory A)
    (x : TensorMetadata) (L : Layout x) (row : Nat) (before : Memory) : State A where
  mem := fun r i => .mk .real (ops.read before .fp32
    (if r="x" then address L.input i else address L.output i))
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then row else 0
  numPids := fun ax => if ax=0 then rows x else 1
  undef := fun d _ _ => defaultValue d

/-- Every row witness executes the full typed quotation from that exact
physical prestate; the next witness starts at the preceding native result. -/
inductive SourceChain {Memory A : Type} [Inhabited A] (ops : MemoryPrimitives Memory A)
    (algebra : Algebra A) (x : TensorMetadata) (L : Layout x) :
    List Nat → Memory → Memory → Prop where
  | nil (s : Memory) : SourceChain ops algebra x L [] s s
  | cons (row : Nat) (rest : List Nat) (before final : Memory) (t : State A)
      (execution : FP.Structural.exec (FPProof.backend algebra)
        (fp32Kernel (cols x) (block (cols x))) (sourceState ops x L row before)=some t)
      (readback : ∀ c,c<cols x → (t.mem "out" (row*cols x+c)).read .real=
        ops.read (sourceRow ops (algebra.binary (some .fp32) .real .mul) L row before)
          .fp32 (address L.output (row*cols x+c)))
      (tail : SourceChain ops algebra x L rest
        (sourceRow ops (algebra.binary (some .fp32) .real .mul) L row before) final) :
      SourceChain ops algebra x L (row::rest) before final

theorem source_chain {Memory A : Type} [Inhabited A] (ops : MemoryPrimitives Memory A)
    (algebra : Algebra A) (x : TensorMetadata) (h : InputContract x)
    (L : Layout x) (memory : LocalMemory ops L) (schedule : List Nat)
    (valid : ∀ row∈schedule,row<rows x) (nonempty : 0<x.numel) (before : Memory) :
    SourceChain ops algebra x L schedule before
      (Native.runRows ops (algebra.binary (some .fp32) .real .mul) L schedule before) := by
  induction schedule generalizing before with
  | nil => exact .nil before
  | cons row rest ih =>
    obtain ⟨t,he,hv,_⟩ := row_refines ops algebra x h L memory row (valid row (by simp))
      nonempty before (sourceState ops x L row before) (by simp [sourceState])
      (by intro i _; simp [sourceState])
    exact .cons row rest before _ t he hv
      (ih (fun r hr => valid r (by simp [hr])) _)

#axiomsClean source_chain

#axiomsClean precision
#axiomsClean row_refines
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SquareMatrix.SourceBinding
