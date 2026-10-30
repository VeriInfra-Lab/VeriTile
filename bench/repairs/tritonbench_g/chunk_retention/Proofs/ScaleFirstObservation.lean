import bench.repairs.tritonbench_g.chunk_retention.Proofs.ScaleFirstSafety
import VeriTile.Triton.Float.ScalarOpsLaws
import bench.repairs.tritonbench_g.chunk_retention.KernelOnly.IO

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.ScaleFirstSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel KernelStaging ForwardOSource
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000

theorem value_observed (ops : Ops α) (literal : ℝ → α)
    (comm : ∀ x y, ops.mul x y=ops.mul y x)
    (c : Shape) (scale : ℝ) (s : State (Term α)) (i : Index c) :
    lower ops literal (ForwardOSource.value (algebra ops.zero) dot c scale s i)=
      lower ops literal (value (algebra ops.zero) dot c scale s i) := by
  change ops.cast (ops.mul _ (literal scale))=ops.cast (ops.mul (literal scale) _)
  rw [comm]

/-- The complete source runs have the same observed cells; outside their actual
output stores they retain the identical raw caller cells. The live-dot seed and
all ordered FMA chains occur under the unchanged addition subtree. -/
theorem comparison (ops : Ops α) (literal : ℝ → α)
    (comm : ∀ x y, ops.mul x y=ops.mul y x)
    (c : Shape) (scale : ℝ) (s left : State (Term α))
    (ran : FP.Structural.exec (engine (withDot (algebra ops.zero) dot))
      (ForwardOSource.kernel c scale) s=some left) :
    ∃ right, FP.Structural.exec (engine (withDot (algebra ops.zero) dot)) (kernel c scale) s=some right ∧
      (∀ (r : RegionName) (off : Nat), mapCell (lower ops literal) (left.mem r off)=mapCell (lower ops literal) (right.mem r off)) ∧
      (∀ (r : RegionName) (off : Nat), (r≠"o" ∨ ∀ i, active c s i → off≠outputAddress c s i) → right.mem r off=s.mem r off) := by
  classical
  obtain ⟨old,oldRan,oldMem⟩:=ForwardOSource.source_memory (algebra ops.zero) dot c scale s
  have heq : old=left := Option.some.inj (oldRan.symm.trans ran)
  subst old
  obtain ⟨right,rightRan,rightMem⟩:=source_memory (algebra ops.zero) dot c scale s
  refine ⟨right,rightRan,?_,?_⟩
  · intro r off
    rw [oldMem,rightMem]
    by_cases h : r="o" ∧ ∃ i : Index c, active c s i ∧ off=outputAddress c s i
    · obtain ⟨rfl,i,hi,rfl⟩:=h
      rw [ForwardOSource.readback (algebra ops.zero) dot c scale s i hi (address_injective c s),
        readback (algebra ops.zero) dot c scale s i hi (address_injective c s)]
      exact congrArg (Cell.mk .real) (value_observed ops literal comm c scale s i)
    · have exterior : r≠"o" ∨ ∀ i, active c s i → off≠outputAddress c s i := by
        by_cases hr : r="o"
        · right; intro i hi eq; exact h ⟨hr,i,hi,eq⟩
        · exact Or.inl hr
      rw [ForwardOSource.frame (algebra ops.zero) dot c scale s r off exterior,
        frame (algebra ops.zero) dot c scale s r off exterior]
  · intro r off outside
    rw [rightMem]
    exact frame (algebra ops.zero) dot c scale s r off outside

/-- All-word strict multiplication certificate for the selected software model.
Native scalar ports retain the original explicit model-conformance boundary. -/
theorem native_mul_comm (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (x y : BitVec 32) :
    (wordOps (Numeric.nativeISA ports)).mul x y=(wordOps (Numeric.nativeISA ports)).mul y x := by
  rw [Numeric.isa_eq u ports laws]
  change Numeric.mul32 x y=Numeric.mul32 y x
  unfold Numeric.mul32
  rw [FP.ScalarOpsLaws.mul_comm]

#axiomsClean value_observed
#axiomsClean comparison
#axiomsClean native_mul_comm
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.ScaleFirstSource
