import bench.repairs.tritonbench_g.int8_quantization.Kernels
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.Repairs.Int8Quantization.Model
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

variable {α : Type} [Inhabited α]

/-- Availability of the exact scalar comparison/conversion primitives, not
an execution, output, memory or host law. Numeric meaning is kept separate. -/
structure Primitives (α : Type) where
  less : α → α → Bool
  le : α → α → Bool
  int8 : α → Int

def backend (M : Algebra α) (ops : Primitives α) : Algebra α :=
  { M with compareLt := fun _ _=> some ops.less
           compareLe := fun _ _=> some ops.le
           floatToInt := fun _ _ _=> some (fun x=> some (ops.int8 x)) }

/-- A constructed successful execution of the complete source. This does not
yet connect readback with the independent target or native memory. -/
def execution (M : Algebra α) (ops : Primitives α) (p : Parameters)
    (positive : 0< p.width) (s : State α) :
    {t : State α // FP.Structural.exec (backend M ops) (source p) s=some t} := by
  cases query : p.query
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp (config := {maxSteps := 1000000}) [source,FP.Structural.exec,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      backend,Option.bind,numericLt,numericLe,numericEq,natLt,natLe,
      bop,numeric,ofFloat,toFloat,Region.cast,ComputeDType.eraseDType,
      positive,query,TileShape.axisDim,TileShape.discreteMaximum,
      FP.Structural.store,integralMod,integralFloorDiv]
    try rfl

def group (p : Parameters) (s : State α) : Nat := s.pids 0+p.base
def offset (p : Parameters) (s : State α) (i : TileIndex [p.width]) : Nat :=
  group p s/p.blocks*p.length*p.channels+group p s%p.blocks*p.blockRows*p.channels+i.1.val
def active (p : Parameters) (s : State α) (i : TileIndex [p.width]) : Prop :=
  i.1.val< p.blockRows*p.channels ∧ group p s%p.blocks*p.blockRows+i.1.val/p.channels< p.length
instance (p : Parameters) (s : State α) (i : TileIndex [p.width]) : Decidable (active p s i) :=
  inferInstanceAs (Decidable (_ ∧ _))

def quantized (p : Parameters) (s : State α) (i : TileIndex [p.width]) : Int :=
  (s.regs .int [p.width] "quantized").getD (fun _=>0) i
def scaleValue (M : Algebra α) (s : State α) : α :=
  if (s.regs .int [] "bad").getD (fun _=>0) PUnit.unit=0 then
    (s.regs .real [] "scale").getD (fun _=> default) PUnit.unit
  else M.literal (some .fp32) .real 0

def stored (M : Algebra α) (p : Parameters) (entry registers : State α) : State α :=
  ((TileShape.allIndices [p.width]).foldl (fun t i=>
    if active p entry i then t.write outRegion (offset p entry i) (.mk .int (quantized p registers i)) else t) entry).write
      scaleRegion (group p entry) (.mk .real (scaleValue M registers))

/-- Readback of the explicit two-store memory model. The complete source-to-model bridge is proved by SourceExecution;
this helper itself does not assume it. -/
theorem stored_quantized (M : Algebra α) (p : Parameters) (entry registers : State α)
    (i : TileIndex [p.width]) (hi : active p entry i) :
    (stored M p entry registers).mem outRegion (offset p entry i)=
      .mk .int (quantized p registers i) := by
  unfold stored
  rw [State.write_other _ _ _ _ _ _ (Or.inl (by decide : outRegion≠scaleRegion))]
  apply masked_readback_active _ _ _ _ _ i hi
  intro a b _ _ equality
  apply Prod.ext _ (Subsingleton.elim _ _)
  apply Fin.ext
  exact Nat.add_left_cancel equality

theorem stored_scale (M : Algebra α) (p : Parameters) (entry registers : State α) :
    (stored M p entry registers).mem scaleRegion (group p entry)=.mk .real (scaleValue M registers) :=
  State.write_same _ _ _ _

theorem stored_frame (M : Algebra α) (p : Parameters) (entry registers : State α)
    (r : RegionName) (o : Nat)
    (notScale : r≠scaleRegion ∨ o≠group p entry)
    (notOutput : r≠outRegion ∨ ∀ i,active p entry i → o≠offset p entry i) :
    (stored M p entry registers).mem r o=entry.mem r o := by
  unfold stored
  rw [State.write_other _ _ _ _ _ _ notScale]
  exact State.masked_scatter_frame _ _ _ _ _ r o
    (notOutput.imp id (fun h i _ hi=> h i hi)) entry

#axiomsClean stored_quantized
#axiomsClean stored_scale
#axiomsClean stored_frame

end
end VeriTile.Bench.Repairs.Int8Quantization.Model
