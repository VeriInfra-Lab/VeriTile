import bench.tritonbench_g.layer_norm_liger.Proofs.GridCells
import bench.tritonbench_g.layer_norm_liger.Proofs.ReductionPlan
/- The source launcher selects X.device through the standard CUDA device
context. Formal device/stream observations refer to that effective owning
device; leaving the context restores the ambient caller device. -/
namespace VeriTile.Bench.Repairs.LayerNormLiger.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum softwareSum)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000

structure Matrix where
  region : RegionName
  (rows columns stride : Nat)

structure Tensor where
  region : RegionName
  shape : List Nat
  dtype : FloatDType

/-- Scalar operations of generic Torch axis-sum and the explicit same-dtype
`.to(W.dtype)` call. No value equation for DW or DB is a premise. -/
structure Scalar (α : Type) where
  zero : α
  add : α → α → α
  castFP32 : α → α

def column {α : Type} [Inhabited α] (s : State α) (input : Matrix) (col : Nat) : Fin input.rows → α :=
  fun row => (s.mem input.region (row.val*input.stride+col)).read .fp32

def sumValue {α : Type} [Inhabited α] (ops : Scalar α) (input : Matrix)
    (plan : ReductionPlan.Plan input.rows) (s : State α) (col : Nat) : α :=
  ops.castFP32 (plan.tree.eval ops.zero ops.add (column s input col))

/-- Generic axis-0 sum/cast writes a fresh FP32 vector. Every tree leaf reads
one actual (PID,column) cell in the input tensor descriptor. -/
def axisSum {α : Type} [Inhabited α] (ops : Scalar α) (input : Matrix)
    (plan : ReductionPlan.Plan input.rows) (output : RegionName) (s : State α) : State α :=
  (TileShape.allIndices [input.columns]).foldl (fun t i =>
    if True then t.write output i.1.val (.mk .fp32 (sumValue ops input plan s i.1.val)) else t) s

theorem axisSum_read {α : Type} [Inhabited α] (ops : Scalar α) (input : Matrix)
    (plan : ReductionPlan.Plan input.rows) (output : RegionName) (s : State α)
    (i : Fin input.columns) :
    (axisSum ops input plan output s).mem output i.val=.mk .fp32 (sumValue ops input plan s i.val) := by
  unfold axisSum
  apply State.masked_scatter_readback (shape := [input.columns]) (active := fun _ => True)
    (i := (i,PUnit.unit))
  · trivial
  · intro a b hab
    rcases a with ⟨a,⟨⟩⟩
    rcases b with ⟨b,⟨⟩⟩
    simpa [Fin.ext_iff] using hab

theorem axisSum_frame {α : Type} [Inhabited α] (ops : Scalar α) (input : Matrix)
    (plan : ReductionPlan.Plan input.rows) (output : RegionName) (s : State α)
    (r : RegionName) (o : Nat) (h : r≠output) :
    (axisSum ops input plan output s).mem r o=s.mem r o := by
  unfold axisSum
  apply State.masked_scatter_frame
  exact Or.inl h

structure Return (α : Type) where
  state : State α
  (dx dw db : Tensor)

def partialDW (p : Backward.Parameters) (programs : Nat) : Matrix :=
  ⟨p.dw,programs,p.columns,p.dwStride⟩
def partialDB (p : Backward.Parameters) (programs : Nat) : Matrix :=
  ⟨p.db,programs,p.columns,p.dbStride⟩

def finish {α : Type} [Inhabited α] (ops : Scalar α) (p : Backward.Parameters)
    (programs : Nat) (pw pb : ReductionPlan.Plan programs)
    (dw db : RegionName) (shape : List Nat) (s : State α) : Return α where
  state := axisSum ops (partialDB p programs) pb db
    (axisSum ops (partialDW p programs) pw dw s)
  dx := ⟨p.dx,shape,.fp32⟩
  dw := ⟨dw,[p.columns],.fp32⟩
  db := ⟨db,[p.columns],.fp32⟩

theorem finish_dw {α : Type} [Inhabited α] (ops : Scalar α) (p : Backward.Parameters)
    (programs : Nat) (pw pb : ReductionPlan.Plan programs)
    (dw db : RegionName) (shape : List Nat) (s : State α) (hne : dw≠db)
    (i : Fin p.columns) :
    (finish ops p programs pw pb dw db shape s).state.mem dw i.val=
      .mk .fp32 (sumValue ops (partialDW p programs) pw s i.val) := by
  unfold finish
  rw [axisSum_frame _ _ _ _ _ _ _ hne]
  exact axisSum_read ops (partialDW p programs) pw dw s i

theorem finish_db {α : Type} [Inhabited α] (ops : Scalar α) (p : Backward.Parameters)
    (programs : Nat) (pw pb : ReductionPlan.Plan programs)
    (dw db : RegionName) (shape : List Nat) (s : State α) (hfresh : p.db≠dw)
    (i : Fin p.columns) :
    (finish ops p programs pw pb dw db shape s).state.mem db i.val=
      .mk .fp32 (sumValue ops (partialDB p programs) pb s i.val) := by
  unfold finish
  rw [axisSum_read]
  congr 1
  unfold sumValue
  congr 2
  funext row
  unfold column
  simp only [partialDB]
  rw [axisSum_frame _ _ _ _ _ _ _ hfresh]

theorem finish_frame {α : Type} [Inhabited α] (ops : Scalar α) (p : Backward.Parameters)
    (programs : Nat) (pw pb : ReductionPlan.Plan programs)
    (dw db : RegionName) (shape : List Nat) (s : State α) (r : RegionName) (o : Nat)
    (hw : r≠dw) (hb : r≠db) :
    (finish ops p programs pw pb dw db shape s).state.mem r o=s.mem r o := by
  unfold finish
  rw [axisSum_frame _ _ _ _ _ _ _ hb,axisSum_frame _ _ _ _ _ _ _ hw]

def backward {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (ops : Scalar α) (p : Backward.Parameters) (programs : Nat)
    (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (shape : List Nat)
    (s : State α) : Option (Return α) := do
  let t ← Grid.run (withSum M R) (Backward.kernel p) s programs
  pure (finish ops p programs pw pb dw db shape t)

/-- The actual complete grid is composed with actual axis-sum/cast memory
operations and the three returned tensor descriptors. -/
theorem backward_complete {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (ops : Scalar α) (p : Backward.Parameters) (programs : Nat)
    (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (shape : List Nat) (s : State α) :
    backward M R ops p programs pw pb dw db shape s=
      some (finish ops p programs pw pb dw db shape
        (Grid.reference (Backward.reference M R p) s programs)) := by
  simp [backward,Grid.backward_complete,Option.bind]

def scalarFP32 (cfg : FP.Config) : Scalar (FP.Value .fp32) :=
  ⟨FP.Value.zero .fp32,FP.add cfg,id⟩

theorem backward_bits [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (rsqrt : FP.Value .fp32 → FP.Value .fp32)
    (p : Backward.Parameters) (programs : Nat) (pw pb : ReductionPlan.Plan programs)
    (dw db : RegionName) (shape : List Nat) (s : State (FP.Value .fp32)) :
    backward (Forward.software rest cfg literal rsqrt) (softwareSum cfg) (scalarFP32 cfg)
      p programs pw pb dw db shape s=
      some (finish (scalarFP32 cfg) p programs pw pb dw db shape
        (Grid.reference (ClosedFP.backwardReference cfg literal p) s programs)) := by
  simp [backward,Grid.backward_bits,Option.bind]

/-- Tensor identity is literal state flow from forward's produced Mean/RSTD
to backward's loads. Ordinary saved-tensor/allocator contracts bind the same
regions; there is no independently supplied statistics correctness premise. -/
def forwardBackward {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (ops : Scalar α) (fp : Forward.Parameters) (bp : Backward.Parameters)
    (rows programs : Nat) (pw pb : ReductionPlan.Plan programs)
    (dw db : RegionName) (shape : List Nat) (s : State α) : Option (Return α) := do
  let forward ← Grid.run (withSum M R) (Forward.kernel fp) s rows
  backward M R ops bp programs pw pb dw db shape forward

theorem forwardBackward_complete {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (ops : Scalar α) (fp : Forward.Parameters) (bp : Backward.Parameters)
    (rows programs : Nat) (pw pb : ReductionPlan.Plan programs)
    (dw db : RegionName) (shape : List Nat) (s : State α) :
    forwardBackward M R ops fp bp rows programs pw pb dw db shape s=
      some (finish ops bp programs pw pb dw db shape
        (Grid.reference (Backward.reference M R bp)
          (Grid.reference (Forward.reference M R fp) s rows) programs)) := by
  simp [forwardBackward,Grid.forward_complete,backward_complete,Option.bind]

theorem forwardBackward_bits [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (literal : ℝ → FP.Value .fp32) (rsqrt : FP.Value .fp32 → FP.Value .fp32)
    (fp : Forward.Parameters) (bp : Backward.Parameters) (rows programs : Nat)
    (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (shape : List Nat)
    (s : State (FP.Value .fp32)) :
    forwardBackward (Forward.software rest cfg literal rsqrt) (softwareSum cfg) (scalarFP32 cfg)
      fp bp rows programs pw pb dw db shape s=
      some (finish (scalarFP32 cfg) bp programs pw pb dw db shape
        (Grid.reference (ClosedFP.backwardReference cfg literal bp)
          (Grid.reference (ClosedFP.forwardReference cfg literal rsqrt fp) s rows) programs)) := by
  simp [forwardBackward,Grid.forward_bits,backward_bits,Option.bind]

#axiomsClean axisSum_read
#axiomsClean axisSum_frame
#axiomsClean finish_dw
#axiomsClean finish_db
#axiomsClean finish_frame
#axiomsClean backward_complete
#axiomsClean backward_bits
#axiomsClean forwardBackward_complete
#axiomsClean forwardBackward_bits
end
end VeriTile.Bench.Repairs.LayerNormLiger.Host
