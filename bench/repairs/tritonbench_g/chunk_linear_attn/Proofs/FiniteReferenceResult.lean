import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.FiniteReferenceTables

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.FiniteReadLocality
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

def Cells (r : RegionName) (size : Nat) (a b : Memory α) : Prop :=
  ∀ off, off<size → a r off=b r off

theorem table_preserves_cells {ι : Type} (region : ι → RegionName) (index : ι → Nat)
    (left right : ι → Cell α) (a b : Memory α) (equal : ∀ i, left i=right i)
    (r : RegionName) (size : Nat) (same : Cells r size a b) :
    Cells r size (KernelReference.table region index left a) (KernelReference.table region index right b) := by
  intro off bound
  unfold KernelReference.table
  split_ifs with present
  · exact equal _
  · exact same off bound

theorem table_covers_cells {ι : Type} (region : ι → RegionName) (index : ι → Nat)
    (left right : ι → Cell α) (a b : Memory α) (equal : ∀ i, left i=right i)
    (r : RegionName) (size : Nat) (covers : ∀ off, off<size → ∃ i,region i=r ∧ index i=off) :
    Cells r size (KernelReference.table region index left a) (KernelReference.table region index right b) := by
  intro off bound
  unfold KernelReference.table
  rw [dif_pos (covers off bound),dif_pos (covers off bound),equal]

structure InputsAgree (d : Host.Dimensions) (seed terminal : Bool) (a b : Memory α) : Prop where
  query : Reads "query" (d.B*d.H*d.T*d.K) a b
  key : Reads "key" (d.B*d.H*d.T*d.K) a b
  value : Reads "value" (d.B*d.H*d.T*d.V) a b
  incoming : Reads "incoming" (d.B*d.H*d.T*d.V) a b
  initial : seed=Bool.true → Reads "initial" (d.B*d.H*d.K*d.V) a b
  terminalGradient : terminal=Bool.true → Reads "terminalGradient" (d.B*d.H*d.K*d.V) a b

/-- All written channels, including intermediates, agree as exact cells.
The premises mention only finite decoded reads of the six source inputs. -/
structure OutputsAgree (d : Host.Dimensions) (seed final : Bool) (a b : Memory α) : Prop where
  output : Cells "output" (d.B*d.H*d.T*d.V) a b
  queryGradient : Cells "queryGradient" (d.B*d.H*d.T*d.K) a b
  keyGradient : Cells "keyGradient" (d.B*d.H*d.T*d.K) a b
  valueGradient : Cells "valueGradient" (d.B*d.H*d.T*d.V) a b
  finalState : final=Bool.true → Cells "final" (d.B*d.H*d.K*d.V) a b
  initialGradient : seed=Bool.true → Cells "initialGradient" (d.B*d.H*d.K*d.V) a b
  history : Cells "history" (d.B*d.H*d.NT*d.K*d.V) a b
  historyGradient : Cells "historyGradient" (d.B*d.H*d.NT*d.K*d.V) a b
  valuePartials : Cells "valuePartials" (keyBlocks (d.parameters Bool.true)*(d.B*d.H)*d.T*d.V) a b

theorem result_agrees (M : Algebra α) (D : DotPrimitive α) (d : Host.Dimensions)
    (valid : Host.Valid d) (scale : ℝ) (seed final terminal : Bool)
    (selection : LibrarySelection (d.parameters Bool.true)) (a b : Memory α)
    (same : InputsAgree d seed terminal a b) :
    OutputsAgree d seed final
      (KernelReference.result M D d valid scale seed final terminal selection a)
      (KernelReference.result M D d valid scale seed final terminal selection b) := by
  let f := d.parameters Bool.false
  let g := d.parameters Bool.true
  let fs := Host.shape_valid d valid Bool.false
  let gs := Host.shape_valid d valid Bool.true
  let fa := KernelReference.state M D f fs scale Bool.false seed final a
  let fb := KernelReference.state M D f fs scale Bool.false seed final b
  have freads : StateReads f Bool.false seed a b := ⟨same.key,same.value,same.initial⟩
  have fv := state_value M D f fs scale Bool.false seed final a b freads
  have fp := state_preserves M D f fs scale Bool.false seed final a b freads
  have fh := state_history M D f fs scale Bool.false seed final a b freads
  have oreads : OutputReads f fa fb :=
    ⟨fp _ _ same.query,fp _ _ same.key,fp _ _ same.value,fh⟩
  let oa := KernelReference.output M D f fs scale fa
  let ob := KernelReference.output M D f fs scale fb
  have ov := output_value M D f fs scale fa fb oreads
  have op := output_preserves M D f fs scale fa fb oreads
  have oc := table_preserves_cells (fun _ : KernelReference.OutputCell f => "output")
    KernelReference.outputIndex _ _ fa fb ov
  have binput : StateReads g Bool.true terminal oa ob :=
    ⟨op _ _ (fp _ _ same.query),op _ _ (fp _ _ same.incoming),
      fun h => op _ _ (fp _ _ (same.terminalGradient h))⟩
  let ba := KernelReference.state M D g gs scale Bool.true terminal seed oa
  let bb := KernelReference.state M D g gs scale Bool.true terminal seed ob
  have bv := state_value M D g gs scale Bool.true terminal seed oa ob binput
  have bp := state_preserves M D g gs scale Bool.true terminal seed oa ob binput
  have bc := table_preserves_cells (KernelReference.stateRegion Bool.true)
    KernelReference.stateIndex _ _ oa ob bv
  have bh := state_history M D g gs scale Bool.true terminal seed oa ob binput
  have ginput : GradientReads g ba bb :=
    ⟨bp _ _ (op _ _ (fp _ _ same.query)),bp _ _ (op _ _ (fp _ _ same.key)),
      bp _ _ (op _ _ (fp _ _ same.value)),bp _ _ (op _ _ (fp _ _ same.incoming)),
      bp _ _ (op _ _ fh),bh⟩
  let ga := KernelReference.gradient M D g gs scale ba
  let gb := KernelReference.gradient M D g gs scale bb
  have gv := gradient_value_table M D g gs scale ba bb ginput
  have gc := table_preserves_cells KernelReference.gradientRegion KernelReference.gradientIndex _ _ ba bb gv
  have partials := gradient_partial M D g gs scale ba bb ginput
  have sv := fun i => congrArg (Cell.mk .real) (sum_value M g selection ga gb partials i)
  have sc := table_preserves_cells (fun _ : Fin (valueCount g) => "valueGradient") Fin.val _ _ ga gb sv
  constructor
  · exact sc _ _ (gc _ _ (bc _ _ (table_covers_cells _ _ _ _ fa fb ov _ _ (output_coverage f))))
  · exact sc _ _ (table_covers_cells _ _ _ _ ba bb gv _ _ (gradient_query_coverage g Bool.false))
  · exact sc _ _ (table_covers_cells _ _ _ _ ba bb gv _ _ (gradient_query_coverage g Bool.true))
  · exact table_covers_cells _ _ _ _ ga gb sv _ _ (fun off bound => ⟨⟨off,bound⟩,rfl,rfl⟩)
  · intro enabled
    exact sc _ _ (gc _ _ (bc _ _ (oc _ _ (table_covers_cells _ _ _ _ a b fv _ _
      (endpoint_coverage f Bool.false final enabled)))))
  · intro enabled
    exact sc _ _ (gc _ _ (table_covers_cells _ _ _ _ oa ob bv _ _
      (endpoint_coverage g Bool.true seed enabled)))
  · exact sc _ _ (gc _ _ (bc _ _ (oc _ _ (table_covers_cells _ _ _ _ a b fv _ _
      (history_coverage f Bool.false final)))))
  · exact sc _ _ (gc _ _ (table_covers_cells _ _ _ _ oa ob bv _ _
      (history_coverage g Bool.true seed)))
  · exact sc _ _ (table_covers_cells _ _ _ _ ba bb gv _ _ (gradient_partial_coverage g))

def RealCells (m : Memory α) : Prop := ∀ r off,m r off=Cell.mk .real ((m r off).read .real)

theorem table_real {ι : Type} (region : ι → RegionName) (index : ι → Nat)
    (value : ι → Cell α) (before : Memory α)
    (typed : ∀ i,value i=Cell.mk .real ((value i).read .real)) (initial : RealCells before) :
    RealCells (KernelReference.table region index value before) := by
  intro r off
  unfold KernelReference.table
  split_ifs with present
  · exact typed _
  · exact initial r off

theorem result_real (M : Algebra α) (D : DotPrimitive α) (d : Host.Dimensions)
    (valid : Host.Valid d) (scale : ℝ) (seed final terminal : Bool)
    (selection : LibrarySelection (d.parameters Bool.true)) (before : Memory α)
    (initial : RealCells before) :
    RealCells (KernelReference.result M D d valid scale seed final terminal selection before) := by
  have stateReal (p : Parameters) (shape : Geometry.ShapeValid p) (rev seed final : Bool)
      (m : Memory α) (hm : RealCells m) : RealCells (KernelReference.state M D p shape scale rev seed final m) := by
    apply table_real
    · intro i; cases i <;> rfl
    · exact hm
  have outputReal (p : Parameters) (shape : Geometry.ShapeValid p)
      (m : Memory α) (hm : RealCells m) : RealCells (KernelReference.output M D p shape scale m) :=
    table_real _ _ _ _ (fun _ => rfl) hm
  have gradientReal (p : Parameters) (shape : Geometry.ShapeValid p)
      (m : Memory α) (hm : RealCells m) : RealCells (KernelReference.gradient M D p shape scale m) := by
    apply table_real
    · intro i; cases i <;> rfl
    · exact hm
  unfold KernelReference.result
  apply table_real
  · intro i; rfl
  · exact gradientReal _ _ _ (stateReal _ _ _ _ _ _ (outputReal _ _ _ (stateReal _ _ _ _ _ _ initial)))

#axiomsClean result_agrees
#axiomsClean result_real
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.FiniteReadLocality
