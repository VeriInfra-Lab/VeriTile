import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Returned
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardSaved
namespace VeriTile.Bench.Repairs.LayerNormLiger.Public
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (withSum)
open Logical
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000

/-- Ordinary wrapper shape, stride, launch and allocation requirements. -/
structure Layout (p : Backward.Parameters) (programs : Nat) (dw db : RegionName)
    (shape : List Nat) : Prop where
  programs_pos : 0<programs
  rows_pos : 0<p.rows
  columns_pos : 0<p.columns
  columns_bound : p.columns≤65536
  block : p.columns≤p.block
  dx_stride : p.dxStride=p.columns
  dw_stride : p.dwStride=p.columns
  db_stride : p.dbStride=p.columns
  group : p.rowsPerProgram=Launch.rowsPerProgram p.rows programs
  shape_elements : shape.prod=p.rows*p.columns
  inputs_fresh : Returned.Fresh p
  partial_dw_db : p.dw≠p.db
  partial_dx_dw : p.dx≠p.dw
  partial_dx_db : p.dx≠p.db
  return_dw_db : dw≠db
  return_dw_partial_db : p.db≠dw
  return_dw_dx : p.dx≠dw
  return_db_dx : p.dx≠db

/-- Full returned tensor values against independent row-gradient formulas.
No executor, kernel, or scalar reference appears in this specification. -/
def Correct (p : Backward.Parameters) (s : State ℝ) (programs : Nat)
    (dw db : RegionName) (shape : List Nat) (out : Host.Return ℝ) : Prop :=
  out.dx=⟨p.dx,shape,.fp32⟩ ∧ out.dw=⟨dw,[p.columns],.fp32⟩ ∧ out.db=⟨db,[p.columns],.fp32⟩ ∧
  (∀ row, row<p.rows → ∀ i : TileIndex [p.block], i.1.val<p.columns →
    out.state.mem p.dx (row*p.dxStride+i.1.val)=.mk .fp32 (rowDX p s row i.1.val)) ∧
  (∀ i : Fin p.columns, out.state.mem dw i.val=.mk .fp32
    (Logical.dw p.rows (fun row => bx p (saved p s) row) (fun row => bdy p (saved p s) row)
      (fun row => Backward.mean p (saved p s) row) (fun row => Backward.rstd p (saved p s) row) i.val)) ∧
  (∀ i : Fin p.columns, out.state.mem db i.val=.mk .fp32
    (Logical.db p.rows (fun row => bdy p (saved p s) row) i.val)) ∧
  (∀ r o, r≠dw → r≠db → (∀ pid, pid<programs → Backward.Outside p (atPid s pid) r o) →
    out.state.mem r o=s.mem r o)

theorem backward (rest : Algebra ℝ) (p : Backward.Parameters) (s : State ℝ)
    (programs : Nat) (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (shape : List Nat)
    (h : Layout p programs dw db shape) :
    ∃ out, Host.backward (algebra rest) sumPrimitive Returned.scalar p programs pw pb dw db shape s=some out ∧
      Correct p s programs dw db shape out := by
  refine ⟨Host.finish Returned.scalar p programs pw pb dw db shape
    (Grid.reference (Backward.reference (algebra rest) sumPrimitive p) s programs),
    Host.backward_complete _ _ _ _ _ _ _ _ _ _ _,rfl,rfl,rfl,?_,?_,?_,?_⟩
  · intro row hr i hi
    rw [Host.finish_frame _ _ _ _ _ _ _ _ _ _ _ h.return_dw_dx h.return_db_dx]
    exact Returned.dx_cell rest p s programs h.programs_pos h.inputs_fresh h.partial_dx_dw h.partial_dx_db
      h.block (by rw [h.dx_stride]) h.group row hr i hi
  · exact Returned.dw_return rest p s programs h.programs_pos h.rows_pos h.inputs_fresh
      h.partial_dw_db h.partial_dx_dw.symm (by rw [h.dw_stride]) h.block h.group pw pb dw db shape h.return_dw_db
  · exact Returned.db_return rest p s programs h.programs_pos h.rows_pos h.inputs_fresh
      h.partial_dw_db.symm h.partial_dx_db.symm (by rw [h.db_stride]) h.block h.group pw pb dw db shape h.return_dw_partial_db
  · intro r o hw hb ho
    rw [Host.finish_frame _ _ _ _ _ _ _ _ _ _ _ hw hb]
    apply Grid.frame
    intro pid hp mem
    apply Backward.reference_frame
    simpa [Backward.Outside,Backward.start,Backward.stop,Grid.launch,atPid] using ho pid hp

structure Link (fp : Forward.Parameters) (bp : Backward.Parameters) (rows : Nat) : Prop where
  x : bp.x=fp.x
  weight : bp.weight=fp.weight
  mean : bp.mean=fp.mean
  rstd : bp.rstd=fp.rstd
  columns : bp.columns=fp.columns
  block : bp.block=fp.block
  row_count : bp.rows=rows
  x_stride : bp.xStride=fp.xStride
  mean_stride : fp.meanStride=1
  rstd_stride : fp.rstdStride=1
  y_stride : fp.columns≤fp.yStride
  fresh : ForwardSaved.Fresh fp
  mean_y : fp.mean≠fp.y
  mean_rstd : fp.mean≠fp.rstd
  rstd_y : fp.rstd≠fp.y

/-- The public composition includes the produced Y and saved statistics and
the actual three backward return tensors. Saved-statistics correctness is a
conclusion about the forward grid, never an input premise. -/
theorem roundtrip (rest : Algebra ℝ) (fp : Forward.Parameters) (bp : Backward.Parameters)
    (s : State ℝ) (rows programs : Nat) (pw pb : ReductionPlan.Plan programs)
    (dw db : RegionName) (shape : List Nat) (h : Layout bp programs dw db shape)
    (link : Link fp bp rows) :
    ∃ f out, Grid.run (withSum (algebra rest) sumPrimitive) (Forward.kernel fp) s rows=some f ∧
      Host.forwardBackward (algebra rest) sumPrimitive Returned.scalar fp bp rows programs pw pb dw db shape s=some out ∧
      Correct bp f programs dw db shape out ∧
      (∀ row, row<rows →
        Backward.mean bp (saved bp f) row=mean fp.columns (rowX fp (atPid s row)) ∧
        Backward.rstd bp (saved bp f) row=rstd fp.columns fp.epsilon (rowX fp (atPid s row))) ∧
      (∀ row, row<rows → ∀ i : TileIndex [fp.block], i.1.val<fp.columns →
        f.mem fp.y (row*fp.yStride+i.1.val)=.mk .fp32
          (y fp.columns fp.epsilon (rowX fp (atPid s row)) (weight fp s) (bias fp s) i.1.val)) := by
  let f := Grid.reference (Forward.reference (algebra rest) sumPrimitive fp) s rows
  obtain ⟨out,hex,hcorrect⟩ := backward rest bp f programs pw pb dw db shape h
  have hf := Grid.forward_complete (algebra rest) sumPrimitive fp s rows
  have hblock : fp.columns≤fp.block := by rw [← link.columns,← link.block]; exact h.block
  refine ⟨f,out,hf,?_,hcorrect,?_,?_⟩
  · simpa [Host.forwardBackward,hf,Option.bind,f] using hex
  · intro row hr
    exact ForwardSaved.saved_statistics rest fp bp s rows row hr link.fresh hblock
      link.mean_y link.mean_rstd link.rstd_y link.mean_stride link.rstd_stride link.mean link.rstd
  · intro row hr i hi
    exact ForwardSaved.output rest fp s rows row hr link.fresh hblock link.mean_y.symm
      link.rstd_y.symm link.y_stride i hi

#axiomsClean backward
#axiomsClean roundtrip
#specNonCircular Correct avoiding [Forward.kernel,Backward.kernel,Forward.reference,Backward.reference,
  Grid.reference,Host.backward,Host.forwardBackward]
end
end VeriTile.Bench.Repairs.LayerNormLiger.Public
