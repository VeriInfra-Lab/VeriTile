import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelContract

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelGeometry
open VeriTile.Triton FP.Structural KernelSource KernelValues KernelContract
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem branch_min (a b : Nat) : (if a<b then a else b)=min a b := by
  by_cases h : a<b
  · simp [h,min_eq_left (le_of_lt h)]
  · simp [h,min_eq_right (le_of_not_gt h)]

theorem row_configured (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName)
    (s : State α) (i : Fin (configs c).bm) :
    KernelValues.row (parameters r c regions) s i=TritonLinearActivation.row r.shape c (s.pids 0) i.val := by
  simp only [KernelValues.row,blockRow,groupIdx,width,KernelValues.groupSize,gridRows,gridCols,parameters,branch_min]
  rfl

theorem col_configured (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName)
    (s : State α) (i : Fin (configs c).bn) :
    KernelValues.col (parameters r c regions) s i=TritonLinearActivation.col r.shape c (s.pids 0) i.val := by
  simp only [KernelValues.col,blockCol,groupIdx,width,KernelValues.groupSize,gridRows,gridCols,parameters,branch_min]
  rfl

def atProgram (s : State α) (pid : Nat) : State α :=
  {s with pids:=fun ax => if ax=0 then pid else s.pids ax}

/-- All logical output coordinates have an active source lane in the actual
98-configuration grouped launch, including the final short group. -/
theorem all_outputs_covered (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName)
    (s : State α) (m n : Nat) (hm : m<r.batch.rows) (hn : n<r.n) :
    ∃ pid, pid<r.shape.programs c ∧ ∃ i : TileIndex [(configs c).bm,(configs c).bn],
      active (parameters r c regions) (atProgram s pid) i=Bool.true ∧
      KernelValues.row (parameters r c regions) (atProgram s pid) i.1=m ∧
      KernelValues.col (parameters r c regions) (atProgram s pid) i.2.1=n := by
  obtain ⟨pid,hp,i,j,hi,hj,hr,hc⟩ := source_cell_covered r.shape c m n hm hn
  refine ⟨pid,hp,(⟨i,hi⟩,⟨j,hj⟩,PUnit.unit),?_,?_,?_⟩
  · simp only [active,row_configured,col_configured,atProgram,if_true]
    simpa [parameters,hr,hc] using And.intro hm hn
  · simpa [row_configured,atProgram] using hr
  · simpa [col_configured,atProgram] using hc

/-- Checked signed-32 formed output offsets are reused for the exact DSL
geometry. This includes inactive ragged lanes, not just performed stores. -/
theorem formed_output_bound (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName)
    (s : State α) (ts : Fin 4 → Checked.Observed) (flag : Bool)
    (admit : Checked.Domain r ts flag) (pid : s.pids 0<r.shape.programs c)
    (i : TileIndex [(configs c).bm,(configs c).bn]) :
    KernelValues.row (parameters r c regions) s i.1<2^31 ∧
      KernelValues.col (parameters r c regions) s i.2.1<2^31 ∧
      KernelValues.outputOffset (parameters r c regions) s i<2^31 := by
  simp only [KernelValues.outputOffset,row_configured,col_configured]
  simpa only [parameters,Request.shape,Nat.mul_one] using
    formed_output r.shape c admit.bounds (s.pids 0) i.1.val i.2.1.val pid i.1.isLt i.2.1.isLt

#axiomsClean all_outputs_covered
#axiomsClean formed_output_bound
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelGeometry
