import bench.tritonbench_g.flash_attn.Proofs.RealLoop
import bench.tritonbench_g.flash_attn.Proofs.GridValues

namespace VeriTile.Bench.Repairs.FlashAttn.KernelReal
open VeriTile Triton FP.Structural ActualSource
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
attribute [local irreducible] Reduction.plan Reduction.Tree.eval

theorem grid_output (x : Layout) (hs : x.shape.Valid) (ho : OutputInjective x)
    (m : Memory ℝ) (scale : ℝ) (b h row col : Nat)
    (active : (Work.output b h row col).active x.shape) :
    referenceGrid algebra dot (parameters x) (some scale) (programs (parameters x))
      (liftMemory (Real.lift m)) "O" (FlashAttn.address x.o b h row col)=
      .mk .real (some (Spec.output x m scale b h row col)) := by
  let job : Program:=(row/128,b*x.shape.heads+h)
  let ri : Fin 128:=⟨row%128,Nat.mod_lt _ (by decide)⟩
  let ci : Fin x.shape.dim:=⟨col,active.2.2.2⟩
  have member:=output_program_present x b h row ⟨active.1,active.2.1,active.2.2.1⟩
  have valid : job.Valid x := (program_valid x job).mp member
  have hd:=flatten_decode x.shape.heads b h active.2.1
  have hr : row/128*128+ri.val=row := by dsimp [ri];omega
  have laneActive (memory : Mem R) : ActualSource.active (parameters x) (stateAt (parameters x) job memory) (ri,ci,()) := by
    simpa [ActualSource.active,stateAt,job,parameters,hr] using active.2.2.1
  apply grid_cell algebra dot (parameters x) (some scale) "O" (FlashAttn.address x.o b h row col)
    job (.mk .real (some (Spec.output x m scale b h row col))) (liftMemory (Real.lift m))
  · intro memory inputs
    have write:=program_output algebra dot x hs ho (some scale) job valid memory (ri,ci,()) (laneActive memory)
    have addr : ActualSource.address (parameters x) (stateAt (parameters x) job memory) (ri,ci,())=
        FlashAttn.address x.o b h row col := by
      simp [ActualSource.address,base,stateAt,job,parameters,hd.1,hd.2,hr,ci,FlashAttn.address]
    rw [addr,accumulated_input_eq algebra dot (parameters x) (some scale) job memory _ inputs] at write
    rw [write]
    congr 1
    have value:=output_math x hs m scale job.1 job.2 ri ci (by simpa [job,hr] using active.2.2.1)
    simpa [state,job,hr,hd.1,hd.2,ci] using value
  · exact member
  · intro j hj hw
    exact output_writer_unique x hs ho b h row col active j ((program_valid x j).mp hj) hw
  · exact fun _ _ _=>rfl

theorem grid_log (x : Layout) (hs : x.shape.Valid) (m : Memory ℝ) (scale : ℝ) (b h row : Nat)
    (active : (Work.logsum b h row).active x.shape) :
    referenceGrid algebra dot (parameters x) (some scale) (programs (parameters x))
      (liftMemory (Real.lift m)) "L" (FlashAttn.lAddress x.shape b h row)=
      .mk .real (some (Spec.lse x m scale b h row)) := by
  let job : Program:=(row/128,b*x.shape.heads+h)
  let ri : Fin 128:=⟨row%128,Nat.mod_lt _ (by decide)⟩
  have member:=output_program_present x b h row active
  have hd:=flatten_decode x.shape.heads b h active.2.1
  have hr : row/128*128+ri.val=row := by dsimp [ri];omega
  have laneActive (memory : Mem R) : lActive (parameters x) (stateAt (parameters x) job memory) (ri,()) := by
    simpa [lActive,stateAt,job,parameters,hr] using active.2.2
  apply grid_cell algebra dot (parameters x) (some scale) "L" (FlashAttn.lAddress x.shape b h row)
    job (.mk .real (some (Spec.lse x m scale b h row))) (liftMemory (Real.lift m))
  · intro memory inputs
    have write:=program_log algebra dot (parameters x) (some scale) job memory (ri,()) (laneActive memory)
    have addr : ActualSource.lAddress (parameters x) (stateAt (parameters x) job memory) (ri,())=
        FlashAttn.lAddress x.shape b h row := by simp [ActualSource.lAddress,stateAt,job,parameters,hr,FlashAttn.lAddress]
    rw [addr,accumulated_input_eq algebra dot (parameters x) (some scale) job memory _ inputs] at write
    rw [write]
    congr 1
    have value:=log_math x hs m scale job.1 job.2 ri (by simpa [job,hr] using active.2.2)
    simpa [state,job,hr,hd.1,hd.2] using value
  · exact member
  · intro j _ hw
    exact log_writer_unique x b h row active j hw
  · exact fun _ _ _=>rfl

/-- The complete shared-source launch, independent Real output/L formulas,
all exterior cells, and every Q/K/V storage cell. No result premise occurs. -/
theorem whole (x : Layout) (hs : x.shape.Valid) (ho : OutputInjective x) (m : Memory ℝ) (scale : ℝ) :
    ∃ final,runGrid algebra dot (parameters x) (some scale) (programs (parameters x)) (liftMemory (Real.lift m))=some final ∧
      (∀ b h row col,(Work.output b h row col).active x.shape →
        final "O" (FlashAttn.address x.o b h row col)=.mk .real (some (Spec.output x m scale b h row col))) ∧
      (∀ b h row,(Work.logsum b h row).active x.shape →
        final "L" (FlashAttn.lAddress x.shape b h row)=.mk .real (some (Spec.lse x m scale b h row))) ∧
      (∀ r off,¬Writes (parameters x) (programs (parameters x)) r off →
        final r off=liftMemory (Real.lift m) r off) ∧
      (∀ r∈(["Q","K","V"] : List RegionName),∀ off,final r off=liftMemory (Real.lift m) r off) := by
  refine ⟨referenceGrid algebra dot (parameters x) (some scale) (programs (parameters x)) (liftMemory (Real.lift m)),
    source_grid _ _ _ _ _ _,grid_output x hs ho m scale,grid_log x hs m scale,?_,?_⟩
  · exact fun r off h=>grid_frame _ _ _ _ _ _ r off h
  · exact fun r hr off=>input_frame _ _ _ _ _ _ r hr off

#axiomsClean grid_output
#axiomsClean grid_log
#axiomsClean whole
#specNonCircular Spec.output avoiding [FP.Structural.exec,FP.Structural.run,runGrid,referenceGrid]
#specNonCircular Spec.lse avoiding [FP.Structural.exec,FP.Structural.run,runGrid,referenceGrid]
end
end VeriTile.Bench.Repairs.FlashAttn.KernelReal
