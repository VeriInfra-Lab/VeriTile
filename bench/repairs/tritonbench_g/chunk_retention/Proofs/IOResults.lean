import bench.repairs.tritonbench_g.chunk_retention.Proofs.RealFormulaTransport
import bench.repairs.tritonbench_g.chunk_retention.Proofs.NativeInputTags
import bench.repairs.tritonbench_g.chunk_retention.Proofs.OutputTags

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel KernelPacket KernelStaging
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 1200000

private theorem output_dtype (c : Shape) (port : Fin (outputs c).length) :
    (outputs c)[port].dtype=.real := by
  rcases port with ⟨port,hp⟩
  have h : port<7 := hp
  interval_cases port <;> rfl

private theorem observed_cell {β : Type} [Inhabited β] (observe : β→α)
    (cell : Cell β) (dtype : TileDType) (expected : Value α dtype)
    (hd : dtype=.real) (tag : ∃ value,cell=Cell.mk .real value)
    (value : observe (cell.read .real)=(Cell.mk dtype expected).read .real) :
    mapCell observe cell=Cell.mk dtype expected := by
  subst dtype
  obtain ⟨v,rfl⟩ := tag
  exact congrArg (Cell.mk .real) value

/-- Reuses original successful execution, independent mathematical result,
typed store readback, and the exact exterior frame. Source safety is separate. -/
theorem real_result (a : Call) (p grid : Nat→Nat) (legal : Legal a p grid)
    (xs : Inputs ℝ a.shape) (s : State ℝ) (binding : (realIO a p grid).InputBinding xs s) :
    ∃ final,runSchedule ((realIO a p grid).schedule xs) s=some final ∧
      (realIO a p grid).OutputBinding xs (realFormula a p xs) final ∧
      (realIO a p grid).Frame xs s final := by
  let start := (launch (engine (withDot KernelReal.algebra KernelReal.dot)) a p grid).initial s
  have startLegal : KernelPacket.Legal a start := legal
  have startBinding : (realIO a start.pids start.numPids).InputBinding xs start := binding
  obtain ⟨final,ran,result,frame⟩ := real_complete a start startLegal
  refine ⟨final,?_,?_,?_⟩
  · simpa only [realIO,io,runSchedule_singleton,Invocation.run] using ran
  · intro port j active
    have hactive : Active a start (observation a.shape port j) :=
      (observation_active a start port j).mpr active
    have typed := output_tag KernelReal.algebra KernelReal.dot a start final ran _ hactive
    have value := (result _ hactive).trans (real_formula_transport a start legal xs startBinding port j active)
    rw [observation_region,observation_offset] at typed value
    exact observed_cell id _ _ _ (output_dtype a.shape port) typed value
  · intro r off outside
    exact frame r off ((outside_iff _ id id a start xs r off).mp outside)

/-- Reuses original native graph execution with only finite input provenance.
The primitive laws, casts, dot/FMA graph and all outputs keep their old scope. -/
theorem native_result (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (castZero : ops.cast ops.zero=ops.zero) (a : Call) (p grid : Nat→Nat) (legal : Legal a p grid)
    (xs : Inputs α a.shape) (s : State (Term α))
    (binding : (nativeIO ops literal a p grid).InputBinding xs s) :
    ∃ final,runSchedule ((nativeIO ops literal a p grid).schedule xs) s=some final ∧
      (nativeIO ops literal a p grid).OutputBinding xs (nativeFormula ops literal a p xs) final ∧
      (nativeIO ops literal a p grid).Frame xs s final := by
  let start := (launch (engine (withDot (algebra ops.zero) (nativeDot a))) a p grid).initial s
  have startLegal : KernelPacket.Legal a start := legal
  have startBinding : (nativeIO ops literal a start.pids start.numPids).InputBinding xs start := binding
  have tags : a.stage=.backwardQKV → KernelNative.QKV.ReadTags a.shape start :=
    fun hs=>native_input_tags ops literal a hs start legal xs startBinding
  obtain ⟨final,ran,result,frame⟩ := native_complete_local ops literal laws castZero a start tags startLegal
  refine ⟨final,?_,?_,?_⟩
  · simpa only [nativeIO,runSchedule_singleton,Invocation.run] using ran
  · intro port j active
    have hactive : Active a start (observation a.shape port j) :=
      (observation_active a start port j).mpr active
    have typed := output_tag (algebra ops.zero) (nativeDot a) a start final ran _ hactive
    have value := (result _ hactive).trans
      (native_formula_transport ops literal a start legal xs startBinding port j active)
    rw [observation_region,observation_offset] at typed value
    exact observed_cell (lower ops literal) _ _ _ (output_dtype a.shape port) typed value
  · intro r off outside
    exact frame r off ((outside_iff (engine (withDot (algebra ops.zero) (nativeDot a)))
      (Term.input "" 0) (lower ops literal) a start xs r off).mp outside)

#axiomsClean real_result
#axiomsClean native_result
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
