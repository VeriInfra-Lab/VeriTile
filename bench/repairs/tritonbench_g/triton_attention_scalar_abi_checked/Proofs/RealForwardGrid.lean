import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.RealForwardSaved
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.ForwardGridStores
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def ForwardInputs (mem : Grid.Memory R) (μ : RegionName → Nat → ℝ) : Prop :=
  ∀ r∈(["Q","K","V"] : List RegionName),∀ o,(mem r o).read .fp32=some (μ r o)
def tensorAt (c : Shape) (μ : RegionName → Nat → ℝ) (r : RegionName) (h i d : Nat) : ℝ :=
  μ r ((h*c.rows+i)*c.dim+d)
def outputReference (c : Shape) (scale : ℝ) (μ : RegionName → Nat → ℝ) (i : Coord c) : ℝ :=
  Mathematical.attention c.dim scale (tensorAt c μ "Q" i.1.val) (tensorAt c μ "K" i.1.val)
    (tensorAt c μ "V" i.1.val) i.2.1.val i.2.2.val

theorem inputs_forwardReads (c : Shape) (scale : ℝ) (pid : Grid.Program) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) (h : ForwardInputs mem μ) :
    ForwardReads (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid mem) μ := by
  exact h

theorem forwardStep_inputs (c : Shape) (scale : ℝ) (pid : Grid.Program) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) (h : ForwardInputs mem μ) (hp : Grid.Legal (Grid.forwardGrid c) pid) :
    ForwardInputs (Grid.forwardStep model c scale pid mem) μ := by
  intro r hr o
  rw [Grid.forwardStep_frame model c scale pid mem hp r o]
  · exact h r hr o
  · simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl <;> simp [Grid.Written]

theorem qAt_entry (c : Shape) (scale : ℝ) (pid : Grid.Program) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) :
    qAt (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid mem) μ=tensorAt c μ "Q" pid.2 := by
  funext row d
  simp [qAt,tensorAt,forwardHeadRow,forwardParameters,Grid.entry,Grid.ids,c.dim_pos]

theorem kAt_entry (c : Shape) (scale : ℝ) (pid : Grid.Program) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) :
    kAt (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid mem) μ=tensorAt c μ "K" pid.2 := by
  funext row d
  simp [kAt,tensorAt,forwardHeadRow,forwardParameters,Grid.entry,Grid.ids,c.dim_pos]

theorem vAt_entry (c : Shape) (scale : ℝ) (pid : Grid.Program) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) :
    vAt (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid mem) μ=tensorAt c μ "V" pid.2 := by
  funext row d
  simp [vAt,tensorAt,forwardHeadRow,forwardParameters,Grid.entry,Grid.ids,c.dim_pos]

theorem forwardStep_output (c : Shape) (scale : ℝ) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) (h : ForwardInputs mem μ) (i : Coord c) :
    Grid.forwardStep model c scale (Grid.forwardProgram c i) mem "Out" (offset c i)=
      .mk .fp32 (some (outputReference c scale μ i)) := by
  let pid:=Grid.forwardProgram c i
  let s:=Grid.entry (Grid.forwardGrid c) pid mem
  have hp:=Grid.forwardProgram_legal c i
  have h0 : s.pids 0<c.blocks := hp.1
  have h1 : s.pids 1<c.groups := hp.2
  obtain ⟨final,he,ho,_⟩:=forward_execution_real c scale s μ (inputs_forwardReads c scale pid mem μ h) h0 h1
  dsimp only [s] at he
  have hm : final.mem=Grid.forwardStep model c scale pid mem := by
    simpa only [Grid.execute,he,Option.map_some,Option.some.injEq] using Grid.forward_success algebra model c scale pid mem
  have ans:=ho (Grid.forwardLane c scale i)
  rw [hm,Grid.forward_target_offset] at ans
  change _=Cell.mk .fp32 (some (Mathematical.attention c.dim scale
    (qAt (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid mem) μ)
    (kAt (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid mem) μ)
    (vAt (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid mem) μ)
    (pid.1*128+(Grid.forwardLane c scale i).1.val) i.2.2.val)) at ans
  rw [qAt_entry,kAt_entry,vAt_entry] at ans
  simpa [outputReference,tensorAt,qAt,kAt,vAt,forwardHeadRow,forwardParameters,s,pid,Grid.entry,Grid.ids,
    Grid.forwardProgram,Grid.forwardLane,c.dim_pos,Nat.div_add_mod'] using ans

theorem forward_grid_output (c : Shape) (scale : ℝ) (work : List Grid.Program)
    (schedule : Grid.FullSchedule (Grid.forwardGrid c) work) (mem : Grid.Memory R)
    (μ : RegionName → Nat → ℝ) (h : ForwardInputs mem μ) :
    ∃ final,Grid.run (Grid.execute algebra model (forwardKernel c scale) (Grid.forwardGrid c)) work mem=some final ∧
      ForwardInputs final μ ∧
      ∀ i : Coord c,final "Out" (offset c i)=.mk .fp32 (some (outputReference c scale μ i)) := by
  let final:=work.foldl (fun m pid=>Grid.forwardStep model c scale pid m) mem
  have hlegal:=Grid.FullSchedule.legal (Grid.forwardGrid c) work schedule
  have hinv : ForwardInputs final μ :=
    Grid.fold_invariant (Grid.forwardStep model c scale) (fun m=>ForwardInputs m μ) work mem h
      (by intro pid hp m hm;exact forwardStep_inputs c scale pid m μ hm (hlegal pid hp))
  refine ⟨final,Grid.run_eq_fold _ _ (Grid.forward_success algebra model c scale) work mem,hinv,?_⟩
  intro i
  exact Grid.fold_readback (Grid.forwardStep model c scale) (fun m=>ForwardInputs m μ) work mem
    (Grid.forwardProgram c i) "Out" (offset c i) (.mk .fp32 (some (outputReference c scale μ i)))
    (Grid.FullSchedule.covers (Grid.forwardGrid c) work schedule _ (Grid.forwardProgram_legal c i)) h
    (by intro pid hp m hm;exact forwardStep_inputs c scale pid m μ hm (hlegal pid hp))
    (by intro m hm;exact forwardStep_output c scale m μ hm i)
    (by intro pid hp hn m;exact Grid.forwardStep_otherOut model c scale pid m (hlegal pid hp) i hn)

#axiomsClean forward_grid_output
#specNonCircular outputReference avoiding [FP.Structural.exec,forwardRounds,Grid.run,Grid.forwardStep]
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
