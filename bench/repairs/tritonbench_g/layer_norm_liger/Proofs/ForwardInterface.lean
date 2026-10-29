import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.OrderedMemory
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardRun
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardContract

namespace VeriTile.Bench.Repairs.LayerNormLiger.Ordered
open VeriTile.Triton FP.Structural LayerIO
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive)
noncomputable section
variable {α : Type} [Inhabited α]
set_option linter.unusedSimpArgs false

private theorem f_loads (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (pid : Nat) (v : ChannelValues α (fInputs p)) (s : State α)
    (hin : (forwardIO M R p pid).InputBinding v s) :
    let t := (fInvocation M R p pid).initial s
    Forward.xValues M p t=fLoad M p v 0 ∧
    Forward.wValues M p t=fLoad M p v 1 ∧
    Forward.bValues M p t=fLoad M p v 2 := by
  refine ⟨?_,?_,?_⟩
  all_goals
    funext i
    by_cases hi : i.1.val<p.columns
  · have cell := hin (0 : Fin 3) i hi
    simpa [Forward.xValues,Forward.input,fLoad,fInput,hi,forwardIO,fMemory,fInputs,mapValue] using cell
  · simp [Forward.xValues,Forward.input,fLoad,hi]
  · have cell := hin (1 : Fin 3) i hi
    simpa [Forward.wValues,Forward.input,fLoad,fInput,hi,forwardIO,fMemory,fInputs,mapValue] using cell
  · simp [Forward.wValues,Forward.input,fLoad,hi]
  · have cell := hin (2 : Fin 3) i hi
    simpa [Forward.bValues,Forward.input,fLoad,fInput,hi,forwardIO,fMemory,fInputs,mapValue] using cell
  · simp [Forward.bValues,Forward.input,fLoad,hi]

private theorem forward_match (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (pid : Nat) (v : ChannelValues α (fInputs p)) (s : State α)
    (hin : (forwardIO M R p pid).InputBinding v s) :
    let initial := (fInvocation M R p pid).initial s
    Match initial (Forward.reference M R p initial) (forwardWrites M R p pid v) := by
  dsimp only
  let initial := (fInvocation M R p pid).initial s
  obtain ⟨hx,hw,hb⟩ := f_loads M R p pid v s hin
  change Forward.xValues M p initial=fLoad M p v 0 at hx
  change Forward.wValues M p initial=fLoad M p v 1 at hw
  change Forward.bValues M p initial=fLoad M p v 2 at hb
  have mean : Forward.meanValue M R p initial=fMean M R p v := by
    simp only [Forward.meanValue,fMean,hx]
  have center : Forward.centered M R p initial=fCentered M R p v := by
    funext i
    simp only [Forward.centered,fCentered,hx,mean]
  have variance : Forward.varianceValue M R p initial=fVariance M R p v := by
    simp only [Forward.varianceValue,fVariance,center]
  have rstd : Forward.rstdValue M R p initial=fRstd M R p v := by
    simp only [Forward.rstdValue,fRstd,variance]
  have y : Forward.yValue M R p initial=fY M R p v := by
    funext i
    simp only [Forward.yValue,fY,center,rstd,hw,hb]
  unfold Forward.reference forwardWrites
  simp only [Forward.meanOffset,Forward.rstdOffset,Forward.outputOffset,Forward.stored,mean,rstd,y,
    initial,f_pid]
  exact match_scatter
    (match_put (match_put (match_initial _) _ _ _) _ _ _) _ _ _ _

theorem forward_present (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (pid : Nat) (v : ChannelValues α (fInputs p)) (o : Fin 3)
    (i : TileIndex ((fOutputs p)[o]).shape) (active : (fMemory α p pid).outputMask o v i) :
    Present (forwardWrites M R p pid v) (fOutputs p)[o].buf ((fMemory α p pid).write o v i) := by
  fin_cases o
  · exact scatter_present _ _ _ _ _ i active
  · apply scatter_preserves
    exact put_preserves _ _ _ _ _ _ (put_present _ _ _ _)
  · apply scatter_preserves
    exact put_present _ _ _ _

theorem forward_implements (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters) (pid : Nat) :
    (forwardIO M R p pid).Implements (forward M R p pid) := by
  apply ProgramIO.Implements.intro
  · intro v bounds hb s _
    exact forward_safe _ _ _ _ _ _ hb s
  · intro v s hin
    let initial := (fInvocation M R p pid).initial s
    obtain ⟨t,ran,memory⟩ := Forward.complete_run M R p initial
    have matched := forward_match M R p pid v s hin
    refine ⟨t,?_,?_,?_⟩
    · exact (runSchedule_singleton _ _).trans ran
    · intro o i active
      have cell := match_output matched (fOutputs p)[o].buf ((fMemory α p pid).write o v i)
        (forward_present M R p pid v o i active)
      fin_cases o
      all_goals
        change mapCell id (t.mem _ _) = _
        rw [memory]
        simpa only [forwardIO,fMemory,fOutputs,forward,mapCell,mapValue,initial] using congrArg (mapCell id) cell
    · intro r o outside
      rw [memory]
      exact (Forward.reference_frame _ _ _ _ r o
        (forward_outside _ _ _ _ _ _ _ _ outside)).trans
          (congrFun (congrFun (Invocation.initial_mem _ _) r) o)

#axiomsClean forward_implements
end
end VeriTile.Bench.Repairs.LayerNormLiger.Ordered
