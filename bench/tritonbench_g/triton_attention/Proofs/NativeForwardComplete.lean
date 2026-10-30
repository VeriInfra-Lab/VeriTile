import bench.tritonbench_g.triton_attention.Proofs.NativeForwardGrid
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native.Selected
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

theorem forwardStep_otherOut (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) (mem : Grid.Memory Graph)
    (hp : Grid.Legal (Grid.forwardGrid c) pid) (i : Coord c) (hn : pid≠Grid.forwardProgram c i) :
    store one .forward c scale owners pid mem "Out" (offset c i)=mem "Out" (offset c i) := by
  change mapCell _ (Grid.forwardStep (modelScalar one .forward) c scale pid (embed mem) "Out" (offset c i))=_
  rw [Grid.forwardStep_otherOut _ c scale pid (embed mem) hp i hn]
  exact congrFun (congrFun (finish_embed one .forward c owners mem) "Out") (offset c i)

theorem forwardStep_otherRow (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) (mem : Grid.Memory Graph)
    (hp : Grid.Legal (Grid.forwardGrid c) pid) (i : Coord c) (hn : pid≠Grid.forwardProgram c i)
    (r : RegionName) (hr : r="L" ∨ r="M") :
    store one .forward c scale owners pid mem r (rowOffset c i.1 i.2.1)=mem r (rowOffset c i.1 i.2.1) := by
  change mapCell _ (Grid.forwardStep (modelScalar one .forward) c scale pid (embed mem) r (rowOffset c i.1 i.2.1))=_
  rw [Grid.forwardStep_otherRow _ c scale pid (embed mem) hp i hn r hr]
  exact congrFun (congrFun (finish_embed one .forward c owners mem) r) (rowOffset c i.1 i.2.1)

theorem forward_grid (one : Bool) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (work : List Grid.Program)
    (schedule : Grid.FullSchedule (Grid.forwardGrid c) work) (mem : Grid.Memory Graph)
    (μ : RegionName → Nat → Graph) (hi : Inputs mem μ) :
    ∃ final,Grid.run (execute one .forward c scale owners) work mem=some final ∧ Inputs final μ ∧
      ∀ i : Coord c,final "Out" (offset c i)=.mk .fp32 ((Target.forward c one scale (input c μ) owners).output i.1.val i.2.1.val i.2.2.val) ∧
        final "L" (rowOffset c i.1 i.2.1)=.mk .fp32 ((Target.forward c one scale (input c μ) owners).denominator i.1.val i.2.1.val) ∧
        final "M" (rowOffset c i.1 i.2.1)=.mk .fp32 ((Target.forward c one scale (input c μ) owners).maximum i.1.val i.2.1.val) := by
  let final:=work.foldl (fun m pid=>store one .forward c scale owners pid m) mem
  have inv : Inputs final μ := Grid.fold_invariant (store one .forward c scale owners) (fun m=>Inputs m μ) work mem hi
    (by intro pid hp m hm;exact forwardStep_inputs one c scale owners pid m μ hm)
  refine ⟨final,Grid.run_eq_fold _ _ (execute_success one .forward c scale owners) work mem,inv,?_⟩
  intro i
  have writer:=Grid.FullSchedule.covers _ _ schedule _ (Grid.forwardProgram_legal c i)
  have keep := fun pid (_ : pid∈work) m hm=>forwardStep_inputs one c scale owners pid m μ hm
  constructor
  · exact Grid.fold_readback (store one .forward c scale owners) (fun m=>Inputs m μ) work mem
      (Grid.forwardProgram c i) "Out" (offset c i) _ writer hi keep
      (by intro m hm;exact (forwardStep_values one c scale owners m μ hm i).1)
      (by intro pid hp hn m;exact forwardStep_otherOut one c scale owners pid m (Grid.FullSchedule.legal _ _ schedule pid hp) i hn)
  constructor
  · exact Grid.fold_readback (store one .forward c scale owners) (fun m=>Inputs m μ) work mem
      (Grid.forwardProgram c i) "L" (rowOffset c i.1 i.2.1) _ writer hi keep
      (by intro m hm;exact (forwardStep_values one c scale owners m μ hm i).2.1)
      (by intro pid hp hn m;exact forwardStep_otherRow one c scale owners pid m (Grid.FullSchedule.legal _ _ schedule pid hp) i hn "L" (Or.inl rfl))
  · exact Grid.fold_readback (store one .forward c scale owners) (fun m=>Inputs m μ) work mem
      (Grid.forwardProgram c i) "M" (rowOffset c i.1 i.2.1) _ writer hi keep
      (by intro m hm;exact (forwardStep_values one c scale owners m μ hm i).2.2)
      (by intro pid hp hn m;exact forwardStep_otherRow one c scale owners pid m (Grid.FullSchedule.legal _ _ schedule pid hp) i hn "M" (Or.inr rfl))

#axiomsClean forward_grid
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native.Selected
