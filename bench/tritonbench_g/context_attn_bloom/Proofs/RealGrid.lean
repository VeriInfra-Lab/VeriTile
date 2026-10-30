import bench.tritonbench_g.context_attn_bloom.Proofs.RealProgramConnection

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.RealKernelIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
noncomputable section
open Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

def TileOutside (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : Values r extent)
    (launches : List (GridIndex (Host.grid g.arguments))) (rg : RegionName) (off : Nat) : Prop :=
  ∀ idx∈launches,∀ i,tileActive g r extent xs idx i → rg≠r.out ∨ off≠tileOffset g r extent xs idx i

/-- The same actual serial grid as the retained physical Real proof. Source
safety belongs to this execution witness at every intermediate memory. -/
theorem schedule_complete (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (valid : domain g r extent xs)
    (launches : List (GridIndex (Host.grid g.arguments))) (s : BlockState)
    (hin : (io g r extent).InputBinding s xs) :
    ∃ t,∃ trace : AlgorithmExecution (launches.map (fun idx=>.single (source g r) (launch g idx))) s t,
      (∀ bounds,(io g r extent).InBounds bounds s xs → trace.Safe bounds) ∧
      (io g r extent).InputBinding t xs ∧
      (∀ idx∈launches,∀ i,tileActive g r extent xs idx i →
        t.readMem r.out (tileOffset g r extent xs idx i)=tileResult g r extent xs idx i) ∧
      (∀ rg off,TileOutside g r extent xs launches rg off → t.mem rg off=s.mem rg off) := by
  induction launches generalizing s with
  | nil => exact ⟨s,.nil s,fun _ _=>trivial,hin,by simp,fun _ _ _=>rfl⟩
  | cons idx rest ih =>
    obtain ⟨mid,ran,headValues,frame⟩ := stage_correct g r extent xs valid s hin idx
    have hmid : (io g r extent).InputBinding mid xs := binding_preserved g r extent xs valid s mid hin
      (fun rg off different=>frame rg off (fun _ _=>Or.inl different))
    obtain ⟨t,tail,safe,hout,values,tailFrame⟩ := ih mid hmid
    refine ⟨t,.cons ran tail,?_,hout,?_,?_⟩
    · intro bounds hb
      exact ⟨stage_safe g r extent xs valid s hin bounds hb idx,safe bounds ⟨hb.1,hb.2.1,fun k=>Fin.elim0 k⟩⟩
    · intro j hj lane active
      by_cases later : j∈rest
      · exact values j later lane active
      · have equal : j=idx := by simpa [later] using hj
        subst j
        have stable := tailFrame r.out (tileOffset g r extent xs idx lane) (by
          intro k hk other on
          right
          intro same
          have unique := (KernelIO.unique_writer g.arguments r extent (mathInputs r extent xs) valid idx k lane other active on same).1
          exact later (unique ▸ hk))
        simpa only [BlockState.readMem,stable] using headValues lane active
    · intro rg off outside
      rw [tailFrame rg off (fun j hj lane active=>outside j (List.mem_cons_of_mem idx hj) lane active)]
      exact frame rg off (fun lane active=>outside idx (List.mem_cons_self ..) lane active)

theorem implements (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) :
    (io g r extent).ImplementsOn (domain g r extent) (formula g r extent) := by
  intro xs valid s hin
  have hin0 : (io g r extent).InputBinding (AlgorithmInitialUndef.zero.initial s) xs := by
    intro i j active
    have bound := hin i j active
    change Fin 8 at i
    fin_cases i <;> exact bound
  obtain ⟨t,trace,safe,bound,values,frame⟩ := schedule_complete g r extent xs valid
    (KernelIO.jobs g) (AlgorithmInitialUndef.zero.initial s) hin0
  refine ⟨t,trace,?_,?_,?_⟩
  · intro bounds hb
    exact safe bounds ⟨hb.1,hb.2.1,fun k=>Fin.elim0 k⟩
  · intro o i active
    exact values (KernelIO.coordinate g i).1 (by simp [KernelIO.jobs]) (KernelIO.coordinate g i).2 active
  · intro rg off outside _
    apply frame rg off
    intro idx _ lane active
    have miss := outside (0 : Fin 1) ((KernelIO.coordinate g).symm (idx,lane))
    simp only [io,Equiv.apply_symm_apply] at miss
    exact miss active

#axiomsClean implements
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.RealKernelIO
