import bench.tritonbench_g.context_attn_bloom.Proofs.IOProgramConnection

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
noncomputable section
open Classical
set_option maxHeartbeats 2200000
set_option maxRecDepth 20000
variable {α : Type} [Inhabited α]

theorem input_output_separate (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent)
    (valid : domain g r extent xs) (i : Fin (io g r extent M dots).inputs.length) :
    (io g r extent M dots).inputs[i].buf≠r.out := by
  have h := valid.separate
  change Fin 8 at i
  fin_cases i <;> simp only [io,memory,inputs,List.getElem_cons_zero,List.getElem_cons_succ] <;> tauto

theorem binding_preserved (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent)
    (valid : domain g r extent xs) (s t : State α)
    (hin : (io g r extent M dots).InputBinding xs s)
    (frame : ∀ rg off,rg≠r.out → t.mem rg off=s.mem rg off) :
    (io g r extent M dots).InputBinding xs t := by
  intro i j active
  rw [frame _ _ (input_output_separate g r extent M dots xs valid i)]
  exact hin i j active

def TileOutside (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : Values α r extent)
    (launches : List (GridIndex (Host.grid g.arguments))) (rg : RegionName) (off : Nat) : Prop :=
  ∀ idx∈launches,∀ i,tileActive g r extent xs idx i → rg≠r.out ∨ off≠tileOffset g r extent xs idx i

/-- Compose the existing complete source theorem over the actual serial grid.
Every successor reads the preceding state; only original input separation and
unique output ownership are used to preserve logical bindings and outputs. -/
theorem schedule_complete (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent)
    (valid : domain g r extent xs) (bounds : RegionBounds)
    (hb : (io g r extent M dots).InBounds bounds xs)
    (launches : List (GridIndex (Host.grid g.arguments))) (s : State α)
    (hin : (io g r extent M dots).InputBinding xs s) :
    ScheduleSafe bounds (launches.map (invocation g r M dots)) s ∧
      ∃ t,runSchedule (launches.map (invocation g r M dots)) s=some t ∧
        (io g r extent M dots).InputBinding xs t ∧
        (∀ idx∈launches,∀ i,tileActive g r extent xs idx i →
          t.mem r.out (tileOffset g r extent xs idx i)=.mk .real (tileResult g r extent M dots xs idx i)) ∧
        (∀ rg off,TileOutside g r extent xs launches rg off → t.mem rg off=s.mem rg off) := by
  induction launches generalizing s with
  | nil => exact ⟨trivial,s,rfl,hin,by simp,fun _ _ _=>rfl⟩
  | cons idx rest ih =>
    obtain ⟨mid,ran,memory⟩ := invocation_correct g r extent M dots xs valid s hin idx
    have frame : ∀ rg off,(∀ i,tileActive g r extent xs idx i → rg≠r.out ∨ off≠tileOffset g r extent xs idx i) →
        mid.mem rg off=s.mem rg off := by
      intro rg off outside
      rw [memory]
      exact writeTile_frame g r extent M dots xs idx s rg off outside
    have hmid : (io g r extent M dots).InputBinding xs mid := binding_preserved g r extent M dots xs valid s mid hin
      (fun rg off different=>frame rg off (fun _ _=>Or.inl different))
    obtain ⟨safe,t,tailRun,hout,values,tailFrame⟩ := ih mid hmid
    refine ⟨?_,t,?_,hout,?_,?_⟩
    · refine ⟨invocation_safe g r extent M dots xs valid bounds hb s hin idx,?_⟩
      intro after hafter
      have equal : after=mid := Option.some.inj (hafter.symm.trans ran)
      subst after
      exact safe
    · simp only [List.map_cons,runSchedule,ran,Option.bind_some]
      exact tailRun
    · intro j hj lane active
      by_cases later : j∈rest
      · exact values j later lane active
      · have equal : j=idx := by simpa [later] using hj
        subst j
        rw [tailFrame r.out (tileOffset g r extent xs idx lane) (by
          intro k hk other on
          right
          intro same
          have unique := (unique_writer g.arguments r extent (metadata r extent xs) valid idx k lane other active on same).1
          exact later (unique ▸ hk)),memory]
        exact writeTile_output g r extent M dots xs valid idx s lane active
    · intro rg off outside
      rw [tailFrame rg off (fun j hj lane active=>outside j (List.mem_cons_of_mem idx hj) lane active)]
      exact frame rg off (fun lane active=>outside idx (List.mem_cons_self ..) lane active)

theorem active_scheduled (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values α r extent) (idx : GridIndex (Host.grid g.arguments))
    (lane : RealIO.Lane (Host.layout g.arguments)) (active : tileActive g r extent xs idx lane) :
    idx∈order g r extent xs := by
  have need : needed g r extent xs := by
    refine ⟨(Host.launchOf g.arguments idx).batch,Host.batch_lt g.arguments idx,?_⟩
    have h := active.1
    have hn := Int.natCast_nonneg (Math.queryRow (Host.layout g.arguments) (Host.launchOf g.arguments idx) lane.1.val)
    change (_ : Int)<(metadata r extent xs).sequence (Host.launchOf g.arguments idx).batch-
      (metadata r extent xs).prompt (Host.launchOf g.arguments idx).batch at h
    omega
  simp [order,need,jobs]

theorem implements (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) :
    (io g r extent M dots).ImplementsOn (domain g r extent) (formula g r extent M dots) := by
  intro xs valid bounds hb s hin
  obtain ⟨safe,t,ran,bound,values,frame⟩ := schedule_complete g r extent M dots xs valid bounds hb
    (order g r extent xs) s hin
  refine ⟨safe,t,ran,?_,?_⟩
  · intro o lane active
    fin_cases o
    have result := values (coordinate g lane.1).1
      (active_scheduled g r extent xs (coordinate g lane.1).1 (coordinate g lane.1).2 active)
      (coordinate g lane.1).2 active
    change mapCell id (t.mem r.out (tileOffset g r extent xs (coordinate g lane.1).1 (coordinate g lane.1).2))=_
    rw [result]
    rfl
  · intro rg off outside
    apply frame rg off
    intro idx _ lane active
    have miss := outside.1 (0 : Fin 1) ((coordinate g).symm (idx,lane),PUnit.unit)
    simp only [io,memory,address,KernelIO.active,Equiv.apply_symm_apply] at miss
    exact miss active

#axiomsClean implements
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelIO
