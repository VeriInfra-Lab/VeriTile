import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.InnerFiniteInputs

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RealInterface
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealHMath RealHProgram
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 16000

theorem inner_allocated (p : Layout) (heads K V : Nat) (r : Regions)
    (gateK hasSeed storeFinal : Bool) (values : (innerIO p heads K V r gateK hasSeed storeFinal).Ctx)
    (before : BlockState) (bounds : RegionBounds)
    (hb : (innerIO p heads K V r gateK hasSeed storeFinal).InBounds bounds before values) :
    StagedLayout.Allocated p heads r gateK hasSeed storeFinal bounds := by
  refine ⟨?_,?_,?_,?_,?_,?_⟩
  · exact capacity_bound (fun j => hb.1 (0 : Fin 4) j (Or.inl (by decide)))
  · exact capacity_bound (fun j => hb.1 (1 : Fin 4) j (Or.inl (by decide)))
  · exact capacity_bound (fun j => hb.1 (2 : Fin 4) j (Or.inl (by decide)))
  · exact capacity_bound (fun j => hb.2.2 (0 : Fin 2) j (Or.inl rfl))
  · intro enabled
    exact capacity_bound (fun j => hb.1 (3 : Fin 4) j (Or.inr enabled))
  · intro enabled
    exact capacity_bound (fun j => hb.2.2 (1 : Fin 2) j (Or.inr enabled))

theorem inner_plan_valid (p : Layout) (heads : Nat) (plan : LaunchPlan.Item)
    (hm : plan ∈ LaunchPlan.plans (heads*(p.K/p.BK)*(p.V/p.BV))) (u : BlockState)
    (idx : GridIndex (Batch.grid plan.size)) :
    StagedLayout.TileValid p heads (RebasedInit.decoded p plan.base (u.withGridIndex idx)) := by
  apply StagedLayout.decoded_valid
  rw [Batch.pid_lane]
  have valid := LaunchPlan.plans_valid _ plan hm
  have bound := valid.2.2
  have lane := Batch.lane_lt idx
  omega

theorem inner_logical_outputs (p : Layout) (heads T K V : Nat) (r : Regions)
    (gateK hasSeed storeFinal : Bool) (x : Nat → Host.InnerInputs) (a : Inputs)
    (hp : StagedLayout.Canonical p) (hk : K≤p.K) (hv : V≤p.V)
    (finite : FiniteInputs p heads T K V gateK hasSeed x a) (start after : BlockState)
    (out : ∀ plan ∈ LaunchPlan.plans (heads*(p.K/p.BK)*(p.V/p.BV)),
      ∀ idx : GridIndex (Batch.grid plan.size),
        RealHExecution.Outputs p (RebasedInit.decoded p plan.base (start.withGridIndex idx))
          r a gateK hasSeed storeFinal after) :
    DirectRealWhole.Outputs p heads T K V r x gateK hasSeed storeFinal after := by
  constructor
  · intro h c k j hhh hc hkk hjj
    have hkp := lt_of_lt_of_le hkk hk
    have hjp := lt_of_lt_of_le hjj hv
    obtain ⟨plan,hplan,idx,e,f,hhead,hek,hjv⟩ :=
      LogicalCoverage.coverage p heads h k j start hp hhh hkp hjp
    let l := RebasedInit.decoded p plan.base (start.withGridIndex idx)
    have hr := (out plan hplan idx).history c e f hc
      ⟨by simpa [hek] using hkp,by simpa [hjv] using hjp⟩
    have hcform := finite_closed p heads T K V l gateK hasSeed x a hp
      (inner_plan_valid p heads plan hplan start idx).head hk hv finite c _ _ hc.le
      (by simpa [l,hek] using hkk) (by simpa [l,hjv] using hjj)
    rw [hek,hjv] at hr
    have hvlog := hr.trans hcform
    simpa [historyAddress,DirectRealWhole.historyOffset,l,hhead,hek,hjv] using hvlog
  · intro hon h k j hhh hkk hjj
    have hkp := lt_of_lt_of_le hkk hk
    have hjp := lt_of_lt_of_le hjj hv
    obtain ⟨plan,hplan,idx,e,f,hhead,hek,hjv⟩ :=
      LogicalCoverage.coverage p heads h k j start hp hhh hkp hjp
    let l := RebasedInit.decoded p plan.base (start.withGridIndex idx)
    have hr := (out plan hplan idx).final hon e f
      ⟨by simpa [hek] using hkp,by simpa [hjv] using hjp⟩
    have hcform := finite_closed p heads T K V l gateK hasSeed x a hp
      (inner_plan_valid p heads plan hplan start idx).head hk hv finite p.NT _ _ le_rfl
      (by simpa [l,hek] using hkk) (by simpa [l,hjv] using hjj)
    rw [hek,hjv] at hr
    have hvlog := hr.trans hcform
    simpa [stateAddress,DirectRealWhole.finalOffset,l,hhead,hek,hjv] using hvlog

theorem inner_implements (p : Layout) (heads T K V : Nat) (r : Regions)
    (gateK hasSeed storeFinal : Bool) (x : Nat → Host.InnerInputs)
    (hp : StagedLayout.Canonical p) (hk : K≤p.K) (hv : V≤p.V)
    (hs : DirectRealSequence.InputSeparate r storeFinal)
    (hfsep : storeFinal=Bool.true → r.history≠r.final) :
    (innerIO p heads K V r gateK hasSeed storeFinal).ImplementsOn
      (innerDomain p heads T K V gateK hasSeed x) (innerFormula p heads T K V gateK hasSeed x) := by
  intro values domain before binding
  let start : BlockState := {before with undef:=fun _ _ => 0}
  let a : Inputs := ⟨before.readMem r.key,before.readMem r.value,before.readMem r.gate,before.readMem r.seed⟩
  obtain ⟨after,run,out,frame,hu⟩ := DirectRealSequence.executes p heads r .fp32 .fp32 .fp32 .fp32 .fp32
    gateK hasSeed storeFinal Bool.false a hp hs hfsep
    (LaunchPlan.plans (heads*(p.K/p.BK)*(p.V/p.BV)))
    (LaunchPlan.plans_valid _) (LaunchPlan.plans_ordered _) start rfl
    (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) (fun _ => rfl)
  obtain ⟨trace⟩ := inner_trace p r gateK hasSeed storeFinal run
  have logical := inner_logical_outputs p heads T K V r gateK hasSeed storeFinal x a hp hk hv
    (inner_binding_finite p heads T K V r gateK hasSeed storeFinal x values domain before binding)
    start after out
  refine ⟨after,trace,?_,?_,?_⟩
  · intro bounds hb
    have allocated := inner_allocated p heads K V r gateK hasSeed storeFinal values before bounds hb
    apply trace_safe trace bounds
    intro stage hm u
    obtain ⟨plan,hplan,rfl⟩ := List.mem_map.mp hm
    intro idx
    have valid := inner_plan_valid p heads plan hplan u idx
    simp only [selected_projection]
    exact RebasedSafety.source_safe p r .fp32 .fp32 .fp32 .fp32 .fp32 plan.base
      gateK hasSeed storeFinal Bool.false (u.withGridIndex idx)
      (StagedLayout.full_tile p heads _ hp valid)
      (StagedLayout.key_full p heads _ hp valid) (StagedLayout.value_full p heads _ hp valid)
      ⟨hs.key.1,hs.value.1,hs.gate.1⟩ (StagedLayout.history_noCollision p _ hp)
      bounds (StagedLayout.bounds p heads _ r gateK hasSeed storeFinal hp valid bounds allocated)
  · intro o j active
    change innerOutputActive storeFinal o at active
    by_cases hzero : o.val=0
    · let c := (historyCoordinates heads p.NT K V).symm (Fin.cast (by simp [innerIO,innerOutputArity,hzero]) j)
      simpa only [innerIO,innerOutputOffset,innerOutputRegion,innerFormula,hzero,↓reduceDIte,↓reduceIte,ChanTy.read,c]
        using logical.history c.1.val c.2.1.val c.2.2.1.val c.2.2.2.val
          c.1.isLt c.2.1.isLt c.2.2.1.isLt c.2.2.2.isLt
    · have enabled := active.resolve_left hzero
      let c := (finalCoordinates heads K V).symm (Fin.cast (by simp [innerIO,innerOutputArity,hzero]) j)
      simpa only [innerIO,innerOutputOffset,innerOutputRegion,innerFormula,hzero,↓reduceDIte,↓reduceIte,ChanTy.read,c]
        using logical.final enabled c.1.val c.2.1.val c.2.2.val c.1.isLt c.2.1.isLt c.2.2.isLt
  · intro region off outside scratch
    apply (frame region off ?_).symm
    rintro ⟨plan,hplan,idx,write⟩
    have valid := inner_plan_valid p heads plan hplan start idx
    rcases write with history|final
    · rcases history with ⟨hr,c,e,j,hc,ha,he⟩
      have hb := Address.history_bounded valid.head hc ha.1 ha.2
      have bound : off < heads*p.NT*p.K*p.V := by
        change off = _ at he
        rw [he]
        convert hb using 1 <;>
          simp only [RealHLoop.address,RealHStore.address,RealHStep.historyBase,
            hp.historyHead,hp.historyKey,hp.historyValue,keyIndex,valueIndex,Nat.mul_one] <;> ring
      have hn := scratch (0 : Fin 2) ⟨off,bound⟩ (Or.inl rfl)
      exact hn.elim (fun h => h hr) (fun h => h rfl)
    · rcases final with ⟨enabled,hr,e,j,ha,he⟩
      have hb := state_address_bound p heads _ valid.head _ _ ha.1 ha.2
      have bound : off < heads*p.K*p.V := by
        change off = _ at he
        rw [he]
        simpa only [RealHStore.address,RealHFinal.base,stateAddress,keyIndex,valueIndex,Nat.mul_one,
          Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using hb
      have hn := scratch (1 : Fin 2) ⟨off,bound⟩ (Or.inr enabled)
      exact hn.elim (fun h => h hr) (fun h => h rfl)

#axiomsClean inner_implements
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RealInterface
