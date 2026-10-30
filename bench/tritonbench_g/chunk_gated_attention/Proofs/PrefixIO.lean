import bench.tritonbench_g.chunk_gated_attention.Proofs.RealStages

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RealInterface
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealCumProgram RebasedCumAccess RebasedCumExecution
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

theorem prefix_offset_bound (p : Layout) (heads : Nat) (l : Launch) (c : Nat)
    (hp : CumulativeHost.Canonical p) (hl : CumulativeHost.Valid p heads l c)
    (i : TileIndex [p.BT,p.BS]) :
    RealCumExecution.offset (tileLayout p l c) localLaunch 0 i < heads*p.T*p.S := by
  rw [CumulativeHost.local_offset,hp.head,hp.time,hp.channel,Nat.mul_one]
  simpa only [Nat.mul_assoc,Nat.add_mul] using
    Address.flatten3_bounded hl.head (CumulativeHost.time_bound p heads l c hp hl i.1)
      (CumulativeHost.feature_bound p heads l c hp hl i.2.1)

theorem prefix_expected_local (p : Layout) (heads : Nat) (l : Launch) (c : Nat)
    (hp : CumulativeHost.Canonical p) (hl : CumulativeHost.Valid p heads l c)
    (a b : Nat → ℝ) (hab : ∀ off, off < heads*p.T*p.S → a off=b off)
    (i : TileIndex [p.BT,p.BS]) :
    RealCumExecution.expected (tileLayout p l c) localLaunch 0 a i =
      RealCumExecution.expected (tileLayout p l c) localLaunch 0 b i := by
  apply Finset.sum_congr rfl
  intro t ht
  exact hab _ (prefix_offset_bound p heads l c hp hl (t,i.2))

theorem prefix_implements (p : Layout) (heads T S : Nat) (input output : RegionName)
    (x : Nat → Nat → Nat → ℝ) (hp : CumulativeHost.Canonical p)
    (hbs : p.BS=16 ∨ p.BS=32 ∨ p.BS=64) (hT : T≤p.T) (hS : S≤p.S)
    (hsep : input≠output) :
    (prefixIO p heads T S input output).ImplementsOn (prefixDomain p heads T S x)
      (prefixFormula p heads T S x) := by
  intro values domain before binding
  let start : BlockState := {before with undef:=fun _ _ => 0}
  have reads (off : Nat) (ho : off < heads*p.T*p.S) :
      before.readMem input off = CumulativeHost.materialized p T S x off :=
    (binding (0 : Fin 1) ⟨off,ho⟩ trivial).trans (domain ⟨off,ho⟩)
  obtain ⟨after,run,out,frame,hu⟩ := DirectCumulativeSequence.executes p heads input output Bool.false .fp32 .fp32
    (before.readMem input) hp hsep (DirectCumulativeWhole.plans p heads)
    (AutotunePlan.valid _ _) (AutotunePlan.ordered _ _ (DirectCumulativeWhole.total_le_maximum p heads hbs))
    start rfl (fun _ => rfl)
  obtain ⟨trace⟩ := prefix_trace p heads input output run
  refine ⟨after,trace,?_,?_,?_⟩
  · intro bounds hb
    have hi : heads*p.T*p.S ≤ bounds input := capacity_bound (fun j => hb.1 (0 : Fin 1) j trivial)
    have ho : heads*p.T*p.S ≤ bounds output := capacity_bound (fun j => hb.2.2 (0 : Fin 1) j trivial)
    apply trace_safe trace bounds
    intro stage hm u
    obtain ⟨plan,_,rfl⟩ := List.mem_map.mp hm
    intro idx
    simp only [selected_projection]
    apply RebasedCumSafety.source_safe p input output heads plan.base Bool.false .fp32 .fp32 bounds
    intro enabled
    exact CumulativeHost.bounds p heads _ _ input output hp
      (CumulativeHost.decoded_valid p heads plan.base (u.withGridIndex idx) enabled) bounds hi ho
  · intro o j _
    let c := (prefixCoordinates heads T S).symm j
    obtain ⟨plan,hplan,idx,i,active,hhead,hti,hfi⟩ :=
      DirectCumulativeWhole.coverage p heads c.1.val c.2.1.val c.2.2.val start hp hbs
        c.1.isLt (lt_of_lt_of_le c.2.1.isLt hT) (lt_of_lt_of_le c.2.2.isLt hS)
    have valid := CumulativeHost.decoded_valid p heads plan.base (start.withGridIndex idx) active
    have hr := out plan hplan idx active i
    have localEq := prefix_expected_local p heads _ _ hp valid (before.readMem input)
      (CumulativeHost.materialized p T S x) reads i
    have logical := CumulativeHost.expected_eq p heads T S _ _ hp valid x i
      (by simpa [hti] using c.2.1.isLt) (by simpa [hfi] using c.2.2.isLt)
    have result := hr.trans (localEq.trans logical)
    simpa only [prefixIO,prefixFormula,ChanTy.read,q,CumulativeHost.local_offset,hhead,hti,hfi,c] using result
  · intro region off outside scratch
    apply (frame region off ?_).symm
    rintro ⟨plan,hplan,idx,active,hr,i,hi,he⟩
    have valid := CumulativeHost.decoded_valid p heads plan.base (start.withGridIndex idx) active
    have bound := prefix_offset_bound p heads _ _ hp valid i
    change off = _ at he
    have hexcluded := scratch (0 : Fin 1) ⟨off,by change off < heads*p.T*p.S; rw [he]; exact bound⟩ trivial
    exact hexcluded.elim (fun hn => hn hr) (fun hn => hn rfl)

#axiomsClean prefix_implements
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RealInterface
