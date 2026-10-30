import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_HostGridCore

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
open VeriTile Triton FullSource
set_option maxHeartbeats 2000000

noncomputable def forwardInnerValue (base : BlockState) (X : RegionName) (N : Nat)
    (a : MemCellAddr) : ℝ :=
  RealForwardInner.expected base X (a.2/N) N (a.2%N)

noncomputable def forwardNonInnerValue (base : BlockState) (X : RegionName) (N K : Nat)
    (a : MemCellAddr) : ℝ :=
  RealForwardNonInner.expected base X (a.2/K/N) N K (a.2/K%N) (a.2%K)

theorem forward_inner_grid (input output : Storage) (base : BlockState)
    (Y X : RegionName) (M N B : Nat) (one : Bool)
    (h : RealForwardInner.ExecutionDomain one X Y N B) (hXY : X≠Y)
    (ps : List Program) (schedule : Schedule M 1 ps) :
    ∃ final, run (softmax_kernel_inner input output Y X M N B one).eraseDType.toAlgKernel
        M 1 ps base=some final ∧
      BlockState.WriteWithin (fullSlice Y (M*N)) base final ∧
      (∀ r n, r<M → n<N → final.readMem Y (r*N+n)=RealForwardInner.expected base X r N n) ∧
      final.undef=base.undef := by
  let F : Program → WriteFootprint := fun p ↦ RealForwardInner.outputFootprint Y p.1 N
  let val := forwardInnerValue base X N
  have step : ∀ p s, Preserved Y base s →
      ∃ t, exec (softmax_kernel_inner input output Y X M N B one).eraseDType.toAlgKernel (setup M 1 p s)=some t ∧
        BlockState.WriteWithin (F p) s t ∧
        (∀ a, F p a → t.readMem a.1 a.2=val a) ∧ t.undef=s.undef := by
    intro p s hs
    obtain ⟨t,he,hframe,hvalue,_,_,hundef⟩ :=
      RealForwardInner.original_all_io input output (setup M 1 p s) Y X M N B one h
    refine ⟨t,he,?_,?_,hundef⟩
    · simpa only [F,RealForwardInner.outputFootprint,setup,ite_true] using hframe
    · rintro a ⟨hy,n,hn,ha⟩
      have got := hvalue n hn
      simp only [setup,ite_true] at got
      have transport : RealForwardInner.expected (setup M 1 p s) X p.1 N n =
          RealForwardInner.expected base X p.1 N n := by
        simp only [RealForwardInner.expected,RealForwardInner.denominator,
          BlockState.readMem,setup,hs.1 X hXY]
      have stable := got.trans transport
      simpa only [val,forwardInnerValue,ha,hy,(decode_inner hn).1,(decode_inner hn).2] using stable
  obtain ⟨final,he,hf,hv,hu⟩ := run_io _ M 1 Y F val base
    (fun _ _ ha ↦ ha.1) step ps base (preserved_refl Y base)
  refine ⟨final,he,frame_mono ?_ hf,?_,hu⟩
  · rintro a ⟨p,hp,hy,n,hn,hoff⟩
    have hp' := (mem_grid.mp ((schedule p).1 hp)).1
    exact ⟨hy,hoff ▸ inner_bound hp' hn⟩
  · intro r n hr hn
    have hp : (r,0) ∈ ps := (schedule (r,0)).2 (mem_grid.mpr ⟨hr,by decide⟩)
    have got := hv (Y,r*N+n) ⟨(r,0),hp,rfl,n,hn,rfl⟩
    simpa only [val,forwardInnerValue,(decode_inner hn).1,(decode_inner hn).2] using got

theorem forward_noninner_grid (input output : Storage) (base : BlockState)
    (Y X : RegionName) (M N K B C : Nat) (one : Bool)
    (h : RealForwardNonInner.ExecutionDomain one X Y N B) (hXY : X≠Y) (hC : 0<C)
    (ps : List Program) (schedule : Schedule M (RealAddress.rounds K C) ps) :
    ∃ final, run (softmax_kernel_non_inner input output Y X M N K B C one).eraseDType.toAlgKernel
        M (RealAddress.rounds K C) ps base=some final ∧
      BlockState.WriteWithin (fullSlice Y (M*N*K)) base final ∧
      (∀ r n k, r<M → n<N → k<K →
        final.readMem Y (r*N*K+n*K+k)=RealForwardNonInner.expected base X r N K n k) ∧
      final.undef=base.undef := by
  let F : Program → WriteFootprint := fun p ↦ RealForwardNonInner.outputFootprint Y p.1 N K (p.2*C) C
  let val := forwardNonInnerValue base X N K
  have step : ∀ p s, Preserved Y base s →
      ∃ t, exec (softmax_kernel_non_inner input output Y X M N K B C one).eraseDType.toAlgKernel (setup M (RealAddress.rounds K C) p s)=some t ∧
        BlockState.WriteWithin (F p) s t ∧
        (∀ a, F p a → t.readMem a.1 a.2=val a) ∧ t.undef=s.undef := by
    intro p s hs
    obtain ⟨t,he,hframe,hvalue,_,_,hundef⟩ :=
      RealForwardNonInner.original_all_io input output
        (setup M (RealAddress.rounds K C) p s) Y X M N K B C one h
    refine ⟨t,he,?_,?_,hundef⟩
    · simpa only [F,RealForwardNonInner.outputFootprint,setup,ite_true,
        Nat.reduceEqDiff,ite_false] using hframe
    · rintro a ⟨hy,n,k,hn,hk,hK,ha⟩
      have got := hvalue n k hn hk (by simpa [setup] using hK)
      simp only [setup,ite_true,Nat.reduceEqDiff,ite_false] at got
      have transport : RealForwardNonInner.expected (setup M (RealAddress.rounds K C) p s) X p.1 N K n (p.2*C+k) =
          RealForwardNonInner.expected base X p.1 N K n (p.2*C+k) := by
        simp only [RealForwardNonInner.expected,RealForwardNonInner.denominator,
          BlockState.readMem,setup,hs.1 X hXY]
      have stable := got.trans transport
      simpa only [val,forwardNonInnerValue,ha,hy,(decode_noninner hn hK).1,
        (decode_noninner hn hK).2.1,(decode_noninner hn hK).2.2] using stable
  obtain ⟨final,he,hf,hv,hu⟩ := run_io _ M (RealAddress.rounds K C) Y F val base
    (fun _ _ ha ↦ ha.1) step ps base (preserved_refl Y base)
  refine ⟨final,he,frame_mono ?_ hf,?_,hu⟩
  · rintro a ⟨p,hp,hy,n,k,hn,_,hk,hoff⟩
    have hp' := (mem_grid.mp ((schedule p).1 hp)).1
    exact ⟨hy,hoff ▸ RealAddress.non_inner_bound hp' hn hk⟩
  · intro r n k hr hn hk
    have hp : (r,k/C) ∈ ps := (schedule (r,k/C)).2 (mem_grid.mpr ⟨hr,owner_lt hC hk⟩)
    have lane := Nat.mod_lt k hC
    have off := owner_offset k C
    have got := hv (Y,r*N*K+n*K+k) ⟨(r,k/C),hp,rfl,n,k%C,hn,lane,by simpa [off] using hk,by simp [off]⟩
    simpa only [val,forwardNonInnerValue,(decode_noninner hn hk).1,
      (decode_noninner hn hk).2.1,(decode_noninner hn hk).2.2] using got

#axiomsClean forward_inner_grid
#axiomsClean forward_noninner_grid
#specNonCircular forwardInnerValue avoiding [run,exec]
#specNonCircular forwardNonInnerValue avoiding [run,exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
