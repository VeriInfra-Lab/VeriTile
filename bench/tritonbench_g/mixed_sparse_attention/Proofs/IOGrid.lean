import bench.tritonbench_g.mixed_sparse_attention.Proofs.IOAccessBounds

namespace VeriTile.Bench.Repairs.MixedSparseAttention.KernelIO
open VeriTile Triton FP.Structural
open Whole002 Source Whole Checked
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

theorem schedule_safe (M : Algebra α) (D : Dot α) (c : Config)
    (sep : Separated c) (layout : Contiguous c) (hbn : 0<c.BN)
    (work : List Program) (covered : ∀ p∈work,p∈programs (grid c))
    (xs : Values α c) (hd : domain c xs) (capacity : RegionName → Nat)
    (hb : (io M D c work).InBounds capacity xs) (s : State α) (h : Reads c xs s) :
    ScheduleSafe capacity (work.map (invocation M D c)) s := by
  induction work generalizing s with
  | nil=>trivial
  | cons p ps ih=>
    constructor
    · exact IOSourceSafety.kernel_safe M D c (start (grid c) p s) hbn capacity
        (actual_bounds M D c layout (p::ps) xs s h hd capacity hb p (covered p (by simp)))
    · intro t ran
      obtain ⟨next,he,hm⟩:=source_memory M D c (start (grid c) p s) hbn
      change FP.Structural.exec (backend M D) (kernel c) (start (grid c) p s)=some t at ran
      have eq:=Option.some.inj (ran.symm.trans he)
      subst t
      have stable : Reads c xs next := KernelIO.Reads.stable c xs s next h sep (by
        intro r off different
        rw [hm]
        exact Source.frame M D c (start (grid c) p s) (finalValues M D c (start (grid c) p s))
          r off (Or.inl different))
      exact ih (fun q hq=>covered q (by simp [hq])) hb next stable

theorem frame_of_result (M : Algebra α) (D : Dot α) (c : Config) (work : List Program)
    (hbm : 0<c.BM) (xs : Values α c) (s final : State α) (h : Reads c xs s)
    (result : KernelGrid.Result (fun _=>M) (fun _=>D) c hbm s final) :
    (io M D c work).Frame xs s final := by
  intro r off outside
  apply result.frame
  rintro ⟨rfl,q,hq,hoff⟩
  let j : TileIndex [c.Z*c.H,c.T,c.BD] := (q.1,q.2.1,q.2.2,())
  have active : KernelIO.active c xs j := by
    simpa [KernelIO.active,j,KernelGrid.live,Raw.row,Raw.hz,h.length] using hq
  rcases outside.1 (0 : Fin 1) j active with different|different
  · exact different rfl
  · exact different hoff

theorem whole (M : Algebra α) (D : Dot α) (c : Config)
    (sep : Separated c) (layout : Contiguous c) (hbm : 0<c.BM) (hbn : 0<c.BN)
    (work : List Program) (order : work.Perm (programs (grid c)))
    (xs : Values α c) (hd : domain c xs) (s : State α) (h : Reads c xs s) :
    (∀ bounds,(io M D c work).InBounds bounds xs →
      ScheduleSafe bounds (work.map (invocation M D c)) s) ∧
    ∃ final,runSchedule (work.map (invocation M D c)) s=some final ∧
      KernelGrid.Result (fun _=>M) (fun _=>D) c hbm s final ∧
      (io M D c work).Frame xs s final := by
  constructor
  · intro bounds hb
    exact schedule_safe M D c sep layout hbn work (fun _ hp=>order.mem_iff.mp hp) xs hd bounds hb s h
  · obtain ⟨final,ran,result⟩:=KernelGrid.source (fun _=>M) (fun _=>D)
      c sep layout s (KernelIO.Reads.lengths c xs s h hd) hbm hbn work order
    exact ⟨final,(schedule_run M D c work s).trans ran,result,
      frame_of_result M D c work hbm xs s final h result⟩

#axiomsClean whole
end
end VeriTile.Bench.Repairs.MixedSparseAttention.KernelIO
