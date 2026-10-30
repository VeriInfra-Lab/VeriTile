import bench.tritonbench_g.rmsnorm_fused_llama.NativeIO
import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.NativeSafety
import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.RuntimeCandidate

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.NativePublicIO
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem run_eq (A : Algebra α) (candidate : Bool) (X Y W : RegionName)
    (rows N B : Nat) (epsilon : α) (count : Nat) (s : State α) :
    runSchedule (NativeIO.schedule A candidate X Y W rows N B epsilon count) s=
      RuntimeGrid.run A X Y W rows N B epsilon s count := by
  induction count with
  | zero => rfl
  | succ k ih =>
    have jobs : NativeIO.schedule A candidate X Y W rows N B epsilon (k+1)=
        NativeIO.schedule A candidate X Y W rows N B epsilon k++
          [NativeIO.launch A candidate X Y W rows N B epsilon k] := by
      simp only [NativeIO.schedule,List.range_succ,List.map_append,List.map_singleton]
    rw [jobs,runSchedule_append,ih,RuntimeGrid.run]
    cases h : RuntimeGrid.run A X Y W rows N B epsilon s k with
    | none => rfl
    | some t =>
      simp only [Option.bind_some,runSchedule_singleton]
      cases candidate
      · rfl
      · exact RuntimeCandidate.typed A X Y W N B _

theorem schedule_safe (A : Algebra α) (candidate : Bool) (X Y W : RegionName)
    (rows N B : Nat) (epsilon : α) (hN : 0<N) (hNB : N≤B)
    (a : (NativeIO.grid A candidate X Y W rows N B epsilon).Inputs)
    (bounds : RegionName → Nat) (hb : (NativeIO.grid A candidate X Y W rows N B epsilon).InBounds bounds a)
    (jobs : List Nat) (inside : ∀ r∈jobs, r<rows) (s : State α) :
    ScheduleSafe bounds (jobs.map (NativeIO.launch A candidate X Y W rows N B epsilon)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons r rest ih =>
    refine ⟨?_,fun t _ => ih (fun r hr => inside r (by simp [hr])) t⟩
    have hr := inside r (by simp)
    apply NativeSafety.row A candidate X Y W N B epsilon (RuntimeGrid.setup rows r s) hN hNB bounds
    · intro k hk
      exact hb.2.1 (0 : Fin 2) (⟨r,hr⟩,⟨k,hk⟩,PUnit.unit) trivial
    · intro k hk
      exact hb.2.1 (1 : Fin 2) (⟨k,hk⟩,PUnit.unit) trivial
    · intro k hk
      exact hb.2.2.1 (0 : Fin 1) (⟨r,hr⟩,⟨k,hk⟩,PUnit.unit) trivial

theorem safe (A : Algebra α) (candidate : Bool) (X Y W : RegionName)
    (rows N B : Nat) (epsilon : α) (hN : 0<N) (hNB : N≤B)
    (a : (NativeIO.grid A candidate X Y W rows N B epsilon).Inputs)
    (bounds : RegionName → Nat) (hb : (NativeIO.grid A candidate X Y W rows N B epsilon).InBounds bounds a)
    (s : State α) : ScheduleSafe bounds ((NativeIO.grid A candidate X Y W rows N B epsilon).schedule a) s :=
  schedule_safe A candidate X Y W rows N B epsilon hN hNB a bounds hb _
    (fun _ h => List.mem_range.mp h) s

theorem frame (A : Algebra α) (candidate : Bool) (X Y W : RegionName)
    (rows N B : Nat) (epsilon : α) (hN : 0<N)
    (a : (NativeIO.grid A candidate X Y W rows N B epsilon).Inputs) (s t : State α)
    (hf : RuntimeGrid.Frame Y (rows*N) s t) :
    (NativeIO.grid A candidate X Y W rows N B epsilon).Frame a s t := by
  intro r off outside
  apply hf r off
  by_cases hy : r=Y
  · subst r
    right
    by_contra h
    have hi : off<rows*N := by omega
    have hr : off/N<rows := (Nat.div_lt_iff_lt_mul hN).mpr hi
    have hc := Nat.mod_lt off hN
    have h := outside.1 (0 : Fin 1) (⟨off/N,hr⟩,⟨off%N,hc⟩,PUnit.unit) trivial
    exact h.elim (fun h => h rfl) (fun h => h (by
      change off=off/N*N+off%N
      simpa only [Nat.mul_comm] using (Nat.div_add_mod off N).symm))
  · exact Or.inl hy

theorem equivalent (A : Algebra α) (X Y W : RegionName) (rows N B : Nat) (epsilon : α)
    (hN : 0<N) (hNB : N≤B) (xy : X≠Y) (wy : W≠Y) :
    (NativeIO.grid A Bool.false X Y W rows N B epsilon).Equivalent
      (NativeIO.commuted A X Y W rows N B epsilon) := by
  intro a _ bounds hb initial _
  refine ⟨safe A Bool.false X Y W rows N B epsilon hN hNB a bounds hb initial,
    safe A Bool.true X Y W rows N B epsilon hN hNB a bounds hb initial,?_⟩
  obtain ⟨final,ran,memory⟩ := RuntimeGrid.complete A X Y W rows N B epsilon initial hN hNB rows
  have hf : (NativeIO.grid A Bool.false X Y W rows N B epsilon).Frame a initial final := by
    apply frame A Bool.false X Y W rows N B epsilon hN a initial final
    intro r off outside
    rw [memory]
    exact (RuntimeGrid.graph_result A X Y W N B epsilon initial hNB xy wy rows).2 r off outside
  exact ⟨final,final,(run_eq _ Bool.false _ _ _ _ _ _ _ _ _).trans ran,
    (run_eq _ Bool.true _ _ _ _ _ _ _ _ _).trans ran,by intro o j _; rfl,hf,hf⟩

end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.NativePublicIO
