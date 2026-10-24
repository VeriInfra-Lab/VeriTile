import bench.repairs.tritonbench_g.rmsnorm_fused_llama.IO
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.KernelCandidate
import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.FullContracts

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.RealPublicIO
open VeriTile Triton
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

theorem row (candidate : Bool) (s : BlockState) (X Y W : RegionName) (N B : Nat) (eps : ℝ)
    (hB : 0<B) (hXY : X≠Y) (hWY : W≠Y) :
    ∃ t, exec (IO.source candidate X Y W N B eps).toAlgKernel s=some t ∧
      (∀ k : Fin N, t.readMem Y (s.pids 0*N+k.val)=
        RealGrid.value (s.readMem X) (s.readMem W) (s.pids 0) N eps k.val) ∧
      VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.General.OutputFrame s t Y N N := by
  obtain ⟨t,ran,values,frame⟩ := RealGrid.row_complete s X Y W N B eps hB hXY hWY
  refine ⟨t,?_,values,frame⟩
  cases candidate
  · exact ran
  · exact (KernelCandidate.real X Y W N B eps s).trans ran

/-- Compose the existing arbitrary-state row theorem with the shared launch
schedule. No kernel execution is reproved and no register precondition added. -/
theorem grid (candidate : Bool) (X Y W : RegionName) (rows N B : Nat) (eps : ℝ)
    (initial : BlockState) (hB : 0<B) (hXY : X≠Y) (hWY : W≠Y) (count : Nat) :
    ∃ t, runAlgorithmSchedule (IO.source candidate X Y W N B eps).toAlgKernel
        (IO.schedule rows count) initial=some t ∧
      (∀ r, r<count → ∀ k, k<N → t.readMem Y (r*N+k)=
        RealGrid.value (initial.readMem X) (initial.readMem W) r N eps k) ∧
      RealGrid.Frame Y (count*N) initial t := by
  induction count with
  | zero => exact ⟨initial,rfl,by intro r hr; omega,by intro R o _; rfl⟩
  | succ c ih =>
    obtain ⟨s,hs,hval,hframe⟩ := ih
    obtain ⟨t,ht,hread,hf⟩ := row candidate ((IO.launch rows c).initial s) X Y W N B eps hB hXY hWY
    have hx : s.mem X=initial.mem X := funext (fun o => hframe X o (Or.inl hXY))
    have hw : s.mem W=initial.mem W := funext (fun o => hframe W o (Or.inl hWY))
    refine ⟨t,?_,?_,?_⟩
    · have jobs : IO.schedule rows (c+1)=IO.schedule rows c++[IO.launch rows c] := by
        simp only [IO.schedule,List.range_succ,List.map_append,List.map_singleton]
      rw [jobs,runAlgorithmSchedule_append,hs,Option.bind_some,runAlgorithmSchedule_singleton]
      exact ht
    · intro r hr k hk
      by_cases same : r=c
      · subst r
        simpa [IO.launch,AlgorithmLaunch.initial,RealGrid.value,BlockState.readMem,hx,hw] using hread ⟨k,hk⟩
      · have before : r*N+k<c*N := by
          calc r*N+k < (r+1)*N := by nlinarith
               _ ≤ c*N := Nat.mul_le_mul_right N (by omega)
        have mem := hf Y (r*N+k) (Or.inr (fun j => by
          change r*N+k≠c*N+j.val
          omega))
        have read : t.readMem Y (r*N+k)=s.readMem Y (r*N+k) := by
          simp only [BlockState.readMem,mem,AlgorithmLaunch.initial_mem]
        exact read.trans (hval r (by omega) k hk)
    · intro R o ho
      have mem := hf R o (by
        rcases ho with different|outside
        · exact Or.inl different
        · right
          intro k
          change o≠c*N+k.val
          have hk := k.isLt
          have eq : (c+1)*N=c*N+N := by simp [Nat.add_mul]
          omega)
      rw [mem,AlgorithmLaunch.initial_mem]
      apply hframe R o
      rcases ho with different|outside
      · exact Or.inl different
      · exact Or.inr (le_trans (Nat.mul_le_mul_right N (by omega : c≤c+1)) outside)

theorem safe (candidate : Bool) (X Y W : RegionName) (rows N B : Nat) (eps : ℝ)
    (hB : 0<B) (hXY : X≠Y) (hWY : W≠Y) (u : BlockState)
    (a : (IO.grid candidate X Y W rows N B eps).Ctx) (bounds : RegionBounds)
    (hb : (IO.grid candidate X Y W rows N B eps).RegionalBounds bounds u a) :
    AlgorithmScheduleSafe (IO.source candidate X Y W N B eps).toAlgKernel bounds
      (IO.schedule rows rows) u := by
  apply algorithmScheduleSafe_of_forall
  intro job member s
  obtain ⟨r,hr,rfl⟩ := List.mem_map.mp member
  have inside : r<rows := List.mem_range.mp hr
  have flat (k : Fin N) : r*N+k.val<rows*N := by
    calc r*N+k.val < (r+1)*N := by nlinarith [k.isLt]
         _ ≤ rows*N := Nat.mul_le_mul_right N inside
  have hx (k : Fin N) := hb.1 (0 : Fin 2) ⟨r*N+k.val,flat k⟩ trivial
  have hw (k : Fin N) := hb.1 (1 : Fin 2) k trivial
  have hy (k : Fin N) := hb.2.1 (0 : Fin 1) ⟨r*N+k.val,flat k⟩ trivial
  have result := VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.General.typed_full_trace_safe
    .fp32 .fp32 .fp32 candidate bounds ((IO.launch rows r).initial s) X Y W N N B eps hB hXY hWY hx hw hy
  cases candidate <;> exact result

theorem implements (candidate : Bool) (X Y W : RegionName) (rows N B : Nat) (eps : ℝ)
    (hB : 0<B) (hXY : X≠Y) (hWY : W≠Y) :
    (IO.grid candidate X Y W rows N B eps).RegionalScheduledImplements
      (fun _ => IO.schedule rows rows) (Reference.real candidate X Y W rows N B eps) := by
  apply UKernelIO.RegionalScheduledImplements.intro
  · intro u a _ bounds hb
    exact safe candidate X Y W rows N B eps hB hXY hWY u a bounds hb
  · intro u a hin
    obtain ⟨v,ran,values,frame⟩ := grid candidate X Y W rows N B eps u hB hXY hWY rows
    refine ⟨v,ran,?_,?_⟩
    · intro o j _
      have hj : j.val<rows*N := j.isLt
      have positive : 0<N := by nlinarith [hj]
      have hr : j.val/N<rows := (Nat.div_lt_iff_lt_mul positive).mpr j.isLt
      have hc : j.val%N<N := Nat.mod_lt _ positive
      have index : j.val/N*N+j.val%N=j.val := by simpa only [Nat.mul_comm] using Nat.div_add_mod j.val N
      have hx (k : Fin N) : u.readMem X (j.val/N*N+k.val)=
          IO.vector (a (0 : Fin 2)) (j.val/N*N+k.val) := by
        have bound : j.val/N*N+k.val<rows*N := by
          calc j.val/N*N+k.val < (j.val/N+1)*N := by nlinarith [k.isLt]
               _ ≤ rows*N := Nat.mul_le_mul_right N hr
        simpa [IO.grid,ChanTy.read,IO.vector,bound] using hin (0 : Fin 2) ⟨_,bound⟩ trivial
      have hw : u.readMem W (j.val%N)=IO.vector (a (1 : Fin 2)) (j.val%N) := by
        simpa [IO.grid,ChanTy.read,IO.vector,hc] using hin (1 : Fin 2) ⟨_,hc⟩ trivial
      change v.readMem Y j.val=Reference.real candidate X Y W rows N B eps a o j
      rw [←index,values _ hr _ hc]
      change RealGrid.value (u.readMem X) (u.readMem W) _ N eps _=
        RealGrid.value (IO.vector (a (0 : Fin 2))) (IO.vector (a (1 : Fin 2))) _ N eps _
      simp only [index,RealGrid.value,hw]
      have self := hx ⟨j.val%N,hc⟩
      simp only [Fin.val_mk,index] at self
      rw [self]
      have sum : (∑ k : Fin N, u.readMem X (j.val/N*N+k.val)^2)=
          ∑ k : Fin N, IO.vector (a (0 : Fin 2)) (j.val/N*N+k.val)^2 := by
        apply Finset.sum_congr rfl
        intro k _
        rw [hx k]
      rw [sum]
    · intro r off outside _
      apply frame r off
      by_cases hy : r=Y
      · subst r
        right
        by_contra h
        have hi : off<rows*N := by omega
        exact (outside (0 : Fin 1) ⟨off,hi⟩ trivial).elim (fun h => h rfl) (fun h => h rfl)
      · exact Or.inl hy

end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.RealPublicIO
