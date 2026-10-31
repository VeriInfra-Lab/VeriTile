import bench.tritonbench_g.triton_matmul.IO
import bench.tritonbench_g.triton_matmul.Proofs.SourceSafety
import bench.tritonbench_g.triton_matmul.Proofs.Native
import bench.tritonbench_g.triton_matmul.Proofs.KernelRealModel
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.PublicIO
open VeriTile.Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 32000
variable {α : Type} [Inhabited α]

theorem initial_eq (g : Geometry) (M : Algebra α) (D : DotAcc α) (pid : Nat) (s : State α) :
    (invocation g M D pid).initial s=instanceState g.request pid s := rfl

theorem schedule_eq (g : Geometry) (M : Algebra α) (D : DotAcc α)
    (ids : List Nat) (s : State α) :
    runSchedule (ids.map (invocation g M D)) s=runPrograms M D g.request ids s := by
  induction ids generalizing s with
  | nil => rfl
  | cons pid ids ih =>
    simp only [List.map_cons,runSchedule,runPrograms]
    change (exec (backend M D) (kernel g.request) (instanceState g.request pid s)).bind _=_
    cases h : exec (backend M D) (kernel g.request) (instanceState g.request pid s) <;> simp [ih]

theorem schedule_safe (g : Geometry) (M : Algebra α) (D : DotAcc α)
    (bounds : RegionName → Nat)
    (ha : ∀ r k,r<g.rows → k<g.inner → r*g.strideAM+k*g.strideAK<bounds g.A)
    (hb : ∀ k n,k<g.inner → n<g.columns → k*g.strideBK+n*g.strideBN<bounds g.B)
    (hc : ∀ r n,r<g.rows → n<g.columns → max g.columns 1*r+n<bounds g.C)
    (ids : List Nat) (legal : ∀ pid∈ids,pid<programs g.request) (s : State α) :
    ScheduleSafe bounds (ids.map (invocation g M D)) s := by
  induction ids generalizing s with
  | nil => trivial
  | cons pid ids ih =>
    refine ⟨?_,fun t _ => ih (fun q hq => legal q (by simp [hq])) t⟩
    have active := legal pid (by simp)
    have hm : 0<g.rows := by
      by_contra h
      have hz : g.rows=0 := by omega
      simp [programs,tileRows,Geometry.request,hz] at active
    have hn : 0<g.columns := by
      by_contra h
      have hz : g.columns=0 := by omega
      simp [programs,tileColumns,Geometry.request,hz] at active
    change TraceSafe (backend M D) bounds (Source.typed g.request.parameters g.storage)
      (instanceState g.request pid s)
    apply SourceSafety.source_safe
    · exact SourceLaunch.request_m g.request
    · exact SourceLaunch.request_n g.request
    · exact SourceLaunch.request_first g.request pid s active
    · exact SourceLaunch.request_count g.request
    · intro q i hk
      have rowBound : aRow g.request.parameters (instanceState g.request pid s) i.1<g.rows := by
        unfold aRow
        split <;> assumption
      have bound := ha (aRow g.request.parameters (instanceState g.request pid s) i.1)
        (q*BK g.request.parameters+i.2.1.val) rowBound (by simpa [Geometry.request,Request.parameters,SourceScope.parameters,Nat.add_comm] using hk)
      convert bound using 1
      simp only [aOffset,Geometry.request,Request.parameters,SourceScope.parameters,BK]
      ring
    · intro q i hk
      have colBound : bCol g.request.parameters (instanceState g.request pid s) i.2.1<g.columns := by
        unfold bCol
        split <;> assumption
      have bound := hb (q*BK g.request.parameters+i.1.val)
        (bCol g.request.parameters (instanceState g.request pid s) i.2.1)
        (by simpa [Geometry.request,Request.parameters,SourceScope.parameters,Nat.add_comm] using hk) colBound
      convert bound using 1
      simp only [bOffset,Geometry.request,Request.parameters,SourceScope.parameters,BK]
      ring
    · intro i hi
      simpa [cOffset,Geometry.request,Request.parameters,SourceScope.parameters] using
        hc (row g.request.parameters (instanceState g.request pid s) i.1)
          (col g.request.parameters (instanceState g.request pid s) i.2.1) hi.1 hi.2

private theorem read_binding (st : Storage) (c : Cell α) (v : Value α (cellType st))
    (h : c.read (cellType st)=mapValue id (cellType st) v) : read st c=payload st v := by
  cases st <;> exact h

/-- Assemble the shared source judgment from the retained complete grid theorem
and finite logical input congruence. No execution/safety callback is an IO field. -/
theorem implements (g : Geometry) (M : Algebra α) (D : DotAcc α) (Q : Numerics α)
    (law : PrimitiveLaws M D Q g.storage) (ids : List Nat)
    (order : ids.Perm (List.range (programs g.request))) (sepA : g.C≠g.A) (sepB : g.C≠g.B)
    (f : (Fin g.rows → Fin g.inner → α) → (Fin g.inner → Fin g.columns → α) → Fin g.rows → Fin g.columns → α)
    (formula : ∀ (initial : State α) (a : Fin g.rows → Fin g.inner → α) (b : Fin g.inner → Fin g.columns → α),
      (∀ r k,inputA g.request initial r.val k.val=a r k) →
      (∀ k n,inputB g.request initial k.val n.val=b k n) →
      ∀ r n,reference Q g.request initial r.val n.val=f a b r n) :
    (io g M D ids).Implements (expected g f) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bound initial _
    apply schedule_safe g M D bounds
    · intro r k hr hk
      exact bound.2.1 ⟨0,by simp [io,inputs]⟩ (⟨r,hr⟩,⟨k,hk⟩,()) trivial
    · intro k n hk hn
      exact bound.2.1 ⟨1,by simp [io,inputs]⟩ (⟨k,hk⟩,⟨n,hn⟩,()) trivial
    · intro r n hr hn
      exact bound.2.2.1 ⟨0,by simp [io,outputs]⟩ (⟨r,hr⟩,⟨n,hn⟩,()) trivial
    · exact fun pid hpid => List.mem_range.mp (order.mem_iff.mp hpid)
  · intro vals initial hin
    have ha : ∀ r k,inputA g.request initial r.val k.val=arrayA g vals r k := by
      intro r k
      exact read_binding g.storage _ _ (hin ⟨0,by simp [io,inputs]⟩ (r,k,()) trivial)
    have hb : ∀ k n,inputB g.request initial k.val n.val=arrayB g vals k n := by
      intro k n
      exact read_binding g.storage _ _ (hin ⟨1,by simp [io,inputs]⟩ (k,n,()) trivial)
    obtain ⟨final,ran,values,frame⟩ := grid_order_correct M D Q g.request law initial ids order sepA sepB
    refine ⟨final,?_,?_,?_⟩
    · exact (schedule_eq g M D ids initial).trans ran
    · intro port lane _
      fin_cases port
      have value := values lane.1.val lane.1.isLt lane.2.1.val lane.2.1.isLt
      rw [formula initial _ _ ha hb lane.1 lane.2.1] at value
      change mapCell id (final.mem g.C (outputOffset g.request lane.1.val lane.2.1.val))=
        .mk (cellType g.storage) (tagged g.storage (f (arrayA g vals) (arrayB g vals) lane.1 lane.2.1))
      have cell : final.mem g.C (outputOffset g.request lane.1.val lane.2.1.val)=
          stored g.storage (f (arrayA g vals) (arrayB g vals) lane.1 lane.2.1) := value
      rw [cell]
      cases g.storage <;> rfl
    · intro region offset outside
      apply frame region offset
      rintro ⟨hregion,r,hr,n,hn,hoff⟩
      have miss := outside.1 ⟨0,by simp [io,outputs]⟩ (⟨r,hr⟩,⟨n,hn⟩,()) trivial
      exact (miss.resolve_left (fun h => h hregion)) hoff

#axiomsClean schedule_safe
#axiomsClean implements
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.PublicIO
