import bench.repairs.tritonbench_g.matmul_kernel.IO

namespace VeriTile.Bench.Repairs.MatmulKernel.PublicIO
open VeriTile.Triton FP.Structural KernelSource KernelScalar
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 32000
variable {α : Type} [Inhabited α]

theorem initial_eq (p : Config) (M : Algebra α) (D : DotAcc α) (pid : Program) (s : State α) :
    (invocation p M D pid).initial s=atProgram s pid := by
  simp only [Invocation.initial, invocation, bindRegisters, List.foldl_nil,
    Bool.false_eq_true, if_false, Option.getD_none, atProgram]
  congr 1
  funext d
  by_cases h0 : d=0
  · subst d; rfl
  by_cases h1 : d=1
  · subst d; rfl
  have hd : ¬d<2 := by omega
  simp [h0,h1,hd]

theorem invocation_run (p : Config) (M : Algebra α) (D : DotAcc α)
    (pid : Program) (s : State α) :
    (invocation p M D pid).run s=exec (backend M D) (KernelSource.source p) (atProgram s pid) := by
  unfold Invocation.run
  rw [initial_eq]
  rfl

theorem schedule_eq (p : Config) (M : Algebra α) (D : DotAcc α)
    (jobs : List Program) (s : State α) :
    runSchedule (jobs.map (invocation p M D)) s=KernelSource.execute M D p jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid jobs ih =>
    simp only [List.map_cons, runSchedule, invocation_run, KernelSource.execute]
    cases h : exec (backend M D) (KernelSource.source p) (atProgram s pid) <;> simp [ih]

private theorem launched_nonempty (p : Config) (pid : Program) (hp : pid∈programs p) :
    0<p.M ∧ 0<p.N := by
  have zeroceil (b : Nat) : ceil 0 b=0 := by
    by_cases hz : b=0
    · simp [ceil,hz]
    · exact Nat.div_eq_of_lt (by omega)
  have hm : p.M≠0 := by intro h;simp [programs,h,zeroceil] at hp
  have hn : p.N≠0 := by intro h;simp [programs,h,zeroceil] at hp
  omega

theorem schedule_safe (p : Config) (M : Algebra α) (D : DotAcc α)
    (bounds : RegionName → Nat)
    (ha : ∀ r k, r<p.M → k<p.K → p.A.addr r k<bounds "A")
    (hb : ∀ k n, k<p.K → n<p.N → p.B.addr k n<bounds "B")
    (hc : ∀ r n, r<p.M → n<p.N → p.C.addr r n<bounds "C")
    (jobs : List Program) (legal : ∀ pid∈jobs,pid∈programs p) (s : State α) :
    ScheduleSafe bounds (jobs.map (invocation p M D)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons pid jobs ih =>
    refine ⟨?_, fun t _ => ih (fun q hq => legal q (by simp [hq])) t⟩
    change TraceSafe (backend M D) bounds (KernelSource.source p) ((invocation p M D pid).initial s)
    rw [initial_eq]
    have nonempty := launched_nonempty p pid (legal pid (by simp))
    apply KernelSafety.source_safe M D p bounds (atProgram s pid)
    · intro q i hi
      have bound := ha (row p (atProgram s pid) i.1 % p.M) (q*p.BK+i.2.1.val)
        (Nat.mod_lt _ nonempty.1) (by omega)
      convert bound using 1
      simp only [aOffset,aRow,View.addr,BK]
      ring
    · intro q i hi
      have bound := hb (q*p.BK+i.1.val) (col p (atProgram s pid) i.2.1 % p.N)
        (by omega) (Nat.mod_lt _ nonempty.2)
      convert bound using 1
      simp only [bOffset,bCol,View.addr,BK]
      ring
    · intro i hi
      have bound := hc (row p (atProgram s pid) i.1) (col p (atProgram s pid) i.2.1)
        hi.1 hi.2
      simpa only [cOffset,View.addr,Nat.mul_comm] using bound

/-- Assemble the shared typed IO from the preserved complete source result.
The callback here is a proof obligation in a supporting lemma, never a field
of the IO or an assumption of the public specification. -/
theorem implements (p : Config) (M : Algebra α) (D : DotAcc α)
    (f : (Nat → Nat → α) → (Nat → Nat → α) → Nat → Nat → α)
    (result : ∀ (s : State α) (a b : Nat → Nat → α),
      (∀ r k, r<p.M → k<p.K → read (s.mem "A" (p.A.addr r k))=a r k) →
      (∀ k n, k<p.K → n<p.N → read (s.mem "B" (p.B.addr k n))=b k n) →
      ∃ t, KernelSource.launch M D p s=some t ∧
        (∀ w, p.Active w → t.mem "C" (p.C.addr w.1 w.2)=stored (f a b w.1 w.2)) ∧
        (∀ (region : RegionName) offset,
          (region≠"C" ∨ ∀ w,p.Active w → offset≠p.C.addr w.1 w.2) → t.mem region offset=s.mem region offset)) :
    (io p M D).Implements (expected p f) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    apply schedule_safe p M D bounds
    · intro r k hr hk
      exact hb.2.1 ⟨0,by simp [io,inputs]⟩ (⟨r,hr⟩,⟨k,hk⟩,PUnit.unit) trivial
    · intro k n hk hn
      exact hb.2.1 ⟨1,by simp [io,inputs]⟩ (⟨k,hk⟩,⟨n,hn⟩,PUnit.unit) trivial
    · intro r n hr hn
      exact hb.2.2.1 ⟨0,by simp [io,outputs]⟩ (⟨r,hr⟩,⟨n,hn⟩,PUnit.unit) trivial
    · exact fun _ h => h
  · intro vals initial hin
    have ha : ∀ r k,r<p.M → k<p.K → read (initial.mem "A" (p.A.addr r k))=arrayA p vals r k := by
      intro r k hr hk
      simpa [ProgramIO.InputBinding,io,inputs,arrayA,hr,hk,mapValue,KernelSource.read] using
        hin ⟨0,by simp [io,inputs]⟩ (⟨r,hr⟩,⟨k,hk⟩,PUnit.unit) trivial
    have hb : ∀ k n,k<p.K → n<p.N → read (initial.mem "B" (p.B.addr k n))=arrayB p vals k n := by
      intro k n hk hn
      simpa [ProgramIO.InputBinding,io,inputs,arrayB,hk,hn,mapValue,KernelSource.read] using
        hin ⟨1,by simp [io,inputs]⟩ (⟨k,hk⟩,⟨n,hn⟩,PUnit.unit) trivial
    obtain ⟨final,ran,output,frame⟩ := result initial (arrayA p vals) (arrayB p vals) ha hb
    refine ⟨final,?_,?_,?_⟩
    · exact (schedule_eq p M D (programs p) initial).trans ran
    · intro port lane _
      fin_cases port
      have value := output (lane.1.val,lane.2.1.val) ⟨lane.1.isLt,lane.2.1.isLt⟩
      change mapCell id (final.mem "C" (p.C.addr lane.1.val lane.2.1.val))=
        .mk .real (f (arrayA p vals) (arrayB p vals) lane.1.val lane.2.1.val)
      rw [value]
      rfl
    · intro region offset outside
      apply frame region offset
      by_cases other : region≠"C"
      · exact Or.inl other
      · right
        intro w hw
        have miss := outside.1 ⟨0,by simp [io,outputs]⟩
          (⟨w.1,hw.1⟩,⟨w.2,hw.2⟩,PUnit.unit) trivial
        exact miss.resolve_left (by simpa using other)

#axiomsClean schedule_eq
#axiomsClean schedule_safe
#axiomsClean implements
end
end VeriTile.Bench.Repairs.MatmulKernel.PublicIO
