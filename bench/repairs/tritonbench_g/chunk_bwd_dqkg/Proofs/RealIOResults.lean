import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.RealFormulaBinding
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.RealIOResults
open VeriTile Triton Whole AddressBounds
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

theorem preparation (c : Config) (a : CTA c) (s : BlockState) :
    (IO.realJob c a).initial s=Whole.prepare c s a := rfl

theorem schedule (c : Config) (order : List (CTA c)) (s : BlockState) :
    runPreparedAlgorithmSchedule (order.map (IO.realJob c)) s=Whole.runGrid c order s := by
  induction order generalizing s with
  | nil=>rfl
  | cons a rest ih=>
    simp only [List.map_cons,runPreparedAlgorithmSchedule,Whole.runGrid]
    change (VeriTile.Triton.exec (kernel c).eraseDType.toAlgKernel (Whole.prepare c s a)).bind
      (runPreparedAlgorithmSchedule (rest.map (IO.realJob c))) =
      (VeriTile.Triton.exec (kernel c).eraseDType.toAlgKernel (Whole.prepare c s a)).bind (Whole.runGrid c rest)
    congr 1
    funext t
    exact ih t

theorem capacity_of_windows (c : Config) (bounds : RegionName → Nat)
    (hin : ∀ i,IO.inputSize c i≤bounds (IO.inputRegion i))
    (hout : ∀ o,IO.outputSize c o≤bounds (IO.outputRegion o)) : Capacity c bounds := by
  intro r
  by_cases hq : r="q"
  · subst r; exact hin 0
  by_cases hk : r="k"
  · subst r; exact hin 1
  by_cases hv : r="v"
  · subst r; exact hin 2
  by_cases hh : r="h"
  · subst r; exact hin 3
  by_cases hg : r="g"
  · subst r; exact hin 4
  by_cases hd : r="do"
  · subst r; exact hin 5
  by_cases hf : r="dh"
  · subst r; exact hin 6
  by_cases hdq : r="dq"
  · subst r; exact hout 0
  by_cases hdk : r="dk"
  · subst r; exact hout 1
  by_cases hdg : r="dg"
  · subst r; exact hout 2
  simp [extent,hq,hk,hv,hh,hg,hd,hf,hdq,hdk,hdg]

theorem capacity (c : Config) (order : List (CTA c)) (s : BlockState)
    (vals : IO.Inputs ℝ c) (bounds : RegionName → Nat) (hb : (IO.realIO c order).InBounds bounds s vals) :
    Capacity c bounds := by
  apply capacity_of_windows c bounds
  · intro i
    by_contra hn
    have hi : bounds (IO.inputRegion i)<IO.inputSize c i:=Nat.lt_of_not_ge hn
    exact (Nat.lt_irrefl _) (hb.1 i ⟨_,hi⟩ trivial)
  · intro o
    by_contra hn
    have hi : bounds (IO.outputRegion o)<IO.outputSize c o:=Nat.lt_of_not_ge hn
    exact (Nat.lt_irrefl _) (hb.2.1 o ⟨_,hi⟩ trivial)

theorem launch_safe (c : Config) (a : CTA c) (s : BlockState)
    (bounds : RegionName → Nat) (cap : Capacity c bounds) : (IO.realJob c a).Safe bounds s := by
  change (kernel c).eraseDType.toAlgKernel.TraceSafe bounds (Whole.prepare c s a)
  have old:=dense_bounds c s a
  apply RealSafety.source_safe (Whole.prepare c s a) c.layout .fp32 .fp32 .fp32 .fp32
    "q" "k" "v" "h" "g" "do" "dh" "dq" "dk" "dg" bounds _ rfl
  refine ⟨⟨?_,?_⟩,⟨?_,?_,?_,?_⟩,⟨?_,?_,?_,?_,?_⟩⟩
  · intro i hi; exact lt_of_lt_of_le (old.setup.gate i hi) (cap _)
  · exact lt_of_lt_of_le old.setup.last (cap _)
  · intro i j ht hj; exact lt_of_lt_of_le (old.loop.value i j ht hj) (cap _)
  · intro i j ht hj; exact lt_of_lt_of_le (old.loop.dout i j ht hj) (cap _)
  · intro j e hj he; exact lt_of_lt_of_le (old.loop.history j e hj he) (cap _)
  · intro j e hj he; exact lt_of_lt_of_le (old.loop.dhist j e hj he) (cap _)
  · intro i hi; exact lt_of_lt_of_le (old.post.query i hi) (cap _)
  · intro i hi; exact lt_of_lt_of_le (old.post.key i hi) (cap _)
  · intro i hi; exact lt_of_lt_of_le (old.post.dq i hi) (cap _)
  · intro i hi; exact lt_of_lt_of_le (old.post.dk i hi) (cap _)
  · intro i hi; exact lt_of_lt_of_le (old.post.dg i hi) (cap _)

theorem schedule_safe (c : Config) (order : List (CTA c)) (s : BlockState)
    (bounds : RegionName → Nat) (cap : Capacity c bounds) :
    PreparedAlgorithmScheduleSafe bounds (order.map (IO.realJob c)) s := by
  induction order generalizing s with
  | nil=>trivial
  | cons a rest ih=>exact ⟨launch_safe c a s bounds cap,fun t _=>ih t⟩

theorem q_coordinates (c : Config) (n : Nat) (hn : n<c.heads*c.T*c.K) :
    let p:=Target.qCoordinates c n hn
    (p.1.val*c.T+p.2.1.val)*c.K+p.2.2.val=n := by
  change (n/c.K/c.T*c.T+n/c.K%c.T)*c.K+n%c.K=n
  rw [Nat.div_add_mod',Nat.div_add_mod']
theorem g_coordinates (c : Config) (n : Nat) (hn : n<c.NK*c.heads*c.T) :
    let p:=Target.gCoordinates c n hn
    (p.1.val*c.heads+p.2.1.val)*c.T+p.2.2.val=n := by
  change (n/c.T/c.heads*c.heads+n/c.T%c.heads)*c.T+n%c.T=n
  rw [Nat.div_add_mod',Nat.div_add_mod']

theorem implements (c : Config) (order : List (CTA c)) (ho : order.Perm (allCTAs c)) :
    (IO.realIO c order).Implements (Target.mathematical c) := by
  apply PreparedAlgorithmProgramIO.Implements.intro
  · intro vals initial _ bounds bounded
    exact schedule_safe c order initial bounds (capacity c order initial vals bounds bounded)
  · intro vals initial inputs
    obtain ⟨final,ran,result⟩:=KernelReal.source c initial order ho
    have reads:=RealFormulaBinding.logical_inputs c order initial vals inputs
    refine ⟨final,by exact (schedule c order initial).trans ran,?_,?_⟩
    · intro o j _
      fin_cases o
      · change final.readMem "dq" j.val=Target.mathematical c vals 0 j
        have bound : j.val<c.heads*c.T*c.K:=by simpa [IO.realIO,IO.outputSize,IO.outputRegion,extent] using j.isLt
        let p:=Target.qCoordinates c j.val bound
        have value:=result.q p.1 p.2.1 p.2.2
        rw [q_coordinates c j.val bound] at value
        rw [value]
        exact (RealFormulaBinding.formula c _ _ reads (qOwner c p.1 p.2.1 p.2.2)
          (p.2.1.val%64) p.2.2.val).1
      · change final.readMem "dk" j.val=Target.mathematical c vals 1 j
        have bound : j.val<c.heads*c.T*c.K:=by simpa [IO.realIO,IO.outputSize,IO.outputRegion,extent] using j.isLt
        let p:=Target.qCoordinates c j.val bound
        have value:=result.k p.1 p.2.1 p.2.2
        rw [q_coordinates c j.val bound] at value
        rw [value]
        exact (RealFormulaBinding.formula c _ _ reads (qOwner c p.1 p.2.1 p.2.2)
          (p.2.1.val%64) p.2.2.val).2.1
      · change final.readMem "dg" j.val=Target.mathematical c vals 2 j
        have bound : j.val<c.NK*c.heads*c.T:=by simpa [IO.realIO,IO.outputSize,IO.outputRegion,extent] using j.isLt
        let p:=Target.gCoordinates c j.val bound
        have value:=result.g p.1 p.2.1 p.2.2
        rw [g_coordinates c j.val bound] at value
        rw [value]
        exact (RealFormulaBinding.formula c _ _ reads (gOwner c p.1 p.2.1 p.2.2)
          (p.2.2.val%64) 0).2.2
    · intro r i outside _
      apply result.frame
      rintro ⟨a,_,writes⟩
      rcases (writes_cases c a r i).mp writes with ⟨rfl,row,e,active,rfl⟩|⟨rfl,row,e,active,rfl⟩|⟨rfl,row,active,rfl⟩
      · have h:=outside (0 : Fin 3) ⟨_,q_address_bound c a row e active⟩ trivial
        exact h.elim (fun hn=>hn rfl) (fun hn=>hn rfl)
      · have h:=outside (1 : Fin 3) ⟨_,q_address_bound c a row e active⟩ trivial
        exact h.elim (fun hn=>hn rfl) (fun hn=>hn rfl)
      · have h:=outside (2 : Fin 3) ⟨_,g_address_bound c a row active⟩ trivial
        exact h.elim (fun hn=>hn rfl) (fun hn=>hn rfl)

#axiomsClean implements
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.RealIOResults
