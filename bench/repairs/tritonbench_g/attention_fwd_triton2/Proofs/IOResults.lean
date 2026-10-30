import bench.repairs.tritonbench_g.attention_fwd_triton2.Proofs.IOSafety
import bench.repairs.tritonbench_g.attention_fwd_triton2.Proofs.FiniteReference

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.IOResults
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Interface InputValues
noncomputable section
variable {α β : Type} [Inhabited α] [Inhabited β]

/-- Independent source-order arithmetic, observed only at the final payload. -/
def formula (M : Algebra β) (dot : Dot β) (p : Parameters) (encode : α→β)
    (observe : TileIndex [p.batches,p.heads,p.length,96]→β→α) (vals : Inputs α p) :
    ChannelValues α (outputChannels p) :=
  fun o j => by
    have ho : o=0 := Fin.eq_zero o
    subst o
    exact observe j (KernelOnly.Specification.algorithm M dot p (fun r off=>encode (reader p vals r off))
      (j.2.2.1.val/128) j.1.val j.2.1.val (lane j.2.2.1.val j.2.2.2.1.val j.2.2.2.1.isLt))

theorem algorithm_at (M : Algebra β) (dot : Dot β) (p : Parameters) (canon : Canonical p)
    (encode : α→β) (observe : TileIndex [p.batches,p.heads,p.length,96]→β→α)
    (work : List Program) (vals : Inputs α p) (initial : State β)
    (hin : (io M dot p encode observe work).InputBinding vals initial)
    (b h n : Nat) (hb : b<p.batches) (hh : h<p.heads) (hn : n<p.length) :
    answer M dot p (start (grid p) (n/128,b*p.heads+h,0) initial)=
      KernelOnly.Specification.algorithm M dot p (fun r off=>encode (reader p vals r off)) (n/128) b h := by
  have positive : 0<p.heads := by omega
  have div : (b*p.heads+h)/p.heads=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hh]
    omega
  have mod : (b*p.heads+h)%p.heads=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  have covers := (mem_programs _ _ _ _).mp (grid_covers p b h n 0 hb hh hn (by decide)).1
  have actual := FiniteReference.algorithm M dot p (start (grid p) (n/128,b*p.heads+h,0) initial)
    canon covers.1 covers.2.1 (fun r off=>encode (reader p vals r off))
    (fun r off inside=>reader_bound M dot p encode observe work vals initial hin r off inside)
  simpa [start,ids,div,mod] using actual

theorem implements (M : Algebra β) (dot : Dot β) (p : Parameters) (bases : Bases) (dom : Domain p bases)
    (sep : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS)
    (encode : α→β) (observe : TileIndex [p.batches,p.heads,p.length,96]→β→α)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    (io M dot p encode observe work).Implements (formula M dot p encode observe) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds backed initial _
    exact IOSafety.schedule_safe M dot p dom.canonical encode observe work order vals bounds backed initial
  · intro vals initial hin
    obtain ⟨final,ran,outputs,frame⟩ := KernelInterface.complete_raw M dot p bases dom sep initial work order
    refine ⟨final,?_,?_,?_⟩
    · change runSchedule (work.map (invocation M dot p)) initial=some final
      rw [IOSafety.schedule_runs]
      exact ran
    · intro o j _
      have ho : o=(0 : Fin 1) := Fin.eq_zero o
      subst o
      rcases j with ⟨b,h,n,d,⟨⟩⟩
      change mapCell (observe (b,h,n,d,()))
        (final.mem p.Out (offset p.o b.val h.val n.val d.val))=_
      rw [outputs b.val b.isLt h.val h.isLt n.val n.isLt d.val d.isLt,
        algorithm_at M dot p dom.canonical encode observe work vals initial hin b.val h.val n.val b.isLt h.isLt n.isLt]
      rfl
    · intro r off outside
      apply frame
      by_cases same : r=p.Out
      · right
        intro b hb h hh n hn d hd
        have no := outside.1 (0 : Fin 1) (⟨b,hb⟩,⟨h,hh⟩,⟨n,hn⟩,⟨d,hd⟩,()) trivial
        exact no.resolve_left (fun hn=>hn same)
      · exact Or.inl same

#axiomsClean implements
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.IOResults
