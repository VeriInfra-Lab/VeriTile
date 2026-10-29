import bench.repairs.tritonbench_g.attention_fwd_triton2.Proofs.IOResults

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.MathematicalIO
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Interface
noncomputable section

def realState (values : RegionName→Nat→ℝ) : State Real.R where
  mem := fun r off=>.mk .real (some (values r off))
  regs := fun _ _ _=>none
  pids := fun _=>0
  numPids := fun _=>1
  undef := fun d _ _=>defaultValue d

theorem algorithm_real (p : Parameters) (values : RegionName→Nat→ℝ)
    (b h n d : Nat) (hh : h<p.heads) (hn : n<p.length) (hd : d<96) :
    KernelOnly.Specification.algorithm Real.algebra Real.dotModel p (fun r off=>some (values r off))
      (n/128) b h (lane n d hd)=some (RealReturned.target p values b h n d) := by
  have positive : 0<p.heads := by omega
  have div : (b*p.heads+h)/p.heads=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hh]
    omega
  have mod : (b*p.heads+h)%p.heads=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  have bound := KernelOnly.source_algorithm Real.algebra Real.dotModel p
    (start (grid p) (n/128,b*p.heads+h,0) (realState values))
    (fun r off=>some (values r off)) (n/128) b h
    (by simp [start,ids]) (by simpa [start,ids] using div) (by simpa [start,ids] using mod)
    (by simp [start,ids]) (by intro r hr off; rfl)
  have interpreted := Real.answer_real p (start (grid p) (n/128,b*p.heads+h,0) (realState values))
    values (by intro r hr off; rfl) (lane n d hd) (by
      change n/128*128+n%128<p.length ∧ d<96
      exact ⟨by omega,hd⟩)
  rw [bound,RealReturned.target_at_program p (realState values) values b h n d hh hd] at interpreted
  exact interpreted

theorem implements (p : Parameters) (bases : Bases) (dom : Domain p bases)
    (sep : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    (io Real.algebra Real.dotModel p some (fun _ v=>v.unbotD 0) work).Implements (realFormula p) := by
  have target : IOResults.formula Real.algebra Real.dotModel p some (fun _ v=>v.unbotD 0)=realFormula p := by
    funext vals o j
    have ho : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    rcases j with ⟨b,h,n,d,⟨⟩⟩
    change (KernelOnly.Specification.algorithm Real.algebra Real.dotModel p
      (fun r off=>some (reader p vals r off)) (n.val/128) b.val h.val (lane n.val d.val d.isLt)).unbotD 0=_
    rw [algorithm_real p (reader p vals) b.val h.val n.val d.val h.isLt n.isLt d.isLt]
    rfl
  rw [←target]
  exact IOResults.implements Real.algebra Real.dotModel p bases dom sep some (fun _ v=>v.unbotD 0) work order

#axiomsClean implements
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.MathematicalIO
