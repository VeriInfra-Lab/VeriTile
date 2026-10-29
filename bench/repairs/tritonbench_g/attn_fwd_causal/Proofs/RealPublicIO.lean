import bench.repairs.tritonbench_g.attn_fwd_causal.RealIO
import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.IOBinding
import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.AttnFwdCausal.RealPublicIO
open VeriTile Triton FP.Structural Source Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def canonical (p : Parameters) (vals : IO.Inputs ℝ p) (s : State Real.R) : State Real.R :=
  {s with mem:=fun r off=>.mk .real (some (Reference.inputAt p vals r off))}

theorem implements (p : Parameters)
    (apart : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    (RealIO.program p work).Implements (RealIO.formula p work) := by
  intro vals _ bounds bounded initial binding
  refine ⟨IOBinding.schedule_safe Real.algebra Real.dotModel p work some (fun _ x=>x.unbotD 0)
    vals bounds bounded work (fun pid member=>legal.mem_iff.mpr member) initial,?_⟩
  obtain ⟨final,ran,values,frame⟩ := Kernel.grid_result (fun _=>Real.algebra) (fun _=>Real.dotModel)
    p (RawKernel.storage p apart) initial work legal
  refine ⟨final,(IOBinding.run_eq Real.algebra Real.dotModel p work initial).trans ran,?_,?_⟩
  · intro o j active
    fin_cases o
    change mapCell (fun x : Real.R=>x.unbotD 0)
      (final.mem p.Out (logicalAddress p j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val))=
      Cell.mk .fp16 (Reference.mathematical p vals j)
    rw [values _ j.1.isLt _ j.2.1.isLt _ j.2.2.1.isLt _ j.2.2.2.1.isLt]
    let pid : Program := (j.2.2.1.val/128,j.1.val*p.heads+j.2.1.val,0)
    have member : pid∈programs (grid p) :=
      (grid_covers p _ _ _ _ j.1.isLt j.2.1.isLt j.2.2.1.isLt j.2.2.2.1.isLt).1
    have reads : ∀ (i : Fin (IO.inputs p).length) (k : TileIndex ((IO.inputs p)[i]).shape),
        (initial.mem (IO.inputs p)[i].buf (IO.inputAddress p i k)).read .real=
        ((canonical p vals initial).mem (IO.inputs p)[i].buf (IO.inputAddress p i k)).read .real := by
      intro i k
      exact IOBinding.lookup_read Real.algebra Real.dotModel p work some (fun _ x=>x.unbotD 0)
        vals initial binding i k
    have same := IOBinding.inputs_agree p pid member initial (canonical p vals initial) reads
    have tile : (start (grid p) pid initial).pids 0*128<p.length := by
      change j.2.2.1.val/128*128<p.length
      have := j.2.2.1.isLt
      omega
    rw [FiniteReadback.answer_eq Real.algebra Real.dotModel p _ _ same tile]
    have allReads : Real.Reads p (start (grid p) pid (canonical p vals initial))
        (Reference.inputAt p vals) := by intro r hr off; rfl
    have laneActive : Source.active p (start (grid p) pid (canonical p vals initial))
        (lane j.2.2.1.val j.2.2.2.1.val j.2.2.2.1.isLt) := by
      constructor
      · change j.2.2.1.val/128*128+j.2.2.1.val%128<p.length
        have := j.2.2.1.isLt
        omega
      · exact j.2.2.2.1.isLt
    rw [Real.answer_real p _ _ allReads _ laneActive,
      Real.expected_at p (canonical p vals initial) (Reference.inputAt p vals)
        _ _ _ _ j.2.1.isLt j.2.2.2.1.isLt]
    rfl
  · intro r off outside
    apply frame r off
    by_cases region : r=p.Out
    · right
      intro footprint
      obtain ⟨b,hb,h,hh,n,hn,d,hd,address⟩ := footprint
      have absent := outside.1 ⟨0,by simp [RealIO.program,IO.program]⟩ (⟨b,hb⟩,⟨h,hh⟩,⟨n,hn⟩,⟨d,hd⟩,()) trivial
      exact absent.elim (fun h=>h region) (fun h=>h address)
    · exact Or.inl region

#axiomsClean implements
end
end VeriTile.Bench.Repairs.AttnFwdCausal.RealPublicIO
