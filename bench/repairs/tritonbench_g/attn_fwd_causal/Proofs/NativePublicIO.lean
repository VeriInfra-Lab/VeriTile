import bench.repairs.tritonbench_g.attn_fwd_causal.NativeIO
import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.IOBinding
import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.AttnFwdCausal.NativePublicIO
open VeriTile Triton FP.Structural Source Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def canonical (p : Parameters) (vals : IO.Inputs Native.Word p) : State Native.Word where
  mem:=fun r off=>.mk .real (Reference.inputAt p vals r off)
  regs:=fun _ _ _=>none
  pids:=fun _=>0
  numPids:=fun _=>0
  undef:=fun dtype _ _=>defaultValue dtype

theorem implements (instructions : Native.Instructions) (p : Parameters)
    (apart : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    (NativeIO.program instructions p work).Implements (NativeIO.formula instructions p work) := by
  intro vals _ bounds bounded initial binding
  refine ⟨IOBinding.schedule_safe Native.algebra Native.dotModel p work (Native.inject .real)
    (fun feature=>Native.project instructions (Native.physicalOwner feature.val) .fp16)
    vals bounds bounded work (fun pid member=>legal.mem_iff.mpr member) initial,?_⟩
  obtain ⟨final,ran,values,frame⟩ := Kernel.grid_result (fun _=>Native.algebra) (fun _=>Native.dotModel)
    p (RawKernel.storage p apart) initial work legal
  refine ⟨final,(IOBinding.run_eq Native.algebra Native.dotModel p work initial).trans ran,?_,?_⟩
  · intro o j active
    fin_cases o
    change mapCell (Native.project instructions (Native.physicalOwner j.2.2.2.1.val) .fp16)
      (final.mem p.Out (logicalAddress p j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val))=
      Cell.mk .fp16 (Reference.native instructions p vals j)
    rw [values _ j.1.isLt _ j.2.1.isLt _ j.2.2.1.isLt _ j.2.2.2.1.isLt]
    let pid : Program := (j.2.2.1.val/128,j.1.val*p.heads+j.2.1.val,0)
    have member : pid∈programs (grid p) :=
      (grid_covers p _ _ _ _ j.1.isLt j.2.1.isLt j.2.2.1.isLt j.2.2.2.1.isLt).1
    have reads : ∀ (i : Fin (IO.inputs p).length) (k : TileIndex ((IO.inputs p)[i]).shape),
        (initial.mem (IO.inputs p)[i].buf (IO.inputAddress p i k)).read .real=
        ((Native.injectState (canonical p vals)).mem (IO.inputs p)[i].buf (IO.inputAddress p i k)).read .real := by
      intro i k
      exact IOBinding.lookup_read Native.algebra Native.dotModel p work (Native.inject .real)
        (fun feature=>Native.project instructions (Native.physicalOwner feature.val) .fp16)
        vals initial binding i k
    have same := IOBinding.inputs_agree p pid member initial (Native.injectState (canonical p vals)) reads
    have tile : (start (grid p) pid initial).pids 0*128<p.length := by
      change j.2.2.1.val/128*128<p.length
      have := j.2.2.1.isLt
      omega
    rw [FiniteReadback.answer_eq Native.algebra Native.dotModel p _ _ same tile]
    change Cell.mk .fp16 (Native.outputWord instructions p (canonical p vals)
      j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val j.2.2.2.1.isLt)=_
    rw [NativeReference.target_binding instructions p (canonical p vals)
      _ _ _ _ j.2.1.isLt j.2.2.1.isLt j.2.2.2.1.isLt]
    rfl
  · intro r off outside
    apply frame r off
    by_cases region : r=p.Out
    · right
      intro footprint
      obtain ⟨b,hb,h,hh,n,hn,d,hd,address⟩ := footprint
      have absent := outside.1 ⟨0,by simp [NativeIO.program,IO.program]⟩ (⟨b,hb⟩,⟨h,hh⟩,⟨n,hn⟩,⟨d,hd⟩,()) trivial
      exact absent.elim (fun h=>h region) (fun h=>h address)
    · exact Or.inl region

#axiomsClean implements
end
end VeriTile.Bench.Repairs.AttnFwdCausal.NativePublicIO
