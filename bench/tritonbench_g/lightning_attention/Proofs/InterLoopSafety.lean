import bench.tritonbench_g.lightning_attention.Proofs.InterBodySafety
import bench.tritonbench_g.lightning_attention.Proofs.SafetyComposition
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem queryLoop (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head block row2 : Nat) (input : Memory α) (carry : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head (block*p.NC*p.CB) row2 input carry s)
    (hi : s.regs .nat [] "i"=some (fun _ => block))
    (bounds : RegionName → Nat) (backed : HeadBounds p head bounds) :
    Safety.rangeSafe (engine (withDot M D)) bounds "j" 0 p.NC 1 (interFQueryParts p).body s := by
  let I := fun n (t : State α) => InterFReady p head ((block*p.NC+n)*p.CB) row2 input carry t ∧
    t.regs .nat [] "i"=some (fun _ => block)
  apply Safety.rangeSafe_of_invariant _ _ _ _ _ _ I
  · intro n hn t ht
    exact queryBody M D p head _ row2 block n input carry t ht.1 ht.2 bounds backed
  · intro n hn t ht u ran
    let exec := interQueryBody M D p head ((block*p.NC+n)*p.CB) row2 block n input carry t ht.1 ht.2
    have same : u = exec.val := Option.some.inj (ran.symm.trans exec.property)
    subst u
    have next := interQueryBody_ready M D p head ((block*p.NC+n)*p.CB) row2 block n input carry t ht.1 ht.2
    exact ⟨by simpa only [exec,Nat.add_mul,Nat.one_mul,Nat.add_assoc] using next.1,next.2⟩
  · exact ⟨by simpa using ctx,hi⟩

theorem deltaLoop (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 block : Nat) (input : Memory α) (carry : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head row1 (block*p.NC*p.CB) input carry s)
    (ha : InterFDelta p (fun _ => zero M) s)
    (bounds : RegionName → Nat) (backed : HeadBounds p head bounds) :
    Safety.rangeSafe (engine (withDot M D)) bounds "j" 0 p.NC 1 (interFDeltaParts p).body s := by
  let I := fun n (t : State α) => ∃ acc,
    InterFReady p head row1 ((block*p.NC+n)*p.CB) input carry t ∧ InterFDelta p acc t
  apply Safety.rangeSafe_of_invariant _ _ _ _ _ _ I
  · rintro n hn t ⟨acc,hctx,hacc⟩
    exact deltaBody M D p head row1 _ n input carry acc t hctx hacc bounds backed
  · rintro n hn t ⟨acc,hctx,hacc⟩ u ran
    let exec := interDeltaBody M D p head row1 ((block*p.NC+n)*p.CB) n input carry acc t hctx hacc
    have same : u = exec.val := Option.some.inj (ran.symm.trans exec.property)
    subst u
    have next := interDeltaBody_ready M D p head row1 ((block*p.NC+n)*p.CB) n input carry acc t hctx hacc
    exact ⟨_,by simpa only [exec,Nat.add_mul,Nat.one_mul,Nat.add_assoc] using next.1,next.2⟩
  · exact ⟨_,by simpa using ctx,ha⟩

theorem reverseUpdateLoop (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (hbalance : p.NC*p.CB=p.BT) (head block row2 : Nat) (hbnd : block<p.NT)
    (input : Memory α) (carry : Matrix α p.D p.E)
    (s : State α) (ctx : InterRReady p head (reverseRow p block 0) row2 input carry s)
    (hi : s.regs .int [] "i"=some (fun _ => (p.NT : Int)-1-block))
    (bounds : RegionName → Nat) (backed : HeadBounds p head bounds) :
    Safety.rangeIntSafe (engine (withDot M D)) bounds "__rev_j" (1-(p.NC : Int)) 1 1 (interRUpdateParts p).body s := by
  let I := fun n (t : State α) =>
    InterRReady p head (reverseRow p block n) row2 input carry t ∧
    t.regs .int [] "i"=some (fun _ => (p.NT : Int)-1-block)
  apply signed_range _ _ _ _ p.NC I _ _ 0 (Nat.zero_le _) s ⟨ctx,hi⟩
  · intro n hn t ht
    have ctx' : InterRReady p head (reverseRow p block (n+1)+p.CB) row2 input carry t := by
      rw [reverseRow_step p hbalance block n hbnd hn]; exact ht.1
    exact reverseUpdate M D p head _ row2 block n input carry t ctx' ht.2 hbnd bounds backed
  · intro n hn t ht u ran
    have ctx' : InterRReady p head (reverseRow p block (n+1)+p.CB) row2 input carry t := by
      rw [reverseRow_step p hbalance block n hbnd hn]; exact ht.1
    let exec := interReverseUpdate M D p head (reverseRow p block (n+1)) row2 block n input carry t ctx' ht.2 hbnd
    have same : u = exec.val := Option.some.inj (ran.symm.trans exec.property)
    subst u
    exact interReverseUpdate_ready M D p head _ row2 block n input carry t ctx' ht.2 hbnd

theorem reverseDeltaLoop (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (hbalance : p.NC*p.CB=p.BT) (head row1 block : Nat) (hbnd : block<p.NT)
    (input : Memory α) (carry : Matrix α p.D p.E)
    (s : State α) (ctx : InterRReady p head row1 (reverseRow p block 0) input carry s)
    (ha : InterRDelta p (fun _ => zero M) s)
    (bounds : RegionName → Nat) (backed : HeadBounds p head bounds) :
    Safety.rangeIntSafe (engine (withDot M D)) bounds "__rev_j" (1-(p.NC : Int)) 1 1 (interRDeltaParts p).body s := by
  let I := fun n (t : State α) => ∃ acc,
    InterRReady p head row1 (reverseRow p block n) input carry t ∧ InterRDelta p acc t
  apply signed_range _ _ _ _ p.NC I _ _ 0 (Nat.zero_le _) s ⟨_,ctx,ha⟩
  · rintro n hn t ⟨acc,hctx,hacc⟩
    have ctx' : InterRReady p head row1 (reverseRow p block (n+1)+p.CB) input carry t := by
      rw [reverseRow_step p hbalance block n hbnd hn]; exact hctx
    exact reverseDelta M D p head row1 _ n input carry acc t ctx' hacc bounds backed
  · rintro n hn t ⟨acc,hctx,hacc⟩ u ran
    have ctx' : InterRReady p head row1 (reverseRow p block (n+1)+p.CB) input carry t := by
      rw [reverseRow_step p hbalance block n hbnd hn]; exact hctx
    let exec := interReverseDelta M D p head row1 (reverseRow p block (n+1)) n input carry acc t ctx' hacc
    have same : u = exec.val := Option.some.inj (ran.symm.trans exec.property)
    subst u
    have next := interReverseDelta_ready M D p head row1 _ n input carry acc t ctx' hacc
    exact ⟨_,next⟩

#axiomsClean queryLoop
#axiomsClean deltaLoop
#axiomsClean reverseUpdateLoop
#axiomsClean reverseDeltaLoop
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.SourceSafety
