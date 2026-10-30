import bench.tritonbench_g.lightning_attention.Proofs.InterLoopSafety
import bench.tritonbench_g.lightning_attention.Proofs.InterPureSafety
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem forwardBody (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head block : Nat) (input : Memory α) (carry : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head (block*p.NC*p.CB) (block*p.NC*p.CB) input carry s)
    (bounds : RegionName → Nat) (backed : HeadBounds p head bounds) :
    Safety.traceSafeList (engine (withDot M D)) bounds (interForwardParts p).body
      (s.setReg "i" .nat [] (fun _ => block)) := by
  let start := s.setReg "i" .nat [] (fun _ => block)
  have hc := ctx.set_i p head _ _ block input carry s
  have hi : start.regs .nat [] "i"=some (fun _ => block) := by simp [start,State.setReg]
  obtain ⟨q,hq,hqc,hqm⟩ := interQueryLoop M D p head block _ input carry start hc hi
  let init := interDeltaInit M D p q
  have hinit := interDeltaInit_ready M D p head _ _ input carry q hqc
  obtain ⟨v,hv,hvc,hva,hvm⟩ := interDeltaLoop M D p head _ block input carry init.val hinit.1 hinit.2.1
  rw [inter_forward_body_parts]
  apply append _ bounds _ _ start q
    (single_range _ _ _ _ _ _ (queryLoop M D p head block _ input carry start hc hi bounds backed))
    (run_single_range _ _ _ _ _ _ hq)
  apply append _ bounds _ _ q init.val (deltaInit M D p q bounds) init.property
  apply append _ bounds _ _ init.val v
    (single_range _ _ _ _ _ _ (deltaLoop M D p head _ block input carry init.val hinit.1 hinit.2.1 bounds backed))
    (run_single_range _ _ _ _ _ _ hv)
  exact carryFinish M D p head _ _ input carry (queryBlockDelta M D p head block input) v hvc hva bounds

theorem forwardScan (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head : Nat) (input : Memory α) (s : State α)
    (ctx : InterFReady p head 0 0 input (fun _ => zero M) s)
    (bounds : RegionName → Nat) (backed : HeadBounds p head bounds) :
    Safety.rangeSafe (engine (withDot M D)) bounds "i" 0 p.NT 1 (interForwardParts p).body s := by
  let I := fun n (t : State α) => ∃ carry,
    InterFReady p head (n*p.NC*p.CB) (n*p.NC*p.CB) input carry t
  apply Safety.rangeSafe_of_invariant _ _ _ _ _ _ I
  · rintro n hn t ⟨carry,hctx⟩
    exact forwardBody M D p head n input carry t hctx bounds backed
  · rintro n hn t ⟨carry,hctx⟩ u ran
    obtain ⟨v,ran',next,_⟩ := interForwardBody M D p head n input carry t hctx
    have same : u = v := Option.some.inj (ran.symm.trans ran')
    subst u
    exact ⟨_,next⟩
  · exact ⟨_,by simpa using ctx⟩

theorem reverseBody (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (hbalance : p.NC*p.CB=p.BT) (head block : Nat) (hbnd : block<p.NT)
    (input : Memory α) (carry : Matrix α p.D p.E) (s : State α)
    (ctx : InterRReady p head (reverseRow p block 0) (reverseRow p block 0) input carry s)
    (bounds : RegionName → Nat) (backed : HeadBounds p head bounds) :
    Safety.traceSafeList (engine (withDot M D)) bounds (interReverseParts p).body
      (s.setReg "__rev_i" .int [] (fun _ => (block : Int)+1-p.NT)) := by
  let entry := s.setReg "__rev_i" .int [] (fun _ => (block : Int)+1-p.NT)
  let start := entry.setReg "i" .int [] (fun _ => (p.NT : Int)-1-block)
  have hstart : run (engine (withDot M D)) [reverseIndex "i"] entry=some start := by
    have ha : (0 : Int)-((block : Int)+1-p.NT)=(p.NT : Int)-1-block := by ring
    simp [reverseIndex,run,step,evalExpr,evalOp_unfold,entry,start,State.setReg,numeric,
      FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,ha]
    funext d sh n
    split_ifs <;> subst_vars <;>
      simp_all [FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    funext _
    change (0 : Int)-((block : Int)+1-p.NT)=(p.NT : Int)-1-block
    omega
  have hc : InterRReady p head (reverseRow p block 0) (reverseRow p block 0) input carry start := by
    constructor
    · exact ctx.inputs
    all_goals simp [start,entry,State.setReg,ctx.query,ctx.incoming,ctx.key,ctx.value,ctx.dk,ctx.dv,ctx.rows1,ctx.rows2,ctx.total]
  have hi : start.regs .int [] "i"=some (fun _ => (p.NT : Int)-1-block) := by simp [start,State.setReg]
  obtain ⟨q,hq,hqc,hqm⟩ := interReverseUpdateLoop M D p hbalance head block _ hbnd input carry start hc hi
  let init := interReverseDeltaInit M D p q
  have hinit := interReverseDeltaInit_ready M D p head _ _ input carry q hqc
  obtain ⟨v,hv,hvc,hva,hvm⟩ := interReverseDeltaLoop M D p hbalance head _ block hbnd input carry init.val hinit.1 hinit.2.1
  rw [inter_reverse_body_parts]
  apply append _ bounds _ _ entry start ?_ hstart
  · apply append _ bounds _ _ start q
      (single_reverse _ _ _ _ _ _ (reverseUpdateLoop M D p hbalance head block _ hbnd input carry start hc hi bounds backed))
      (run_single_reverse _ _ _ _ _ _ hq)
    apply append _ bounds _ _ q init.val (reverseDeltaInit M D p q bounds) init.property
    apply append _ bounds _ _ init.val v
      (single_reverse _ _ _ _ _ _ (reverseDeltaLoop M D p hbalance head _ block hbnd input carry init.val hinit.1 hinit.2.1 bounds backed))
      (run_single_reverse _ _ _ _ _ _ hv)
    exact reverseCarryFinish M D p head _ _ input carry (keyValueBlockDelta M D p head block input) v hvc hva bounds
  · apply Safety.traceSafeList_cons_intro
    · simp [reverseIndex,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold]
    · intro t ran; exact Safety.traceSafeList_nil _ _ _

theorem reverseScan (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (hbalance : p.NC*p.CB=p.BT) (head : Nat) (input : Memory α) (s : State α)
    (ctx : InterRReady p head (p.NT*p.BT) (p.NT*p.BT) input (fun _ => zero M) s)
    (bounds : RegionName → Nat) (backed : HeadBounds p head bounds) :
    Safety.rangeIntSafe (engine (withDot M D)) bounds "__rev_i" (1-(p.NT : Int)) 1 1 (interReverseParts p).body s := by
  let I := fun n (t : State α) => ∃ carry,
    InterRReady p head (reverseRow p n 0) (reverseRow p n 0) input carry t
  apply signed_range _ _ _ _ p.NT I _ _ 0 (Nat.zero_le _) s
    ⟨_,by simpa [reverseRow] using ctx⟩
  · rintro n hn t ⟨carry,hctx⟩
    exact reverseBody M D p hbalance head n hn input carry t hctx bounds backed
  · rintro n hn t ⟨carry,hctx⟩ u ran
    obtain ⟨v,ran',next,_⟩ := interReverseBody M D p hbalance head n hn input carry t hctx
    have same : u = v := Option.some.inj (ran.symm.trans ran')
    subst u
    exact ⟨_,next⟩

theorem inter (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (hbalance : p.NC*p.CB=p.BT) (s : State α)
    (bounds : RegionName → Nat) (backed : HeadBounds p (s.pids 0) bounds) :
    TraceSafe (engine (withDot M D)) bounds (KernelOnly.inter p) s := by
  let start := interFEntry M D p s
  have hstart := interFEntry_ready M D p s
  obtain ⟨q,hq,hqc,hqm⟩ := interForwardScan M D p (s.pids 0) s.mem start.val hstart
  let reset := interREntry M D p (s.pids 0) _ _ s.mem _ q hqc
  have hreset := interREntry_ready M D p (s.pids 0) _ _ s.mem _ q hqc
  change Safety.traceSafeList _ _ (KernelOnly.inter p).surfaceBody s
  rw [inter_parts]
  apply append _ bounds _ _ s start.val (forwardEntry M D p s bounds) start.property
  apply append _ bounds _ _ start.val q
    (single_range _ _ _ _ _ _ (forwardScan M D p (s.pids 0) s.mem start.val hstart bounds backed))
    (run_single_range _ _ _ _ _ _ hq)
  apply append _ bounds _ _ q reset.val (reverseEntry M D p (s.pids 0) _ _ s.mem _ q hqc bounds) reset.property
  exact single_reverse _ _ _ _ _ _ (reverseScan M D p hbalance (s.pids 0) s.mem reset.val hreset bounds backed)

#axiomsClean inter
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.SourceSafety
