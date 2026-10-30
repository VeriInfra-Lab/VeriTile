import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.InterForwardProgram
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.InterReverseLoops
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def interReverseDeltaInit (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (s : State α) :
    {t : State α // run (engine (withDot M D)) (interRDeltaParts p).before s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [interRDeltaParts,interRUpdateParts,interReverseParts,inter,ComputeKernel.surfaceBody,interForwardParts,Source.dynamicRange?,Source.intRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,State.setReg]
  rfl

theorem interReverseDeltaInit_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 : Nat) (input : Memory α) (carry : Matrix α p.D p.E)
    (s : State α) (ctx : InterRReady p head row1 row2 input carry s) :
    let t := (interReverseDeltaInit M D p s).val
    InterRReady p head row1 row2 input carry t ∧ InterRDelta p (fun _ => zero M) t ∧ t.mem=s.mem := by
  dsimp only
  refine ⟨?_,?_,?_⟩
  · constructor
    · exact ctx.inputs
    all_goals simp [interReverseDeltaInit,ctx.query,ctx.incoming,ctx.key,ctx.value,ctx.dk,ctx.dv,ctx.rows1,ctx.rows2,ctx.total]
  · simp [InterRDelta,interReverseDeltaInit,zero]
  · simp [interReverseDeltaInit]

def interReverseCarryFinish (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 : Nat) (input : Memory α) (carry acc : Matrix α p.D p.E)
    (s : State α) (ctx : InterRReady p head row1 row2 input carry s) (ha : InterRDelta p acc s) :
    {t : State α // run (engine (withDot M D)) (interRDeltaParts p).after s=some t} := by
  simp only [InterRDelta] at ha
  refine ⟨?_,?_⟩
  swap
  simp [interRDeltaParts,interRUpdateParts,interReverseParts,inter,ComputeKernel.surfaceBody,interForwardParts,Source.dynamicRange?,Source.intRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,numeric,bop,State.setReg,ctx.total,ha]
  rfl

theorem interReverseCarryFinish_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 : Nat) (input : Memory α) (carry acc : Matrix α p.D p.E)
    (s : State α) (ctx : InterRReady p head row1 row2 input carry s) (ha : InterRDelta p acc s) :
    let t := (interReverseCarryFinish M D p head row1 row2 input carry acc s ctx ha).val
    InterRReady p head row1 row2 input (fun i => add M (carry i) (acc i)) t ∧ t.mem=s.mem := by
  dsimp only
  refine ⟨?_,?_⟩
  · constructor
    · exact ctx.inputs
    all_goals simp [interReverseCarryFinish,ctx.query,ctx.incoming,ctx.key,ctx.value,ctx.dk,ctx.dv,ctx.rows1,ctx.rows2,add]
    funext i
    simp [FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,pair_eta]
  · simp [interReverseCarryFinish]

theorem inter_reverse_body_parts (p : Parameters) :
    (interReverseParts p).body=
    [reverseIndex "i"] ++
    [.forRangeInt "__rev_j" (reverseStart p.NC) (.constInt 1) (.constNat 1) (interRUpdateParts p).body] ++
    (interRDeltaParts p).before ++
    [.forRangeInt "__rev_j" (reverseStart p.NC) (.constInt 1) (.constNat 1) (interRDeltaParts p).body] ++
    (interRDeltaParts p).after := rfl

theorem interReverseBody (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (hbalance : p.NC*p.CB=p.BT) (head block : Nat) (hbnd : block<p.NT) (input : Memory α) (carry : Matrix α p.D p.E)
    (s : State α) (ctx : InterRReady p head (reverseRow p block 0) (reverseRow p block 0) input carry s) :
    ∃ t, run (engine (withDot M D)) (interReverseParts p).body
        (s.setReg "__rev_i" .int [] (fun _ => (block : Int)+1-p.NT))=some t ∧
      InterRReady p head (reverseRow p (block+1) 0) (reverseRow p (block+1) 0) input
        (fun i => add M (carry i) (keyValueBlockDelta M D p head block input i)) t ∧
      t.mem=keyValueBlockMemory M D p head block carry s.mem := by
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
  let finish := interReverseCarryFinish M D p head _ _ input carry (keyValueBlockDelta M D p head block input) v hvc hva
  have hf := interReverseCarryFinish_ready M D p head _ _ input carry (keyValueBlockDelta M D p head block input) v hvc hva
  refine ⟨finish.val,?_,hf.1,?_⟩
  · change run (engine (withDot M D)) (interReverseParts p).body entry=some finish.val
    rw [inter_reverse_body_parts]
    have hiExec : run (engine (withDot M D)) (interRDeltaParts p).before q=some init.val := init.property
    have hfExec : run (engine (withDot M D)) (interRDeltaParts p).after v=some finish.val := finish.property
    simp only [run_append,hstart,Option.bind_some,run_single_reverse _ _ _ _ _ _ hq,hiExec,
      run_single_reverse _ _ _ _ _ _ hv,hfExec]
  · exact hf.2.trans (hvm.trans (hinit.2.2.trans hqm))

theorem interReverseScan (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (hbalance : p.NC*p.CB=p.BT) (head : Nat) (input : Memory α) (s : State α)
    (ctx : InterRReady p head (p.NT*p.BT) (p.NT*p.BT) input (fun _ => zero M) s) :
    ∃ t, FP.Structural.rangeInt (engine (withDot M D)) "__rev_i" (1-(p.NT : Int)) 1 1 (interReverseParts p).body s=some t ∧
      InterRReady p head (reverseRow p p.NT 0) (reverseRow p p.NT 0) input (keyValueCarry M D p head input p.NT) t ∧
      t.mem=keyValueScanMemory M D p head input s.mem := by
  let update := fun n mem => keyValueBlockMemory M D p head n (keyValueCarry M D p head input n) mem
  let I := fun n (t : State α) =>
    InterRReady p head (reverseRow p n 0) (reverseRow p n 0) input (keyValueCarry M D p head input n) t ∧
    t.mem=foldRange update 0 n 1 s.mem
  have hs : ∀ n, n<p.NT → ∀ t, I n t → ∃ u,
      run (engine (withDot M D)) (interReverseParts p).body
        (t.setReg "__rev_i" .int [] (fun _ => (n : Int)+1-p.NT))=some u ∧ I (n+1) u := by
    intro n hn t ht
    obtain ⟨u,he,hc,hm⟩ := interReverseBody M D p hbalance head n hn input (keyValueCarry M D p head input n) t ht.1
    refine ⟨u,he,hc,?_⟩
    rw [hm,ht.2]
    exact (foldRange_succ update n s.mem).symm
  have initial : I 0 s := by exact ⟨by simpa [keyValueCarry,reverseRow] using ctx,by simp [foldRange_zero]⟩
  simpa using signed_range_invariant (engine (withDot M D)) "__rev_i" (interReverseParts p).body p.NT I hs 0 (Nat.zero_le _) s initial
#axiomsClean interReverseBody
#axiomsClean interReverseScan
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
