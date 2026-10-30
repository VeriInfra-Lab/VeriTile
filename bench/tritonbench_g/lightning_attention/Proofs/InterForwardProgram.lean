import bench.tritonbench_g.lightning_attention.Proofs.InterForwardLoops
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
theorem InterFReady.set_i (p : Parameters) (head row1 row2 block : Nat)
    (input : Memory α) (carry : Matrix α p.E p.D) (s : State α)
    (ctx : InterFReady p head row1 row2 input carry s) :
    InterFReady p head row1 row2 input carry (s.setReg "i" .nat [] (fun _ => block)) := by
  constructor
  · exact ctx.inputs
  all_goals simp [State.setReg,ctx.qkOffset,ctx.vOffset,ctx.oOffset,ctx.query,ctx.incoming,ctx.key,ctx.value,ctx.rows1,ctx.rows2,ctx.total]

def interDeltaInit (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (s : State α) :
    {t : State α // run (engine (withDot M D)) (interFDeltaParts p).before s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [interFDeltaParts,interFQueryParts,interForwardParts,inter,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,State.setReg]
  rfl

theorem interDeltaInit_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 : Nat) (input : Memory α) (carry : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head row1 row2 input carry s) :
    let t := (interDeltaInit M D p s).val
    InterFReady p head row1 row2 input carry t ∧ InterFDelta p (fun _ => zero M) t ∧ t.mem=s.mem := by
  dsimp only
  refine ⟨?_,?_,?_⟩
  · constructor
    · exact ctx.inputs
    all_goals simp [interDeltaInit,ctx.qkOffset,ctx.vOffset,ctx.oOffset,ctx.query,ctx.incoming,ctx.key,ctx.value,ctx.rows1,ctx.rows2,ctx.total]
  · simp [InterFDelta,interDeltaInit,zero]
  · simp [interDeltaInit]

def interCarryFinish (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 : Nat) (input : Memory α) (carry acc : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head row1 row2 input carry s) (ha : InterFDelta p acc s) :
    {t : State α // run (engine (withDot M D)) (interFDeltaParts p).after s=some t} := by
  simp only [InterFDelta] at ha
  refine ⟨?_,?_⟩
  swap
  simp [interFDeltaParts,interFQueryParts,interForwardParts,inter,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,numeric,bop,State.setReg,ctx.total,ha]
  rfl

theorem interCarryFinish_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 : Nat) (input : Memory α) (carry acc : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head row1 row2 input carry s) (ha : InterFDelta p acc s) :
    let t := (interCarryFinish M D p head row1 row2 input carry acc s ctx ha).val
    InterFReady p head row1 row2 input (fun i => add M (carry i) (acc i)) t ∧ t.mem=s.mem := by
  dsimp only
  refine ⟨?_,?_⟩
  · constructor
    · exact ctx.inputs
    all_goals simp [interCarryFinish,ctx.qkOffset,ctx.vOffset,ctx.oOffset,ctx.query,ctx.incoming,ctx.key,ctx.value,ctx.rows1,ctx.rows2,add]
    funext i
    simp [FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,pair_eta]
  · simp [interCarryFinish]

theorem run_single_range (M : Algebra α) (idx : RegName) (stop : Nat) (body : List ComputeStmt)
    (s t : State α) (hr : FP.Structural.range M idx 0 stop 1 body s=some t) :
    run M [.forRangeDyn idx (.constNat 0) (.constNat stop) (.constNat 1) body] s=some t := by
  simp only [run,step,evalOp_unfold,bind,Option.bind,hr]

theorem inter_forward_body_parts (p : Parameters) :
    (interForwardParts p).body=
    [.forRangeDyn "j" (.constNat 0) (.constNat p.NC) (.constNat 1) (interFQueryParts p).body] ++
    (interFDeltaParts p).before ++
    [.forRangeDyn "j" (.constNat 0) (.constNat p.NC) (.constNat 1) (interFDeltaParts p).body] ++
    (interFDeltaParts p).after := rfl

theorem interForwardBody (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head block : Nat) (input : Memory α) (carry : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head (block*p.NC*p.CB) (block*p.NC*p.CB) input carry s) :
    ∃ t, run (engine (withDot M D)) (interForwardParts p).body
        (s.setReg "i" .nat [] (fun _ => block))=some t ∧
      InterFReady p head ((block+1)*p.NC*p.CB) ((block+1)*p.NC*p.CB) input
        (fun i => add M (carry i) (queryBlockDelta M D p head block input i)) t ∧
      t.mem=queryBlockMemory M D p head block carry s.mem := by
  let start := s.setReg "i" .nat [] (fun _ => block)
  have hc := ctx.set_i p head _ _ block input carry s
  have hi : start.regs .nat [] "i"=some (fun _ => block) := by simp [start,State.setReg]
  obtain ⟨q,hq,hqc,hqm⟩ := interQueryLoop M D p head block _ input carry start hc hi
  let init := interDeltaInit M D p q
  have hinit := interDeltaInit_ready M D p head _ _ input carry q hqc
  obtain ⟨v,hv,hvc,hva,hvm⟩ := interDeltaLoop M D p head _ block input carry init.val hinit.1 hinit.2.1
  let finish := interCarryFinish M D p head _ _ input carry (queryBlockDelta M D p head block input) v hvc hva
  have hf := interCarryFinish_ready M D p head _ _ input carry (queryBlockDelta M D p head block input) v hvc hva
  refine ⟨finish.val,?_,hf.1,?_⟩
  · change run (engine (withDot M D)) (interForwardParts p).body start=some finish.val
    rw [inter_forward_body_parts]
    simp only [run_append,run_single_range _ _ _ _ _ _ hq,Option.bind_some,init.property,
      run_single_range _ _ _ _ _ _ hv,finish.property]
  · exact hf.2.trans (hvm.trans (hinit.2.2.trans hqm))

theorem interForwardScan (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head : Nat) (input : Memory α) (s : State α)
    (ctx : InterFReady p head 0 0 input (fun _ => zero M) s) :
    ∃ t, FP.Structural.range (engine (withDot M D)) "i" 0 p.NT 1 (interForwardParts p).body s=some t ∧
      InterFReady p head (p.NT*p.NC*p.CB) (p.NT*p.NC*p.CB) input (queryCarry M D p head input p.NT) t ∧
      t.mem=queryScanMemory M D p head input s.mem := by
  let update := fun n mem => queryBlockMemory M D p head n (queryCarry M D p head input n) mem
  let I := fun n (t : State α) =>
    InterFReady p head (n*p.NC*p.CB) (n*p.NC*p.CB) input (queryCarry M D p head input n) t ∧
    t.mem=foldRange update 0 n 1 s.mem
  have hs : ∀ n, n<p.NT → ∀ t, I n t → ∃ u,
      run (engine (withDot M D)) (interForwardParts p).body
        (t.setReg "i" .nat [] (fun _ => n))=some u ∧ I (n+1) u := by
    intro n _ t ht
    obtain ⟨u,he,hc,hm⟩ := interForwardBody M D p head n input (queryCarry M D p head input n) t ht.1
    refine ⟨u,he,hc,?_⟩
    rw [hm,ht.2]
    exact (foldRange_succ update n s.mem).symm
  have initial : I 0 s := by exact ⟨by simpa [queryCarry] using ctx,by simp [foldRange_zero]⟩
  exact range_invariant (engine (withDot M D)) "i" (interForwardParts p).body p.NT I hs 0 (Nat.zero_le _) s initial
#axiomsClean interForwardBody
#axiomsClean interForwardScan
end
end VeriTile.Bench.Repairs.LightningAttention
