import bench.repairs.tritonbench_g.context_attn_llama.Proofs.NativeRange
import bench.repairs.tritonbench_g.context_attn_llama.Proofs.NativeReachability

namespace VeriTile.Bench.Repairs.ContextAttnLlama.Native
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ContextAttnLlama FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

abbrev stopAt (c : Host.Call) (idx : GridIndex c.grid) :=
  stopIndex (c.launchAt idx).block c.layout.BM (seqAt c idx) (c.prompt (batchAt c idx))

def LoopBridge (c : Host.Call) (w : Widths) (M : Algebra α) : Prop :=
  ∀ (idx : GridIndex c.grid) (s : State α),
    bitLoop (M.withDefaultPrecision .fp32) (FPSuccess.parts c.layout c.regions (Host.scale c.layout.D)).body c.layout.BN
      (countBits w.sequence w.prompt (c.launchAt idx).block c.layout.BM (seqAt c idx) (c.prompt (batchAt c idx)))
      (countAt c idx+1) (word 0) s =
    rangeInt (M.withDefaultPrecision .fp32) "start_n" 0 (stopAt c idx : Int) c.layout.BN
      (FPSuccess.parts c.layout c.regions (Host.scale c.layout.D)).body s

theorem loop_bridge (c : Host.Call) (A : FlatAlloc) (w : Widths) (B : Bytes A)
    (abi : ABI c A w B) (M : Algebra α) : LoopBridge c w M := by
  intro idx s
  have hb := Host.batch_bound c idx
  have hseq := metadata_lt w.sequence _ (abi.sequence _ hb)
  have hstop : stopAt c idx < 2^63 := lt_of_le_of_lt (stop_le _ _ _ _ (by (try dsimp only [seqAt]); omega)) hseq
  rw [count_exact _ _ _ _ _ _ (block_abi c A w B abi idx) c.blockM
    (abi.sequence _ hb) (abi.prompt _ hb) (by (try dsimp only [seqAt]); omega)]
  simpa only [c.blockN,countAt,blocks,stopAt] using
    native_range_eq (M.withDefaultPrecision .fp32) (FPSuccess.parts c.layout c.regions (Host.scale c.layout.D)).body
      (stopAt c idx) c.layout.BM c.blockM hstop s

def Header (c : Host.Call) (M : Algebra α) (initial : State α) : Prop :=
  ∀ idx : GridIndex c.grid, ∃ t,
    run (M.withDefaultPrecision .fp32) (FPSuccess.parts c.layout c.regions (Host.scale c.layout.D)).before
      (FPSuccess.atIndex initial idx) = some t ∧
    evalOp (M.withDefaultPrecision .fp32) none (FPSuccess.parts c.layout c.regions (Host.scale c.layout.D)).start t = some (fun _ => (0 : Int)) ∧
    evalOp (M.withDefaultPrecision .fp32) none (FPSuccess.parts c.layout c.regions (Host.scale c.layout.D)).stop t = some (fun _ => (stopAt c idx : Int)) ∧
    evalOp (M.withDefaultPrecision .fp32) none (FPSuccess.parts c.layout c.regions (Host.scale c.layout.D)).stride t = some (fun _ => c.layout.BN)

/-- Actual source header evaluation supplies the range used by loop_bridge. -/
theorem source_header (c : Host.Call) (M : Algebra α) (s : State α)
    (hm : FullMetadata c s) : Header c M s := by
  intro idx
  have hb := Host.batch_bound c idx
  have hstart := hm.1.1 _ hb
  have hlen := hm.1.2.1 _ hb
  have hprompt := hm.1.2.2 _ hb
  have hreq := hm.2.1 _ hb
  simp only [batchAt,Formula.batch,Host.Call.launchAt] at hstart hlen hprompt hreq
  simp (config := {maxSteps := 500000}) [FPSuccess.parts,kernel,Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnLlama.original,ComputeKernel.surfaceBody,
    Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,State.setReg,Algebra.withDefaultPrecision,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,FPSuccess.atIndex,hstart,hlen,hprompt,hreq,
    Formula.head,Formula.queryRow,Host.Call.launchAt,batchAt,Formula.batch,bop,Nat.mul_comm]
  funext i
  simp only [stopAt,stopIndex,seqAt,batchAt,Formula.batch,Host.Call.launchAt,endIndex,
    Nat.add_sub_cancel_right,Nat.cast_add,Nat.cast_mul,Nat.cast_min]
  rw [mul_comm (c.layout.BM : Int) (idx.toPids 0 : Int)]
  simp only [bop,Broadcast.leftIndex,Broadcast.rightIndex,←Nat.cast_add,←Nat.cast_mul,Int.ofNat_lt]
  split_ifs <;> simp_all [min_def] <;> omega

/-- The public bridge composes actual typed source extraction, reached operand
states, all source range header/control execution, and every native memory
role. It does not postulate a native whole-kernel output equation. -/
def ProgramBridge (c : Host.Call) (A : FlatAlloc) (logical : BlockState)
    (w : Widths) (B : Bytes A) (M : Algebra α) (initial : State α) : Prop :=
  Correspondence c A logical w B ∧ SourceBridge c M initial ∧ Header c M initial ∧ LoopBridge c w M

theorem program_bridge (c : Host.Call) (A : FlatAlloc) (logical : BlockState)
    (valid : Host.Valid c A logical) (w : Widths) (B : Bytes A) (abi : ABI c A w B)
    (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M) (s : State α) (hm : FullMetadata c s) :
    ProgramBridge c A logical w B M s :=
  ⟨correspondence c A logical valid w B abi,source_bridge c M dots s hm,
    source_header c M s hm,loop_bridge c A w B abi M⟩

/-- The same bridge holds after every finite grid prefix: all five integer
read roles are stable while the private output is being written. -/
def GridBridge (c : Host.Call) (A : FlatAlloc) (logical : BlockState)
    (w : Widths) (B : Bytes A) (M : Algebra α) (initial : State α) : Prop :=
  ProgramBridge c A logical w B M initial ∧
  ∀ order : List (GridIndex c.grid), ∃ t,
    FPSuccess.runGrid c M order initial = some t ∧ ProgramBridge c A logical w B M t

theorem grid_bridge (c : Host.Call) (A : FlatAlloc) (logical : BlockState)
    (valid : Host.Valid c A logical) (w : Widths) (B : Bytes A) (abi : ABI c A w B)
    (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M) (s : State α) (hm : FullMetadata c s) :
    GridBridge c A logical w B M s := by
  refine ⟨program_bridge c A logical valid w B abi M dots s hm,?_⟩
  intro order
  obtain ⟨t,ht,hmeta,hframe⟩ := FPSuccess.grid_success_frame c M dots valid.input_separate order s hm.1
  have readRegion : ∀ rg, c.regions.out ≠ rg → ∀ off, t.mem rg off = s.mem rg off := by
    intro rg hrg off
    apply hframe
    rintro ⟨idx,hidx,i,hi,he,ho⟩
    exact hrg he.symm
  have hfull : FullMetadata c t := by
    refine ⟨hmeta,?_,?_⟩
    · intro b hb
      rw [readRegion c.regions.request (valid.input_separate _ (by simp)) b]
      exact hm.2.1 b hb
    · intro b hb j hj
      rw [readRegion c.regions.pages (valid.input_separate _ (by simp)) _]
      exact hm.2.2 b hb j hj
  exact ⟨t,ht,program_bridge c A logical valid w B abi M dots t hfull⟩

#print axioms grid_bridge

#print axioms program_bridge
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnLlama.Native
