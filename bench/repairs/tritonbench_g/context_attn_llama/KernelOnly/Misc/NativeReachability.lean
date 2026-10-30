import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.NativeSites
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.FPReference

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.Integer
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ContextAttnLlama FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The invariant established by the actual prelude and preserved by every
actual typed body execution. It contains no floating-value equation. -/
def Static (c : Host.Call) (idx : GridIndex c.grid) (s : State α) : Prop :=
  s.regs .int [] "cur_batch_in_all_start_index" = some (fun _ => (c.start (batchAt c idx) : Int)) ∧
  s.regs .nat [] "cur_head" = some (fun _ => Formula.head c.layout (c.launchAt idx)) ∧
  s.regs .nat [] "cur_kv_head" = some (fun _ => Formula.head c.layout (c.launchAt idx)/c.layout.kvGroup) ∧
  s.regs .int [] "cur_batch_req_idx" = some (fun _ => (c.request (batchAt c idx) : Int)) ∧
  s.regs .nat [c.layout.BM] "offs_m" = some (fun i => Formula.queryRow c.layout (c.launchAt idx) i.1.val) ∧
  s.regs .nat [c.layout.D] "offs_d" = some (fun i => i.1.val) ∧
  s.regs .nat [c.layout.BN] "offs_n" = some (fun i => i.1.val) ∧
  s.regs .int [] "cur_batch_seq_len" = some (fun _ => (c.length (batchAt c idx) : Int)) ∧
  s.regs .int [] "prompt_cache_len" = some (fun _ => (c.prompt (batchAt c idx) : Int)) ∧
  s.regs .int [] "block_end_loc" = some (fun _ => (endAt c idx : Int))

def Names : List RegName := ["cur_batch_in_all_start_index","cur_head","cur_kv_head","cur_batch_req_idx",
  "offs_m","offs_d","offs_n","cur_batch_seq_len","prompt_cache_len","block_end_loc"]

def Keeps (name : RegName) : ComputeStmt → Prop
  | .assign _ _ out _ => out ≠ name
  | _ => False

theorem run_keeps (M : Algebra α) (code : List ComputeStmt) (name : RegName)
    (hc : ∀ st ∈ code, Keeps name st) (s t : State α) (hr : run M code s = some t)
    (dtype : TileDType) (shape : TileShape) : t.regs dtype shape name = s.regs dtype shape name := by
  induction code generalizing s with
  | nil => simp only [run,Option.some.injEq] at hr; subst t; rfl
  | cons st rest ih =>
    have hk := hc st (List.mem_cons_self)
    cases st <;> try contradiction
    rename_i d sh out e
    change out ≠ name at hk
    simp only [run,step] at hr
    cases hv : evalExpr M e s with
    | none => simp only [hv,Option.bind_none] at hr; contradiction
    | some v =>
      simp only [hv,Option.bind_some] at hr
      rw [ih (fun a ha => hc a (List.mem_cons_of_mem _ ha)) _ hr]
      simp [State.setReg,hk,Ne.symm hk]

theorem body_keeps (p : Layout) (r : Regions) (scale : ℝ) (name : RegName) (hn : name ∈ Names) :
    ∀ st ∈ (FPSuccess.parts p r scale).body, Keeps name st := by
  simp only [Names,List.mem_cons,List.not_mem_nil,or_false] at hn
  rcases hn with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp [FPSuccess.parts,kernel,Program.kernel,VeriTile.Bench.TritonBenchG.ContextAttnLlama.original,
    ComputeKernel.surfaceBody,Source.intRange?,Keeps]

theorem static_preserved (c : Host.Call) (idx : GridIndex c.grid) (M : Algebra α) (s t : State α)
    (h : Static c idx s) (hr : run M (FPSuccess.parts c.layout c.regions (Host.scale c.layout.D)).body s = some t) :
    Static c idx t := by
  have keep : ∀ d sh name, name ∈ Names → t.regs d sh name = s.regs d sh name := by
    intro d sh name hn
    exact run_keeps M _ name (body_keeps _ _ _ name hn) s t hr d sh
  simpa [Static,keep,Names] using h

theorem static_index (c : Host.Call) (idx : GridIndex c.grid) (s : State α) (h : Static c idx s) (step : Nat) :
    Static c idx (s.setReg "start_n" .int [] (fun _ => ((step*c.layout.BN : Nat) : Int))) := by
  simpa [Static,State.setReg] using h

theorem end_choice (block bm length prompt : Nat) :
    (if ((bm*block+bm : Nat) : Int)+(prompt : Int) < ((length+prompt : Nat) : Int) then
      ((bm*block+bm : Nat) : Int)+(prompt : Int) else ((length+prompt : Nat) : Int)) =
      (endIndex block bm (length+prompt) prompt : Int) := by
  simp only [endIndex,Nat.add_sub_cancel_right,Nat.mul_comm bm block,Nat.cast_add,Nat.cast_min]
  split <;> omega

/-- The source prelude establishes the full discrete invariant from the same
resolved metadata state that the FP32 success theorem uses. -/
theorem prelude_static (c : Host.Call) (idx : GridIndex c.grid) (M : Algebra α)
    (s : State α) (hm : FullMetadata c s) :
    ∃ t, run (M.withDefaultPrecision .fp32) (FPSuccess.parts c.layout c.regions (Host.scale c.layout.D)).before
      (FPSuccess.atIndex s idx) = some t ∧ Static c idx t ∧ t.mem = s.mem := by
  have hb := Host.batch_bound c idx
  have hstart := hm.1.1 _ hb
  have hlen := hm.1.2.1 _ hb
  have hprompt := hm.1.2.2 _ hb
  have hreq := hm.2.1 _ hb
  simp only [batchAt,Formula.batch,Host.Call.launchAt] at hstart hlen hprompt hreq
  simp (config := {maxSteps := 500000}) [FPSuccess.parts,kernel,Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnLlama.original,ComputeKernel.surfaceBody,
    Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,Static,State.setReg,Algebra.withDefaultPrecision,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,FPSuccess.atIndex,hstart,hlen,hprompt,hreq,
    Formula.head,Formula.queryRow,Host.Call.launchAt,batchAt,Formula.batch,bop,Nat.mul_comm]
  refine ⟨rfl,rfl,rfl,?_,?_⟩
  · funext i
    simp [bop,Broadcast.leftIndex,Broadcast.rightIndex]
  · funext i
    simp only [endAt,seqAt,batchAt,Formula.batch,Host.Call.launchAt,endIndex,
      Nat.add_sub_cancel_right,Nat.cast_add,Nat.cast_mul,Nat.cast_min]
    rw [mul_comm (c.layout.BM : Int) (idx.toPids 0 : Int)]
    split
    · rename_i h; rw [min_eq_left (le_of_lt h)]
    · rename_i h; rw [min_eq_right (le_of_not_gt h)]


def indexState (c : Host.Call) (step : Nat) (s : State α) :=
  s.setReg "start_n" .int [] (fun _ => ((step*c.layout.BN : Nat) : Int))
def pageState (c : Host.Call) (idx : GridIndex c.grid) (step : Nat) (s : State α) :=
  (indexState c step s).setReg "kv_loc" .int [c.layout.BN] (fun i => (mathPage c idx step i.1.val : Int))

theorem page_discrete (c : Host.Call) (idx : GridIndex c.grid) (step : Nat) (s : State α)
    (h : Static c idx s) : Discrete c idx step (pageState c idx step s) := by
  rcases h with ⟨hs,hh,hkv,hr,hm,hd,hn,hl,hp,he⟩
  constructor <;> simp_all [pageState,indexState,State.setReg]

theorem page_prefix (c : Host.Call) (idx : GridIndex c.grid) (step : Nat)
    (M : Algebra α) (s : State α) (h : Static c idx s) (hm : FullMetadata c s) :
    run M ((FPSuccess.parts c.layout c.regions (Host.scale c.layout.D)).body.take 2) (indexState c step s) =
      some (pageState c idx step s) := by
  rcases h with ⟨hs,hh,hkv,hr,hrows,hd,hn,hl,hp,he⟩
  have hpv : ∀ j : Nat,
      (if step*c.layout.BN+j < endAt c idx then
        (s.mem c.regions.pages (c.layout.reqRow*c.request (batchAt c idx)+c.layout.reqToken*(step*c.layout.BN+j))).read .int
      else 0) = (mathPage c idx step j : Int) := by
    intro j
    by_cases ha : step*c.layout.BN+j < endAt c idx
    · have hf : step*c.layout.BN+j < c.length (batchAt c idx)+c.prompt (batchAt c idx) :=
        lt_of_lt_of_le ha (end_le _ _ _ _ (by dsimp only [seqAt]; omega))
      simp [ha,mathPage,hm.2.2 _ (Host.batch_bound c idx) _ hf]
    · simp [ha,mathPage]
  have hcode : (FPSuccess.parts c.layout c.regions (Host.scale c.layout.D)).body.take 2 =
      [.assign .int [] "start_n" (.alg (.ref .int [] "start_n")),
       .assign .int [c.layout.BN] "kv_loc" (.alg (.load .int
         (.region c.regions.pages (LoopProgram.pageIndex c.layout).castIntToNat)
         (.maskOther (LoopProgram.pageMask c.layout) (.broadcast (.constInt 0) [c.layout.BN]))))] := rfl
  have hi : evalOp M none (Op.ref .int [] "start_n") (indexState c step s) =
      some (fun _ => ((step*c.layout.BN : Nat) : Int)) := by
    simp [evalOp_unfold,indexState,State.setReg]
  have hid : (indexState c step s).setReg "start_n" .int [] (fun _ => ((step*c.layout.BN : Nat) : Int)) =
      indexState c step s := by
    simp [indexState,State.setReg]
    funext d sh n
    by_cases hn : n = "start_n" <;> simp [hn]
  have hid' : (indexState c step s).setReg "start_n" .int [] (fun _ => (step : Int)*(c.layout.BN : Int)) =
      indexState c step s := by simpa only [Nat.cast_mul] using hid
  have hload : evalOp M none (.load .int
      (.region c.regions.pages (LoopProgram.pageIndex c.layout).castIntToNat)
      (.maskOther (LoopProgram.pageMask c.layout) (.broadcast (.constInt 0) [c.layout.BN])))
      (indexState c step s) = some (fun i => (mathPage c idx step i.1.val : Int)) := by
    simp [evalOp_unfold,indexState,State.setReg,LoopProgram.pageIndex,LoopProgram.pageMask,
      LoopProgram.lanes,PreludeProgram.z,hr,hn,he,Option.bind,numeric,numericLt,bop,
      Broadcast.leftIndex,Broadcast.rightIndex]
    funext i
    simpa only [←Nat.cast_add,←Nat.cast_mul,Int.ofNat_lt,Int.toNat_natCast] using hpv i.1.val
  rw [hcode]
  simp [run,FP.Structural.step,evalExpr,hi,Option.bind_some,hid,hid',hload,pageState]

/-- A reached loop header of the actual typed source. The relation uses the
source prelude/body execution, not a caller-supplied mathematical trace. -/
inductive Reached (c : Host.Call) (M : Algebra α) (idx : GridIndex c.grid) (initial : State α) : Nat → State α → Prop
  | first (s : State α)
      (run : FP.Structural.run (M.withDefaultPrecision .fp32)
        (FPSuccess.parts c.layout c.regions (Host.scale c.layout.D)).before (FPSuccess.atIndex initial idx) = some s) :
      Reached c M idx initial 0 s
  | next (step : Nat) (s t : State α) (prior : Reached c M idx initial step s)
      (active : step < countAt c idx)
      (run : FP.Structural.run (M.withDefaultPrecision .fp32)
        (FPSuccess.parts c.layout c.regions (Host.scale c.layout.D)).body (indexState c step s) = some t) :
      Reached c M idx initial (step+1) t

/-- Every finite source-loop prefix succeeds and retains all metadata and
control registers. Thus source_observations applies at every reached body. -/
theorem reached_prefix_live (c : Host.Call) (idx : GridIndex c.grid) (M : Algebra α)
    (dots : FPSuccess.DotSupport c.layout M) (initial : State α) (hm : FullMetadata c initial)
    (step : Nat) (ht : step ≤ countAt c idx) :
    ∃ s acc, Reached c M idx initial step s ∧
      FPReference.Matches c.layout M (FPReference.inputs c.layout c.regions (FPSuccess.atIndex initial idx)) acc s ∧
      Static c idx s ∧ s.mem=initial.mem := by
  induction step with
  | zero =>
    obtain ⟨s,hs,live,hmem⟩ := FPReference.prelude c.layout c.regions (Host.scale c.layout.D) M (FPSuccess.atIndex initial idx)
    obtain ⟨t,ht,hstatic,htmem⟩ := prelude_static c idx M initial hm
    have heq : s=t := Option.some.inj (hs.symm.trans ht)
    subst t
    exact ⟨s,_,Reached.first s hs,live,hstatic,hmem⟩
  | succ step ih =>
    obtain ⟨s,acc,hs,live,hstatic,hmem⟩ := ih (by omega)
    obtain ⟨t,hrun,live',hmem'⟩ := FPReference.body c.layout c.regions (Host.scale c.layout.D)
      M dots (by rw [c.blockN]; exact Host.bm_positive c) (FPSuccess.atIndex initial idx) s acc live hmem
      ((step*c.layout.BN : Nat) : Int)
    exact ⟨t,_,Reached.next step s t hs (by omega) hrun,live',
      static_preserved c idx _ _ _ (static_index c idx s hstatic step) hrun,hmem'⟩

theorem reached_prefix (c : Host.Call) (idx : GridIndex c.grid) (M : Algebra α)
    (dots : FPSuccess.DotSupport c.layout M) (initial : State α) (hm : FullMetadata c initial)
    (step : Nat) (ht : step ≤ countAt c idx) :
    ∃ s, Reached c M idx initial step s ∧ FPSuccess.Ready c.layout s ∧ Static c idx s ∧ FullMetadata c s := by
  obtain ⟨s,acc,hs,live,static,memory⟩ := reached_prefix_live c idx M dots initial hm step ht
  exact ⟨s,hs,FPReference.matches_ready _ _ _ _ _ live,static,
    by simpa only [FullMetadata,FPSuccess.Metadata,memory] using hm⟩

/-- Composed source/native bridge: every actual source iteration has a reached
state; its actual masked page prefix establishes the concrete operands used
by the checked native/source address and mask correspondence. -/
def SourceBridge (c : Host.Call) (M : Algebra α) (initial : State α) : Prop :=
  Sites c.layout c.regions (Host.scale c.layout.D) ∧
  ∀ (idx : GridIndex c.grid) step, step < countAt c idx →
    ∃ s u, Reached c M idx initial step s ∧
      run (M.withDefaultPrecision .fp32) ((FPSuccess.parts c.layout c.regions (Host.scale c.layout.D)).body.take 2)
        (indexState c step s) = some u ∧ Discrete c idx step u ∧ Observed c idx step (M.withDefaultPrecision .fp32) u

theorem source_bridge (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M)
    (initial : State α) (hm : FullMetadata c initial) : SourceBridge c M initial := by
  refine ⟨source_sites _ _ _,?_⟩
  intro idx step ht
  obtain ⟨s,hs,hr,hstatic,hmeta⟩ := reached_prefix c idx M dots initial hm step (Nat.le_of_lt ht)
  exact ⟨s,pageState c idx step s,hs,page_prefix c idx step _ s hstatic hmeta,page_discrete c idx step s hstatic,
    source_observations c idx step _ _ (page_discrete c idx step s hstatic)⟩

#print axioms source_bridge
#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.Integer
