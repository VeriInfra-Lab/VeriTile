import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeSites

namespace VeriTile.Bench.Repairs.ContextAttnFwd.Native
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ContextAttnFwd FP.Structural
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
  s.regs .nat [] "cur_batch" = some (fun _ => batchAt c idx) ∧
  s.regs .nat [c.layout.BM] "offs_m" = some (fun i => Formula.queryRow c.layout (c.launchAt idx) i.1.val) ∧
  s.regs .nat [c.layout.D] "offs_d" = some (fun i => i.1.val) ∧
  s.regs .nat [c.layout.BN] "offs_n" = some (fun i => i.1.val) ∧
  s.regs .int [] "cur_batch_seq_len" = some (fun _ => (c.length (batchAt c idx) : Int)) ∧
  s.regs .int [] "prompt_cache_len" = some (fun _ => (c.prompt (batchAt c idx) : Int)) ∧
  s.regs .int [] "block_end_loc" = some (fun _ => (endAt c idx : Int))

def Names : List RegName := ["cur_batch_in_all_start_index","cur_head","cur_kv_head","cur_batch",
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
  all_goals simp [FPSuccess.parts,kernel,Program.kernel,VeriTile.Bench.TritonBenchG.ContextAttnFwd.original,
    ComputeKernel.surfaceBody,Source.intRange?,Keeps]

theorem static_preserved (c : Host.Call) (idx : GridIndex c.grid) (M : Algebra α) (s t : State α)
    (h : Static c idx s) (hr : run M (FPSuccess.parts c.layout c.regions (Host.scale c)).body s = some t) :
    Static c idx t := by
  have keep : ∀ d sh name, name ∈ Names → t.regs d sh name = s.regs d sh name := by
    intro d sh name hn
    exact run_keeps M _ name (body_keeps _ _ _ name hn) s t hr d sh
  simpa [Static,keep,Names] using h

theorem static_index (c : Host.Call) (idx : GridIndex c.grid) (s : State α) (h : Static c idx s) (step : Nat) :
    Static c idx (s.setReg "start_n" .int [] (fun _ => ((step*c.layout.BN : Nat) : Int))) := by
  simpa [Static,State.setReg] using h

/-- The source prelude establishes the full discrete invariant from the same
resolved metadata state that the FP32 success theorem uses. -/
theorem prelude_static (c : Host.Call) (idx : GridIndex c.grid) (M : Algebra α)
    (s : State α) (hm : FPSuccess.Metadata c s) :
    ∃ t, run (M.withDefaultPrecision .fp32) (FPSuccess.parts c.layout c.regions (Host.scale c)).before
      (FPSuccess.atIndex s idx) = some t ∧ Static c idx t ∧ t.mem = s.mem := by
  have hb := Host.batch_bound c idx
  have hstart := hm.1 _ hb
  have hlen := hm.2.1 _ hb
  have hprompt := hm.2.2 _ hb
  simp only [batchAt,Formula.batch,Host.Call.launchAt] at hstart hlen hprompt
  simp (config := {maxSteps := 500000}) [FPSuccess.parts,kernel,Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnFwd.original,ComputeKernel.surfaceBody,
    Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,Static,State.setReg,Algebra.withDefaultPrecision,
    resolvePrecision,numeric,numericLt,numericLe,natLt,natLe,integralFloorDiv,integralMod,
    ofFloat,toFloat,Region.cast,FPSuccess.atIndex,hstart,hlen,hprompt,
    Formula.head,Formula.queryRow,Host.Call.launchAt,batchAt,Formula.batch,bop,Nat.mul_comm]
  refine ⟨rfl,rfl,rfl,?_,?_,?_⟩
  · funext i
    simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm]
  · funext i
    simp [bop,Broadcast.leftIndex,Broadcast.rightIndex]
  · funext i
    simp only [endAt,seqAt,batchAt,Formula.batch,Host.Call.launchAt,endIndex,
      Nat.add_sub_cancel_right,Nat.cast_add,Nat.cast_mul,Nat.cast_min]
    rw [min_add_add_right]
    split_ifs with h
    · rw [min_eq_left (le_of_lt h)]
    · rw [min_eq_right (le_of_not_gt h)]


def indexState (c : Host.Call) (step : Nat) (s : State α) :=
  s.setReg "start_n" .int [] (fun _ => ((step*c.layout.BN : Nat) : Int))
theorem index_discrete (c : Host.Call) (idx : GridIndex c.grid) (step : Nat) (s : State α)
    (h : Static c idx s) : Discrete c idx step (indexState c step s) := by
  rcases h with ⟨hs,hh,hkv,hb,hm,hd,hn,hl,hp,he⟩
  constructor <;> simp_all [indexState,State.setReg]

/-- A reached loop header of the actual typed source. The relation uses the
source prelude/body execution, not a caller-supplied mathematical trace. -/
inductive Reached (c : Host.Call) (M : Algebra α) (idx : GridIndex c.grid) (initial : State α) : Nat → State α → Prop
  | first (s : State α)
      (run : FP.Structural.run (M.withDefaultPrecision .fp32)
        (FPSuccess.parts c.layout c.regions (Host.scale c)).before (FPSuccess.atIndex initial idx) = some s) :
      Reached c M idx initial 0 s
  | next (step : Nat) (s t : State α) (prior : Reached c M idx initial step s)
      (active : step < countAt c idx)
      (run : FP.Structural.run (M.withDefaultPrecision .fp32)
        (FPSuccess.parts c.layout c.regions (Host.scale c)).body (indexState c step s) = some t) :
      Reached c M idx initial (step+1) t

/-- Every finite source-loop prefix succeeds and retains all metadata and
control registers. Thus source_observations applies at every reached body. -/
theorem reached_prefix (c : Host.Call) (idx : GridIndex c.grid) (M : Algebra α)
    (dots : FPSuccess.DotSupport c.layout M) (initial : State α) (hm : FPSuccess.Metadata c initial)
    (step : Nat) (ht : step ≤ countAt c idx) :
    ∃ s, Reached c M idx initial step s ∧ FPSuccess.Ready c.layout s ∧ Static c idx s ∧ FPSuccess.Metadata c s := by
  induction step with
  | zero =>
    obtain ⟨s,hs,hr,hmem,hcontrols⟩ := FPSuccess.prelude_success c.layout c.regions (Host.scale c) M (FPSuccess.atIndex initial idx)
    obtain ⟨t,ht,hstatic,htmem⟩ := prelude_static c idx M initial hm
    have heq : s=t := Option.some.inj (hs.symm.trans ht)
    subst t
    refine ⟨s,Reached.first s hs,hr,hstatic,?_⟩
    simpa only [FPSuccess.Metadata,htmem] using hm
  | succ step ih =>
    obtain ⟨s,hs,hr,hstatic,hmeta⟩ := ih (by omega)
    obtain ⟨t,hrun,hready,hmem,hfixed⟩ := FPSuccess.body_success c.layout c.regions (Host.scale c)
      M dots (by rw [c.blockN]; exact Host.bm_positive c) s hr ((step*c.layout.BN : Nat) : Int)
    refine ⟨t,Reached.next step s t hs (by omega) hrun,hready,?_,?_⟩
    · exact static_preserved c idx _ _ _ (static_index c idx s hstatic step) hrun
    · simpa only [FPSuccess.Metadata,hmem] using hmeta

/-- Every reached source body has exactly the discrete observations bound to
its actual typed load/store/mask sites. No caller-provided trace is assumed. -/
def SourceBridge (c : Host.Call) (M : Algebra α) (initial : State α) : Prop :=
  Sites c.layout c.regions (Host.scale c) ∧
  ∀ (idx : GridIndex c.grid) step,step<countAt c idx →
    ∃ s,Reached c M idx initial step s ∧
      Discrete c idx step (indexState c step s) ∧
      Observed c idx step (M.withDefaultPrecision .fp32) (indexState c step s)

theorem source_bridge (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M)
    (initial : State α) (hm : FPSuccess.Metadata c initial) : SourceBridge c M initial := by
  refine ⟨source_sites _ _ _,?_⟩
  intro idx step ht
  obtain ⟨s,hs,hr,hstatic,hmeta⟩ := reached_prefix c idx M dots initial hm step (Nat.le_of_lt ht)
  exact ⟨s,hs,index_discrete c idx step s hstatic,
    source_observations c idx step _ _ (index_discrete c idx step s hstatic)⟩

#print axioms source_bridge
end
end VeriTile.Bench.Repairs.ContextAttnFwd.Native
