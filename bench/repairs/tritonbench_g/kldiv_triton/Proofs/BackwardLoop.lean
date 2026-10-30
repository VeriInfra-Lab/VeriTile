import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Backward
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.ForwardIO

namespace VeriTile.Bench.Repairs.KldivTriton.Backward
open VeriTile.Triton FP.Structural
open Tile (Model Invocation input loaded gradient)
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (rounds rounds_cover lt_rounds_iff)
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole (scatter_regs)
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 4000000

def invocation {α : Type} (m : Model α) (s : State α) (so st N B tBase oBase current : Nat)
    (flag : Bool) : Invocation α := ⟨m,0,st,so,N,B,flag,0,0,tBase,oBase,current,s⟩

structure Header {α : Type} (s : State α) (tBase oBase : Nat) : Prop where
  t : s.regs .ptr [] "target_ptr" = some (fun _ => (("target_ptr" : RegionName), tBase))
  o : s.regs .ptr [] "input_ptr" = some (fun _ => (("input_ptr" : RegionName), oBase))

def tick {α : Type} [Inhabited α] (m : Model α) (s : State α) (so st N B tBase oBase k : Nat)
    (flag : Bool) : State α :=
  b5 (invocation m (s.setReg "i" .int [] (fun _ => ((k * B : Nat) : Int))) so st N B tBase oBase (k*B) flag)

theorem tick_header {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (so st N B tBase oBase k : Nat) (flag : Bool) (h : Header s tBase oBase) :
    Header (tick m s so st N B tBase oBase k flag) tBase oBase := by
  constructor <;> simp [tick, b5, b4, b3, b2, b1, invocation, Tile.scatter, scatter_regs, h.t, h.o]

theorem tick_inputs {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (so st N B tBase oBase k : Nat) (flag : Bool) :
    input (tick m s so st N B tBase oBase k flag) "target_ptr" tBase = input s "target_ptr" tBase := by
  funext j
  unfold input tick b5
  rw [Tile.scatter_preserves _ _ _ _ _ _ _ "target_ptr" (tBase+j) (Or.inl (by decide))]
  rfl

theorem tick_cell {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (so st N B tBase oBase k : Nat) (flag : Bool) (lane : Fin B) (active : k*B+lane.val<N) :
    (tick m s so st N B tBase oBase k flag).mem "input_ptr" (oBase+(k*B+lane.val)) =
      .mk .real (gradient m flag (m.alg.fp32Load (input s "target_ptr" tBase (k*B+lane.val)))) := by
  unfold tick b5
  simp only [invocation]
  rw [Tile.scatter_cell _ _ _ _ _ _ _ (lane,PUnit.unit) active]
  simp [invocation, loaded, active, input]

theorem tick_frame {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (so st N B tBase oBase k : Nat) (flag : Bool) (region : RegionName) (a : Nat)
    (outside : region ≠ "input_ptr" ∨ ∀ lane : Fin B, k*B+lane.val<N → a≠oBase+(k*B+lane.val)) :
    (tick m s so st N B tBase oBase k flag).mem region a = s.mem region a := by
  unfold tick b5
  rw [Tile.scatter_preserves]
  · rfl
  · rcases outside with h | h
    · exact Or.inl h
    · exact Or.inr (fun j => h j.1)

def iterate {α : Type} [Inhabited α] (m : Model α) (so st N B tBase oBase : Nat)
    (flag : Bool) : Nat → Nat → State α → State α
  | 0, _, s => s
  | count+1, k, s => iterate m so st N B tBase oBase flag count (k+1)
      (tick m s so st N B tBase oBase k flag)

theorem iterate_runs {α : Type} [Inhabited α] (m : Model α)
    (so st N B tBase oBase : Nat) (flag : Bool) (positive : 0<B)
    (count k : Nat) (s : State α) (ends : k+count=rounds N B) (h : Header s tBase oBase) :
    rangeInt m.alg "i" ((k*B : Nat) : Int) (N : Int) B (body so st N B flag) s =
      some (iterate m so st N B tBase oBase flag count k s) := by
  induction count generalizing k s with
  | zero =>
    have hk : k=rounds N B := by omega
    have covered : N≤k*B := by rw [hk];exact rounds_cover N B positive
    rw [rangeInt]
    simp only [Nat.ne_of_gt positive,if_false,Int.ofNat_lt,not_lt.mpr covered,iterate]
  | succ count ih =>
    have active : k*B<N := (lt_rounds_iff N B k positive).mp (by omega)
    let c := invocation m (s.setReg "i" .int [] (fun _ => ((k*B : Nat) : Int))) so st N B tBase oBase (k*B) flag
    have ready : Ready c := by constructor <;> simp [c,invocation,h.t,h.o]
    have stepRun := block c ready
    dsimp only [c, invocation] at stepRun
    have next : ((k*B : Nat) : Int)+(B:Int)=(((k+1)*B:Nat):Int) := by push_cast;ring
    rw [rangeInt]
    simp only [Nat.ne_of_gt positive,if_false,Int.ofNat_lt,active,if_true]
    rw [stepRun]
    simp only [next]
    exact ih (k+1) _ (by omega) (tick_header m s so st N B tBase oBase k flag h)

def initial {α : Type} (m : Model α) (s : State α) (so st N B : Nat) (flag : Bool) : State α :=
  a5 (invocation m s so st N B 0 0 0 flag)

def sourceRun {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (so st N B : Nat) (flag : Bool) (positive : 0<B) :
    {final : State α // run m.alg (backwardSource so st N B flag).surfaceBody s=some final} := by
  let start := initial m s so st N B flag
  let final := iterate m so st N B (s.pids 0*st) (s.pids 0*so) flag (rounds N B) 0 start
  refine ⟨final, ?_⟩
  have initialized := source_initialize (invocation m s so st N B 0 0 0 flag)
  dsimp only [invocation] at initialized
  rw [←List.take_append_drop 5 (backwardSource so st N B flag).surfaceBody,run_append,initialized]
  simp only [Option.bind_some]
  have header : Header start (s.pids 0*st) (s.pids 0*so) := by
    constructor <;> simp [start,initial,a5,a4,a3,a2,a1,a0,invocation]
  have h := iterate_runs m so st N B (s.pids 0*st) (s.pids 0*so) flag positive
    (rounds N B) 0 start (by omega) header
  simp only [backwardSource,backwardTemplate,ComputeKernel.surfaceBody,List.drop,
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind_eq_bind,Option.bind,Option.pure_def]
  change (rangeInt m.alg "i" 0 (N:Int) B (body so st N B flag) start).bind (fun t => some t)=some final
  have hh : rangeInt m.alg "i" 0 (N:Int) B (body so st N B flag) start=some final := by
    simpa only [Nat.zero_mul,Nat.cast_zero] using h
  rw [hh]
  rfl

theorem iterate_frame {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (so st N B tBase oBase count k : Nat) (flag : Bool) (region : RegionName) (a : Nat)
    (outside : region ≠ "input_ptr" ∨ a<oBase ∨ oBase+N≤a) :
    (iterate m so st N B tBase oBase flag count k s).mem region a=s.mem region a := by
  induction count generalizing k s with
  | zero => rfl
  | succ count ih =>
    rw [iterate,ih]
    apply tick_frame
    rcases outside with h | h | h
    · exact Or.inl h
    · right;intro j _;omega
    · right;intro j active;omega

theorem iterate_before {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (so st N B tBase oBase count k : Nat) (flag : Bool) (a : Nat) (before : a<oBase+k*B) :
    (iterate m so st N B tBase oBase flag count k s).mem "input_ptr" a=s.mem "input_ptr" a := by
  induction count generalizing k s with
  | zero => rfl
  | succ count ih =>
    have next : a<oBase+(k+1)*B := by simp only [Nat.add_mul,Nat.one_mul];omega
    rw [iterate,ih _ (k+1) next]
    apply tick_frame
    right;intro j _;omega

theorem iterate_cell {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (so st N B tBase oBase count k q : Nat) (flag : Bool) (lane : Fin B)
    (lower : k≤q) (upper : q<k+count) (active : q*B+lane.val<N) :
    (iterate m so st N B tBase oBase flag count k s).mem "input_ptr" (oBase+(q*B+lane.val))=
      .mk .real (gradient m flag (m.alg.fp32Load (input s "target_ptr" tBase (q*B+lane.val)))) := by
  induction count generalizing k s with
  | zero => omega
  | succ count ih =>
    rw [iterate]
    by_cases same : q=k
    · subst q
      rw [iterate_before m _ so st N B tBase oBase count (k+1) flag _ (by
        simp only [Nat.add_mul,Nat.one_mul];have := lane.isLt;omega)]
      exact tick_cell m s so st N B tBase oBase k flag lane active
    · rw [ih _ (k+1) (by omega) (by omega),tick_inputs]

theorem source_cell {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (so st N B : Nat) (flag : Bool) (positive : 0<B) (i : Nat) (active : i<N) :
    (sourceRun m s so st N B flag positive).val.mem "input_ptr" (s.pids 0*so+i)=
      .mk .real (gradient m flag (m.alg.fp32Load (input s "target_ptr" (s.pids 0*st) i))) := by
  have cover : i/B*B+i%B=i := by simpa only [Nat.mul_comm] using Nat.div_add_mod i B
  have q : i/B<rounds N B := (lt_rounds_iff N B (i/B) positive).mpr (by omega)
  have h := iterate_cell m (initial m s so st N B flag) so st N B (s.pids 0*st) (s.pids 0*so)
    (rounds N B) 0 (i/B) flag ⟨i%B,Nat.mod_lt i positive⟩ (Nat.zero_le _)
    (by simpa only [Nat.zero_add] using q) (by simpa only [cover] using active)
  simpa only [sourceRun,cover,input,initial,a5,a4,a3,a2,a1,a0,State.setReg_mem,invocation] using h

theorem source_frame {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (so st N B : Nat) (flag : Bool) (positive : 0<B) (region : RegionName) (a : Nat)
    (outside : region ≠ "input_ptr" ∨ a<s.pids 0*so ∨ s.pids 0*so+N≤a) :
    (sourceRun m s so st N B flag positive).val.mem region a=s.mem region a := by
  simpa only [sourceRun, initial, a5, a4, a3, a2, a1, a0, invocation, State.setReg_mem] using
    iterate_frame m (initial m s so st N B flag) so st N B (s.pids 0*st) (s.pids 0*so)
      (rounds N B) 0 flag region a outside

#axiomsClean sourceRun
#axiomsClean source_cell
#axiomsClean source_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Backward
