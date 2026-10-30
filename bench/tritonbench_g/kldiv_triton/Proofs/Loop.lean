import bench.tritonbench_g.kldiv_triton.Proofs.BlockMemory
import bench.tritonbench_g.kldiv_triton.Proofs.Math

namespace VeriTile.Bench.Repairs.KldivTriton.Tile
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (rounds rounds_cover lt_rounds_iff)
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000

def outputBase (base k reduction : Nat) : Nat := base + if reduction = 0 then 0 else k

def tick {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (yBase tBase base N B k : Nat) (flag : Bool) (reduction : Nat) : State α :=
  forwardResult m (s.setReg "i" .int [] (fun _ => ((k * B : Nat) : Int)))
    yBase tBase (outputBase base k reduction) N B (k * B) flag reduction

def iterate {α : Type} [Inhabited α] (m : Model α)
    (yBase tBase base N B : Nat) (flag : Bool) (reduction : Nat) : Nat → Nat → State α → State α
  | 0, _, s => s
  | count + 1, k, s => iterate m yBase tBase base N B flag reduction count (k + 1)
      (tick m s yBase tBase base N B k flag reduction)

theorem tick_header {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (yBase tBase base N B k : Nat) (flag : Bool) (reduction : Nat)
    (h : Header s yBase tBase (outputBase base k reduction) B) :
    Header (tick m s yBase tBase base N B k flag reduction)
      yBase tBase (outputBase base (k + 1) reduction) B := by
  have before : Header (s.setReg "i" .int [] (fun _ => ((k * B : Nat) : Int)))
      yBase tBase (outputBase base k reduction) B := by
    constructor <;> simp [h.y, h.t, h.o, h.base]
  have after := result_header m _ yBase tBase (outputBase base k reduction) N B (k * B) flag reduction before
  by_cases hr : reduction = 0 <;> simpa [tick, outputBase, hr, Nat.add_assoc] using after

theorem iterate_runs {α : Type} [Inhabited α] (m : Model α)
    (sy st so yBase tBase base N B : Nat) (flag : Bool) (reduction : Nat)
    (positive : 0 < B) (count k : Nat) (s : State α)
    (ends : k + count = rounds N B)
    (header : Header s yBase tBase (outputBase base k reduction) B) :
    rangeInt m.alg "i" ((k * B : Nat) : Int) (N : Int) B
      (forwardBody sy st so N B flag reduction) s =
        some (iterate m yBase tBase base N B flag reduction count k s) := by
  induction count generalizing k s with
  | zero =>
    have hk : k = rounds N B := by omega
    have covered : N ≤ k * B := by rw [hk]; exact rounds_cover N B positive
    rw [rangeInt]
    simp only [Nat.ne_of_gt positive, if_false, Int.ofNat_lt, not_lt.mpr covered, iterate]
  | succ count ih =>
    have active : k * B < N := (lt_rounds_iff N B k positive).mp (by omega)
    let c : Invocation α := ⟨m,sy,st,so,N,B,flag,reduction,yBase,tBase,
      outputBase base k reduction,k*B,s.setReg "i" .int [] (fun _ => ((k*B : Nat) : Int))⟩
    have stepRun := forward_step c
      (set_i_ready m s sy st so yBase tBase (outputBase base k reduction) N B (k*B) flag reduction header)
    have next : ((k * B : Nat) : Int) + (B : Int) = (((k + 1) * B : Nat) : Int) := by
      push_cast; ring
    rw [rangeInt]
    simp only [Nat.ne_of_gt positive, if_false, Int.ofNat_lt, active, if_true]
    rw [stepRun]
    simp only [Option.bind_some, next]
    exact ih (k + 1) _ (by omega) (tick_header m s yBase tBase base N B k flag reduction header)

theorem source_decomposition (sy st so N B : Nat) (flag : Bool) (reduction : Nat) :
    (forwardSource sy st so N B flag reduction).surfaceBody =
      (forwardSource sy st so N B flag reduction).surfaceBody.take 5 ++
      (forwardSource sy st so N B flag reduction).surfaceBody.drop 5 := by
  exact (List.take_append_drop 5 _).symm

/-- Every iteration of the actual source runs. The result is constructed by
the checked body step; there is no successful-kernel or output oracle premise. -/
def sourceRun {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (sy st so N B : Nat) (flag : Bool) (reduction : Nat) (positive : 0 < B) :
    {final : State α // run m.alg (forwardSource sy st so N B flag reduction).surfaceBody s = some final} := by
  let initial := forwardInitial s sy st so B
  let final := iterate m (s.pids 0*sy) (s.pids 0*st) (s.pids 0*so) N B flag reduction
    (rounds N B) 0 initial
  refine ⟨final, ?_⟩
  rw [source_decomposition, run_append, forward_initialize]
  simp only [Option.bind_some]
  have h := iterate_runs m sy st so (s.pids 0*sy) (s.pids 0*st) (s.pids 0*so)
    N B flag reduction positive (rounds N B) 0 initial (by omega)
    (by simpa [outputBase] using initial_header s sy st so B)
  simp only [forwardSource, forwardTemplate, ComputeKernel.surfaceBody, List.drop,
    run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind_eq_bind, Option.bind, Option.pure_def]
  change (rangeInt m.alg "i" 0 (N : Int) B (forwardBody sy st so N B flag reduction) initial).bind
    (fun t => some t) = some final
  have hh : rangeInt m.alg "i" 0 (N : Int) B (forwardBody sy st so N B flag reduction) initial =
      some final := by simpa only [Nat.zero_mul, Nat.cast_zero] using h
  rw [hh]
  rfl

#axiomsClean iterate_runs
#axiomsClean sourceRun
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Tile
