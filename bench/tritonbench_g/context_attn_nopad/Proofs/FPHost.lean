import bench.tritonbench_g.context_attn_nopad.Proofs.HostConfig
import bench.tritonbench_g.context_attn_nopad.Proofs.FP
import VeriTile.Triton.Float.ProfiledSoundness

namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.FPHost
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open Host
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000

/-- Selected production profile: fp32 storage and work, preserving both dot
operations, natural exponential, the -infinity mask and probability/output casts. -/
def original (c : Call) (r : Program.Regions) : FP.ProfiledRewrite.Program :=
  FPProof.original .fp32 r.query r.key r.value c.scale r.start r.sequence r.out
    c.q.rowStride c.q.headStride c.q.dimStride
    c.k.rowStride c.k.headStride c.k.dimStride
    c.v.rowStride c.v.headStride c.v.dimStride
    c.o.rowStride c.o.headStride c.o.dimStride 128 c.q.dims 128

def candidate (c : Call) (r : Program.Regions) : FP.ProfiledRewrite.Program :=
  FPProof.scaleFirst .fp32 r.query r.key r.value c.scale r.start r.sequence r.out
    c.q.rowStride c.q.headStride c.q.dimStride
    c.k.rowStride c.k.headStride c.k.dimStride
    c.v.rowStride c.v.headStride c.v.dimStride
    c.o.rowStride c.o.headStride c.o.dimStride 128 c.q.dims 128

theorem original_source (c : Call) (r : Program.Regions) :
    (original c r).kernel = c.source r Bool.false := rfl
theorem candidate_source (c : Call) (r : Program.Regions) :
    (candidate c r).kernel = c.source r Bool.true := rfl

theorem program_equiv (R : FP.ScalarArithmetic.Rules) (c : Call) (r : Program.Regions) :
    original c r ≡[R] candidate c r := by
  exact FPProof.equivalent .fp32 r.query r.key r.value c.scale r.start r.sequence r.out
    c.q.rowStride c.q.headStride c.q.dimStride
    c.k.rowStride c.k.headStride c.k.dimStride
    c.v.rowStride c.v.headStride c.v.dimStride
    c.o.rowStride c.o.headStride c.o.dimStride 128 c.q.dims 128 R

def atProgram {α : Type} (c : Call) (idx : GridIndex c.grid) (s : State α) : State α :=
  { s with pids := idx.toPids, numPids := c.grid.toNumPids }

/-- Every actual program index occurs once. The equivalence below also holds
for every other order, with guards checked on that order's reached states. -/
def jobs (c : Call) : List (GridIndex c.grid) := Finset.univ.toList

theorem jobs_complete (c : Call) : (jobs c).Nodup ∧ ∀ idx : GridIndex c.grid, idx ∈ jobs c := by
  exact ⟨Finset.nodup_toList _, by simp [jobs]⟩

def runJobs {α : Type} [Inhabited α] (M : Algebra α) (c : Call) (r : Program.Regions)
    (swapped : Bool) : List (GridIndex c.grid) → State α → Option (State α)
  | [], s => some s
  | idx :: rest, s =>
      (FP.Structural.exec (engine M) (c.source r swapped) (atProgram c idx s)).bind
        (runJobs M c r swapped rest)

/-- The original numerical domain is checked before each actual reached
program, then at each reached qk/scale multiplication inside its loop. No
successful execution or final output is required by this input condition. -/
def Domain {α : Type} [Inhabited α] (M : Algebra α) (D : FP.Guarded.Domain α)
    (c : Call) (r : Program.Regions) : List (GridIndex c.grid) → State α → Prop
  | [], _ => True
  | idx :: rest, s =>
      (∀ site ∈ (original c r).domain, site.Holds M D (atProgram c idx s)) ∧
      (∀ t, FP.Structural.exec (engine M) (c.source r Bool.false) (atProgram c idx s) = some t →
        Domain M D c r rest t)

theorem runJobs_equiv (R : FP.ScalarArithmetic.Rules) {α : Type} [Inhabited α]
    (M : Algebra α) (D : FP.Guarded.Domain α) (hM : FP.Guarded.Models R.assumptions M D)
    (c : Call) (r : Program.Regions) (order : List (GridIndex c.grid)) (s : State α)
    (hD : Domain M D c r order s) :
    runJobs M c r Bool.false order s = runJobs M c r Bool.true order s := by
  induction order generalizing s with
  | nil => rfl
  | cons idx rest ih =>
    have hp := FP.ProfiledRewrite.sound (FPProof.equivalent .fp32
      r.query r.key r.value c.scale r.start r.sequence r.out
      c.q.rowStride c.q.headStride c.q.dimStride
      c.k.rowStride c.k.headStride c.k.dimStride
      c.v.rowStride c.v.headStride c.v.dimStride
      c.o.rowStride c.o.headStride c.o.dimStride 128 c.q.dims 128 R)
      α M D hM (atProgram c idx s) hD.1
    change FP.Structural.exec (engine M) (c.source r Bool.false) (atProgram c idx s) =
      FP.Structural.exec (engine M) (c.source r Bool.true) (atProgram c idx s) at hp
    simp only [runJobs]
    rw [← hp]
    cases he : FP.Structural.exec (engine M) (c.source r Bool.false) (atProgram c idx s) with
    | none => rfl
    | some t => exact ih t (hD.2 t he)

def runHost {α : Type} [Inhabited α] (M : Algebra α) (c : Call) (r : Program.Regions)
    (swapped : Bool) (s : State α) : Option (State α × Option Unit) :=
  (runJobs M c r swapped (jobs c) s).map (fun t => (t, pythonReturn))

theorem host_equiv (R : FP.ScalarArithmetic.Rules) {α : Type} [Inhabited α]
    (M : Algebra α) (D : FP.Guarded.Domain α) (hM : FP.Guarded.Models R.assumptions M D)
    (c : Call) (r : Program.Regions) (s : State α) (hD : Domain M D c r (jobs c) s) :
    runHost M c r Bool.false s = runHost M c r Bool.true s := by
  unfold runHost
  rw [runJobs_equiv R M D hM c r (jobs c) s hD]

#print_fp_assumptions program_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.FPHost
