import bench.tritonbench_g.chunk_gla_simple.Proofs.FP
import bench.tritonbench_g.chunk_gla_simple.Proofs.HostGeometry
import VeriTile.Triton.Float.ProfiledSoundness

/-! FP host composition imports no Real execution proof. Every actual grid
instance receives the source PIDs and the reached finite-total/scale guard.
Equality includes the entire optional result, hence successful return values,
all registers and tagged memory, as well as matching failure behavior. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.HostFP
open VeriTile Triton FP.Structural FP.Guarded FP.ProfiledRewrite Host
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 2000000

def program (c : Shape) (scale : ℝ) (swapped : Bool) : Program where
  kernel := Host.kernel c scale swapped
  domain := FPProof.requirements (Host.kernel c scale Bool.false) .fp32 c.BT c.BV scale

theorem source_equiv (c : Shape) (scale : ℝ) (R : FP.ScalarArithmetic.Rules) :
    program c scale Bool.false ≡[R] program c scale Bool.true := by
  exact FPProof.equivalent .fp32 .fp32 .fp32 .fp32 "q" "k" "v" "h" "g" "o"
    (c.T*c.K) c.K (c.T*c.V) c.V (c.NT*c.K*c.V) c.V scale c.T c.K c.V c.BT c.BK c.BV R

def atProgram {α : Type} (c : Shape) (s : State α) (idx : GridIndex (grid c)) : State α :=
  { s with pids := idx.toPids, numPids := (grid c).toNumPids }

/-- Only numerical operands of the actual final multiply are assumptions.
No bounds, successful execution, output equality or launch proof is hidden here. -/
def ReachedDomain {α : Type} [Inhabited α] (c : Shape) (scale : ℝ)
    (M : Algebra α) (D : Domain α) (s : State α) : Prop :=
  ∀ idx : GridIndex (grid c), ∀ site ∈ (program c scale Bool.false).domain,
    site.Holds M D (atProgram c s idx)

def gridExecution {α : Type} [Inhabited α] (c : Shape) (scale : ℝ) (swapped : Bool)
    (M : Algebra α) (s : State α) : GridIndex (grid c) → Option (State α) :=
  fun idx => FP.Structural.exec (engine M) (Host.kernel c scale swapped) (atProgram c s idx)

/-- Actual per-program states agree at every host grid index. -/
theorem grid_equiv (c : Shape) (scale : ℝ) (R : FP.ScalarArithmetic.Rules)
    {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (hd : ReachedDomain c scale M D s) :
    gridExecution c scale Bool.false M s = gridExecution c scale Bool.true M s := by
  funext idx
  exact FP.ProfiledRewrite.sound (source_equiv c scale R) α M D hM (atProgram c s idx) (hd idx)

/-- Returned O tensor observed at the unique writer proved by host geometry.
`Option` preserves execution failure rather than supplying a fabricated value. -/
def returned {α : Type} [Inhabited α] (c : Shape) (ht : 0 < c.BT) (scale : ℝ) (swapped : Bool)
    (M : Algebra α) (s : State α) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) : Option α :=
  (gridExecution c scale swapped M s (owner c ht h t v)).map
    fun result => (result.mem "o" (h.val*(c.T*c.V)+t.val*c.V+v.val)).read .real

theorem returned_equiv (c : Shape) (ht : 0 < c.BT) (scale : ℝ) (R : FP.ScalarArithmetic.Rules)
    {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (hd : ReachedDomain c scale M D s) :
    returned c ht scale Bool.false M s = returned c ht scale Bool.true M s := by
  unfold returned
  rw [grid_equiv c scale R M D hM s hd]

#print_fp_assumptions source_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.HostFP
