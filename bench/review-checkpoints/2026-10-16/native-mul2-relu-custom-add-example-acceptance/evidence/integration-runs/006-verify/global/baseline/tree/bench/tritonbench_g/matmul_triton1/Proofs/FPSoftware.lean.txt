import bench.tritonbench_g.matmul_triton1.Proofs.FPComplete
namespace VeriTile.Bench.TritonBenchG.MatmulTriton1.FPValue
open VeriTile.Triton FP.Structural FP.Equational
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Binary32 products and every addition in an explicit valid dot schedule.
Zero seeds are evaluated, never cancelled by a real arithmetic identity. -/
def treeValue (cfg : FP.Config) (xs : Fin n → Word) : ReductionTree n → Word
  | .input i => xs i
  | .zero => FP.Value.zero .fp32
  | .add a b => FP.add cfg (treeValue cfg xs a) (treeValue cfg xs b)

def dotValue (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → Word) (b : TileIndex (batch++[k,n]) → Word) :
    TileIndex (batch++[m,n]) → Word :=
  match batch with
  | [] => fun i => treeValue cfg (fun t => FP.mul cfg
      (a (i.1,t,PUnit.unit)) (b (t,i.2.1,PUnit.unit))) (plans k).tree
  | _::tail => fun i => dotValue cfg plans (batch:=tail) (fun j => a (i.1,j))
      (fun j => b (i.1,j)) i.2
termination_by batch.length

/-- Real binary32 inputs, products, dot additions and the separate outer add.
This is an unfused software graph with TF32 disabled in the source. A valid
plan is not a claim that its tree matches a compiler's native dot schedule. -/
def software (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) : Primitives Word where
  zero := FP.Value.zero .fp32
  add := FP.add cfg
  dot := fun _ _ {batch} {m} {k} {n} a b =>
    dotValue cfg plans (batch:=batch) (m:=m) (k:=k) (n:=n) a b

/-- The public full-loop software reference is independent of execution and
initial-state registers: it reads input cells and program IDs only. -/
noncomputable def expected (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) := accumulation (software cfg plans) p initial p.steps

theorem dot_retains_all_inputs (plan : ReductionPlan n) (k : Fin n) :
    some k ∈ plan.tree.leaves := by
  apply plan.valid.mem_iff.mpr
  simp

theorem fp32_rounding_vector :
    (FP.mul .ieeeValue (FP.Value.ofBits .fp32 0x3f800001)
      (FP.Value.ofBits .fp32 0x3f800001)).bits = BitVec.ofNat 32 0x3f800002 ∧
    (8388609/8388608 : Rat)*(8388609/8388608) ≠ 4194305/4194304 := by decide +kernel

#axiomsClean dot_retains_all_inputs
#axiomsClean fp32_rounding_vector
#specNonCircular expected avoiding [source, FP.Structural.exec, correctness]
end VeriTile.Bench.TritonBenchG.MatmulTriton1.FPValue
