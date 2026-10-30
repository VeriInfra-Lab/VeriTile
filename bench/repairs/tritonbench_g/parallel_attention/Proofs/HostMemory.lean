import bench.repairs.tritonbench_g.parallel_attention.Proofs.Execution

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

def OutputKind.cells (kind : OutputKind) (s : Shape) := s.B*s.H*s.T*kind.width s

def wholeRun (a : Ops α) (p : Plans) (s : Shape) (b : Bases) (kind : OutputKind)
    (tree : SumPlan) (scratch output : Nat) (schedule : List Nat) (m : Memory α) :=
  hostRun a tree (kind.shards s) scratch output (kind.cells s)
    (launch a p s b kind scratch schedule m)

/-- A word reference made solely from source scalar trees, including the actual
host shard reduction and final dtype cast. It is not the real specification. -/
def wordReference (a : Ops α) (p : Plans) (s : Shape) (kind : OutputKind)
    (tree : SumPlan) (x : TensorInputs α) (i : Nat) :=
  a.cast (tree.eval a (fun shard => if shard<kind.shards s ∧ i<kind.cells s then
    scratchValue a p s kind x (shard*kind.cells s+i) else a.zero))

theorem whole_output (a : Ops α) (p : Plans) (s : Shape) (b : Bases) (kind : OutputKind)
    (tree : SumPlan) (scratch output : Nat) (schedule : List Nat)
    (covers : schedule.Perm (List.range (kind.count s)))
    (scratch_fresh : ∀ i, i<kind.count s → scratch+i∉inputCells s b)
    (output_fresh : ∀ i, i<kind.cells s → output+i∉scratchCells (kind.shards s) scratch (kind.cells s))
    (m : Memory α) (i : Nat) (hi : i<kind.cells s) :
    wholeRun a p s b kind tree scratch output schedule m (output+i)=
      wordReference a p s kind tree (currentInputs a s b m) i := by
  unfold wholeRun
  rw [host_output a tree (kind.shards s) scratch output (kind.cells s) _ output_fresh i hi]
  have eq : (fun shard => if shard<kind.shards s ∧ i<kind.cells s then
      launch a p s b kind scratch schedule m (scratch+shard*kind.cells s+i) else a.zero)=
    (fun shard => if shard<kind.shards s ∧ i<kind.cells s then
      scratchValue a p s kind (currentInputs a s b m) (shard*kind.cells s+i) else a.zero) := by
    funext shard
    split_ifs with h
    · have bound : shard*kind.cells s+i<kind.count s := by
        change shard*kind.cells s+i<kind.shards s*kind.cells s
        nlinarith
      simpa [Nat.add_assoc] using launch_output a p s b kind scratch schedule covers scratch_fresh m _ bound
    · rfl
  simpa only [hostValue,wordReference,Nat.add_assoc] using
    congrArg (fun x : Nat → α => a.cast (tree.eval a x)) eq

theorem whole_frame (a : Ops α) (p : Plans) (s : Shape) (b : Bases) (kind : OutputKind)
    (tree : SumPlan) (scratch output : Nat) (schedule : List Nat)
    (covers : schedule.Perm (List.range (kind.count s))) (m : Memory α) (address : Nat)
    (scratch_outside : address<scratch ∨ scratch+kind.count s≤address)
    (output_outside : address<output ∨ output+kind.cells s≤address) :
    wholeRun a p s b kind tree scratch output schedule m address=m address := by
  unfold wholeRun
  rw [host_frame a tree (kind.shards s) scratch output (kind.cells s) _ address output_outside]
  exact launch_frame a p s b kind scratch schedule covers m address scratch_outside

theorem wordReference_real (p : Plans) (s : Shape) (kind : OutputKind)
    (tree : SumPlan) (complete : tree.Covers (kind.shards s)) (x : TensorInputs ℝ)
    (i : Nat) (hi : i<kind.cells s) :
    wordReference realOps p s kind tree x i=
      ∑ shard∈Finset.range (kind.shards s),scratchValue realOps p s kind x (shard*kind.cells s+i) := by
  unfold wordReference
  rw [tree.real_sum _ complete]
  change (∑ shard∈Finset.range (kind.shards s),if shard<kind.shards s ∧ i<kind.cells s then _ else (0:ℝ))=_
  apply Finset.sum_congr rfl
  intro shard hshard
  simp only [Finset.mem_range] at hshard
  rw [if_pos ⟨hshard,hi⟩]

#axiomsClean whole_output
#axiomsClean whole_frame
#axiomsClean wordReference_real
end
end VeriTile.Bench.Repairs.ParallelAttention
