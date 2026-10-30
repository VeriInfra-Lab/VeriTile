import bench.tritonbench_g.kv_cache_filling.Proofs.Schedule

namespace VeriTile.Bench.Repairs.KvCacheFilling

theorem Tree.mem_substitute (t : Tree) (f : Nat → Tree) (i : Nat) :
    i ∈ (t.substitute f).leaves ↔ ∃ j ∈ t.leaves, i ∈ (f j).leaves := by
  induction t with
  | leaf j => simp [Tree.substitute, Tree.leaves]
  | node a b ia ib => simp [Tree.substitute, Tree.leaves, ia, ib, or_and_right, exists_or]

theorem Tree.eval_substitute {α : Type} (t : Tree) (f : Nat → Tree) (op : α → α → α) (x : Nat → α) :
    (t.substitute f).eval op x = t.eval op (fun j => (f j).eval op x) := by
  induction t with
  | leaf i => rfl
  | node a b ia ib => simp [Tree.substitute, Tree.eval, ia, ib]

def Plan.substitute {m n : Nat} (p : Plan m) (f : Nat → Tree)
    (valid : ∀ j, j < m → ∀ i ∈ (f j).leaves, i < n)
    (covers : ∀ i, i < n → ∃ j, j < m ∧ i ∈ (f j).leaves) : Plan n where
  low := p.low.substitute f
  high := p.high.substitute f
  low_valid := by
    intro i hi; obtain ⟨j,hj,hij⟩ := (Tree.mem_substitute ..).mp hi
    exact valid j (p.low_valid j hj) i hij
  high_valid := by
    intro i hi; obtain ⟨j,hj,hij⟩ := (Tree.mem_substitute ..).mp hi
    exact valid j (p.high_valid j hj) i hij
  low_covers := by
    intro i hi; obtain ⟨j,hj,hij⟩ := covers i hi
    exact (Tree.mem_substitute ..).mpr ⟨j,p.low_covers j hj,hij⟩
  high_covers := by
    intro i hi; obtain ⟨j,hj,hij⟩ := covers i hi
    exact (Tree.mem_substitute ..).mpr ⟨j,p.high_covers j hj,hij⟩

def paddedPlan {n lanes : Nat} (hn : 0 < n) (hl : n ≤ lanes) (p : Plan lanes) : Plan n :=
  p.substitute (fun j => .leaf (padIndex n j))
    (by intro j hj i hi; simp [Tree.leaves] at hi; subst i; unfold padIndex; split_ifs <;> omega)
    (by intro i hi; exact ⟨i,lt_of_lt_of_le hi hl,by simp [Tree.leaves,padIndex,hi]⟩)

def packedPlan {n lanes : Nat} (hn : 0 < n) (hl : n ≤ lanes) (p : Plan lanes) : Plan (2*n) :=
  p.substitute (pairLeaves n)
    (by
      intro j hj i hi
      have hp : padIndex n j < n := by unfold padIndex; split_ifs <;> omega
      simp [pairLeaves,Tree.leaves] at hi
      rcases hi with h | h <;> omega)
    (by
      intro i hi
      by_cases h : i < n
      · exact ⟨i,lt_of_lt_of_le h hl,by simp [pairLeaves,Tree.leaves,padIndex,h]⟩
      · refine ⟨i-n,by omega,?_⟩
        have hs : i-n < n := by omega
        simp [pairLeaves,Tree.leaves,padIndex,hs]
        omega)

/-- These are precisely the active mask and fallback in the source loads. -/
theorem paddedLoad_source {α : Type} (zero : α) (n : Nat) (x : Nat → α) (i : Nat) :
    paddedLoad zero n x i = x (padIndex n i) := by
  simp [paddedLoad,maskedLoad,padIndex]; split_ifs <;> rfl

theorem padded_reduction {α : Type} (zero : α) (n : Nat) (x : Nat → α) (op : α → α → α) (t : Tree) :
    t.eval op (paddedLoad zero n x) = (t.substitute (fun i => .leaf (padIndex n i))).eval op x := by
  rw [Tree.eval_substitute]
  congr 1
  funext i
  exact paddedLoad_source zero n x i

theorem pair_reduction {α : Type} (zero : α) (n : Nat) (x : Nat → α) (op : α → α → α) (t : Tree) :
    t.eval op (fun i => op (paddedLoad zero n x i) (paddedLoad zero n (fun j => x (j+n)) i)) =
      (t.substitute (pairLeaves n)).eval op x := by
  rw [Tree.eval_substitute]
  congr 1
  funext i
  simp [pairLeaves,Tree.eval,paddedLoad_source]

/-- Register-level straight-line scalar producer, shared by the two half
calls in policy 4. The ordered assignments match _quant_row. -/
theorem quantSource_exec {α β : Type} (ops : Operations α β) (levels : Nat) (x : Nat → α)
    (low high : Tree) (i : Nat) :
    let regs := runQuantSource ops levels (x i) (low.eval ops.minimum x) (high.eval ops.maximum x)
    ops.uint8 (regs .rounded) = (quantize ops levels x low high).code i ∧
    regs .scale = (quantize ops levels x low high).scale ∧
    regs .zero = (quantize ops levels x low high).zero := by
  simp [runQuantSource,quantSource,QExpr.eval,quantize,Function.update]

/-- Repeated half calls have identical scale and zero because only the input
register differs; metadata is produced by these same assignments. -/
theorem quantSource_metadata {α β : Type} (ops : Operations α β) (levels : Nat) (x y lo hi : α) :
    runQuantSource ops levels x lo hi .scale = runQuantSource ops levels y lo hi .scale ∧
    runQuantSource ops levels x lo hi .zero = runQuantSource ops levels y lo hi .zero := by
  simp [runQuantSource,quantSource,QExpr.eval,Function.update]

theorem source_lane_bound (r : Request) (d : Domain r) (j : Job) (hj : j ∈ jobs r)
    (width i : Nat) (hi : i < width) : sourceRow r j * width + i < r.tokens * r.heads * width := by
  have hr := sourceRow_bound r d j hj
  calc
    _ < sourceRow r j * width + width := Nat.add_lt_add_left hi _
    _ = (sourceRow r j + 1) * width := by ring
    _ ≤ r.tokens * r.heads * width := Nat.mul_le_mul_right _ (by omega)


/-- Source reduction wiring. The lane trees are compiler-selected ordered
binary trees; the bindings record padded loads and paired int4 extrema. -/
theorem sourceQuant_exec {α β : Type} (ops : Operations α β) (levels : Nat) (x : Nat → α) (low high : Tree) :
    sourceQuant ops levels x (low.eval ops.minimum x) (high.eval ops.maximum x) = quantize ops levels x low high := by
  simp [sourceQuant,runQuantSource,quantSource,QExpr.eval,quantize,Function.update]

/-- Operational interpretation of the three actual _fill_rows branches.
It retains masked loads, padding, pairwise extrema, scalar assignment
execution, uint8 conversion and low/high byte packing. Stores are masked
by rowAddresses, whose complete set and bounds are proved separately. -/
theorem sourceRowExec_value {α β : Type} {n : Nat} (ops : Operations α β) (policy : Policy)
    (width : Nat) (x : Nat → α) (p : Plan n) (binding : SourceBinding p policy width) (i : Nat) (hi : i < width) :
    (sourceRowExec ops policy width x binding.reduction.low binding.reduction.high).value i =
      (rowKernel ops policy width x p.low p.high).value i := by
  have hl := binding.low
  have hh := binding.high
  cases policy with
  | copy => simp [sourceRowExec,rowKernel,maskedLoad,hi]
  | int8 =>
    simp only [sourceRowExec,padded_reduction,← hl,← hh,sourceQuant_exec,rowKernel]
  | int4 =>
    simp only [sourceRowExec,pair_reduction,← hl,← hh,sourceQuant_exec,rowKernel]

theorem sourceRowExec_metadata {α β : Type} {n : Nat} (ops : Operations α β) (policy : Policy)
    (width : Nat) (x : Nat → α) (p : Plan n) (binding : SourceBinding p policy width) :
    (sourceRowExec ops policy width x binding.reduction.low binding.reduction.high).scale =
      (rowKernel ops policy width x p.low p.high).scale ∧
    (sourceRowExec ops policy width x binding.reduction.low binding.reduction.high).zero =
      (rowKernel ops policy width x p.low p.high).zero := by
  have hl := binding.low
  have hh := binding.high
  cases policy with
  | copy => exact ⟨rfl,rfl⟩
  | int8 => simp [sourceRowExec,padded_reduction,← hl,← hh,sourceQuant_exec,rowKernel]
  | int4 => simp [sourceRowExec,pair_reduction,← hl,← hh,sourceQuant_exec,rowKernel]

end VeriTile.Bench.Repairs.KvCacheFilling
