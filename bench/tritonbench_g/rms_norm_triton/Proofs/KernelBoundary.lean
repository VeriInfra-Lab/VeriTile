import bench.tritonbench_g.rms_norm_triton.Proofs.WholeMath


namespace VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

/-- The source library's fixed six-shift next_power_of_2 helper. -/
def nextPower (n : Nat) : Nat :=
  if n = 0 then 0 else
    let x := n - 1
    let x := x ||| (x >>> 1)
    let x := x ||| (x >>> 2)
    let x := x ||| (x >>> 4)
    let x := x ||| (x >>> 8)
    let x := x ||| (x >>> 16)
    let x := x ||| (x >>> 32)
    x + 1

theorem nextPower_covers (n : Nat) : n ≤ nextPower n := by
  by_cases zero : n = 0
  · simp [zero,nextPower]
  let x1 := (n-1) ||| ((n-1) >>> 1)
  let x2 := x1 ||| (x1 >>> 2)
  let x3 := x2 ||| (x2 >>> 4)
  let x4 := x3 ||| (x3 >>> 8)
  let x5 := x4 ||| (x4 >>> 16)
  let x6 := x5 ||| (x5 >>> 32)
  have h1 : n-1 ≤ x1 := Nat.left_le_or
  have h2 : x1 ≤ x2 := Nat.left_le_or
  have h3 : x2 ≤ x3 := Nat.left_le_or
  have h4 : x3 ≤ x4 := Nat.left_le_or
  have h5 : x4 ≤ x5 := Nat.left_le_or
  have h6 : x5 ≤ x6 := Nat.left_le_or
  have bound := h1.trans (h2.trans (h3.trans (h4.trans (h5.trans h6))))
  simp only [nextPower,if_neg zero]
  change n ≤ x6+1
  omega

theorem shape_product (h : InputDomain input weight normalized preexisting) :
    input.numel = rows input normalized * columns normalized := by
  have split := congrArg List.prod (List.take_append_drop (dim input normalized) input.shape)
  simpa only [List.prod_append,h.suffix,Tensor.numel,rows,columns] using split.symm

def Prepared.layout (a : Prepared α input weight entry preexisting) (normalized : List Nat) : Layout :=
  ⟨"X","W","Y",rows input normalized,columns normalized,nextPower (columns normalized),nextPower_covers _⟩

def Prepared.state [Inhabited α] (a : Prepared α input weight entry preexisting)
    (normalized : List Nat) : State α where
  mem := fun region i => Cell.mk .real
    (if region = ("X" : RegionName) then a.output.memory a.xc.tensor.allocation (a.xc.tensor.base+i)
     else if region = ("W" : RegionName) then a.output.memory a.wc.tensor.allocation (a.wc.tensor.base+i)
     else a.output.memory a.output.tensor.allocation (a.output.tensor.base+i))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun ax => if ax = 0 then rows input normalized else 1
  undef := fun d _ _ => defaultValue d

theorem Prepared.input_read [Inhabited α] (a : Prepared α input weight entry preexisting)
    (normalized : List Nat) (i : Nat) (hi : i < input.numel) :
    ((a.state normalized).mem "X" i).read .real = input.read entry i := by
  have hi' : i < a.xc.tensor.numel := by simpa [Tensor.numel,a.xc.shape] using hi
  simp only [Prepared.state,if_true,Cell.read_same]
  rw [a.output.frame _ _ (Or.inr (Or.inl rfl)), a.wc.frame _ _ (Or.inr rfl)]
  have copied := a.xc.values i hi
  simpa only [Tensor.read,a.xc.contiguous i hi'] using copied

theorem Prepared.weight_read [Inhabited α] (a : Prepared α input weight entry preexisting)
    (normalized : List Nat) (existed : preexisting weight.allocation)
    (i : Nat) (hi : i < weight.numel) :
    ((a.state normalized).mem "W" i).read .real = weight.read entry i := by
  have hi' : i < a.wc.tensor.numel := by simpa [Tensor.numel,a.wc.shape] using hi
  change a.output.memory a.wc.tensor.allocation (a.wc.tensor.base+i) = weight.read entry i
  rw [a.output.frame _ _ (Or.inr (Or.inr rfl))]
  have copied := a.wc.values i hi
  rw [Tensor.read,a.wc.contiguous i hi'] at copied
  rw [copied]
  exact a.xc.frame _ _ existed

theorem Prepared.entry_frame (a : Prepared α input weight entry preexisting)
    (allocation offset : Nat) (existed : preexisting allocation) :
    a.output.memory allocation offset = entry allocation offset :=
  (a.output.frame _ _ (Or.inl existed)).trans
    ((a.wc.frame _ _ (Or.inl existed)).trans (a.xc.frame _ _ existed))

def Prepared.commit [Inhabited α] (a : Prepared α input weight entry preexisting)
    (final : State α) : Memory α := fun allocation offset =>
  if allocation = a.output.tensor.allocation ∧ a.output.tensor.base ≤ offset then
    (final.mem "Y" (offset-a.output.tensor.base)).read .real else a.output.memory allocation offset

def PhysicalOutside (a : Prepared α input weight entry preexisting) (allocation offset : Nat) : Prop :=
  ∀ i, i < input.numel → (allocation,offset) ≠ (a.output.tensor.allocation,a.output.tensor.base+i)

theorem Prepared.commit_frame [Inhabited α] (a : Prepared α input weight entry preexisting)
    (normalized : List Nat) (shape : input.numel = rows input normalized * columns normalized)
    (final : State α)
    (frame : ∀ region i, (region ≠ ("Y" : RegionName) ∨ input.numel ≤ i) →
      final.mem region i = (a.state normalized).mem region i)
    (allocation offset : Nat) (outside : PhysicalOutside a allocation offset) :
    a.commit final allocation offset = a.output.memory allocation offset := by
  unfold Prepared.commit
  split
  · rename_i h
    rcases h with ⟨same,base⟩
    subst allocation
    have bound : input.numel ≤ offset-a.output.tensor.base := by
      by_contra h
      have hi : offset-a.output.tensor.base < input.numel := Nat.lt_of_not_ge h
      apply outside (offset-a.output.tensor.base) hi
      congr 1
      omega
    rw [frame "Y" (offset-a.output.tensor.base) (Or.inr bound)]
    change a.output.memory a.output.tensor.allocation
      (a.output.tensor.base+(offset-a.output.tensor.base)) = _
    congr 1
    omega
  · rfl

theorem Prepared.preexisting_frame [Inhabited α] (a : Prepared α input weight entry preexisting)
    (normalized : List Nat) (shape : input.numel = rows input normalized * columns normalized)
    (final : State α)
    (frame : ∀ region i, (region ≠ ("Y" : RegionName) ∨ input.numel ≤ i) →
      final.mem region i = (a.state normalized).mem region i)
    (allocation offset : Nat) (existed : preexisting allocation) :
    a.commit final allocation offset = entry allocation offset := by
  have outside : PhysicalOutside a allocation offset := by
    intro i hi equal
    have same : allocation = a.output.tensor.allocation := congrArg Prod.fst equal
    apply a.output.fresh
    exact Or.inl (same ▸ existed)
  exact (a.commit_frame normalized shape final frame allocation offset outside).trans
    (a.entry_frame allocation offset existed)

structure Schedule (input : Tensor) (normalized : List Nat) where
  order : List Nat
  complete : order.Perm (List.range (rows input normalized))

def Prepared.entryRun [Inhabited α] (a : Prepared α input weight entry preexisting)
    (mdl : Algebra α) (normalized : List Nat) (eps : ℝ) (schedule : Schedule input normalized) :
    Option (State α) := runRows mdl (a.layout normalized) eps schedule.order (a.state normalized)

theorem Prepared.entry_result [Inhabited α] (a : Prepared α input weight entry preexisting)
    (mdl : Algebra α) (normalized : List Nat) (eps : ℝ)
    (domain : InputDomain input weight normalized preexisting) (schedule : Schedule input normalized) :
    ∃ final, a.entryRun mdl normalized eps schedule = some final ∧
      (∀ row : Fin (rows input normalized), ∀ i : Fin (columns normalized),
        a.output.tensor.read (a.commit final) (row.val*columns normalized+i.val) =
          rowValue mdl (a.layout normalized) eps
            (fun j => input.read entry (row.val*columns normalized+j)) (weight.read entry) i.val) ∧
      (∀ allocation offset, PhysicalOutside a allocation offset →
        a.commit final allocation offset = a.output.memory allocation offset) ∧
      (∀ allocation offset, preexisting allocation → a.commit final allocation offset = entry allocation offset) := by
  obtain ⟨final,ran,values,frame⟩ := grid_result mdl (a.layout normalized) eps
    (by change ("X" : RegionName) ≠ "Y"; decide)
    (by change ("W" : RegionName) ≠ "Y"; decide)
    schedule.order schedule.complete (a.state normalized)
  have shape := shape_product domain
  have frame' : ∀ region i, (region ≠ ("Y" : RegionName) ∨ input.numel ≤ i) →
      final.mem region i = (a.state normalized).mem region i := by
    simpa only [Prepared.layout,← shape] using frame
  refine ⟨final,ran,?_,a.commit_frame normalized shape final frame',
    a.preexisting_frame normalized shape final frame'⟩
  intro row i
  have bound (j : Nat) (hj : j < columns normalized) :
      row.val*columns normalized+j < input.numel := by
    rw [shape]
    calc
      _ < row.val*columns normalized+columns normalized := Nat.add_lt_add_left hj _
      _ = (row.val+1)*columns normalized := by ring
      _ ≤ rows input normalized*columns normalized := Nat.mul_le_mul_right _ row.isLt
  have outBound : row.val*columns normalized+i.val < a.output.tensor.numel := by
    simpa only [Tensor.numel,a.output.shape,a.xc.shape] using bound i.val i.isLt
  simp only [Tensor.read,a.output.contiguous _ outBound,Prepared.commit,
    le_add_iff_nonneg_right,Nat.zero_le,and_self,if_true,Nat.add_sub_cancel_left]
  erw [values row i,Cell.read_same]
  apply rowValue_congr _ _ _ _ _ _ _ _ i.isLt
  · intro j hj
    exact a.input_read normalized _ (bound j hj)
  · intro j hj
    exact a.weight_read normalized domain.weightExisted j
      (by simpa [Tensor.numel,domain.weightShape,columns] using hj)

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
