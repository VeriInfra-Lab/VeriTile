import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.GemmIdeal

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe
open VeriTile.Triton KernelExecution
open Int8DequantMatmul (Micro)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

theorem native_add_exact (x y : BitVec 32) (h : Fits32 (x.toInt+y.toInt)) :
    (x+y).toInt=x.toInt+y.toInt := by
  calc
    (x+y).toInt = (BitVec.ofInt 32 (x.toInt+y.toInt)).toInt := by
      rw [BitVec.ofInt_add, BitVec.ofInt_toInt, BitVec.ofInt_toInt]
    _ = _ := encode_exact _ h

private theorem option_sum (xs : List (Option α)) (f : α → Int) :
    ((xs.filterMap id).map f).sum = (xs.map (fun x => (x.map f).getD 0)).sum := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    cases x <;> simpa only [List.filterMap_cons, id_eq, Option.map_none, Option.map_some,
      Option.getD_none, Option.getD_some, List.map_cons, List.sum_cons, zero_add,
      add_right_inj] using ih

theorem instruction_total_terms (kind : ISA) (lanes : Fin 32 → Option (Fin BK))
    (xs ys : Fin BK → Byte) (old : BitVec 32) :
    KernelNativeModel.instructionTotal (fun i => ((lanes i).map xs).getD 0)
      (fun i => ((lanes i).map ys).getD 0) old =
      old.toInt + ((Micro.instruction kind lanes).terms.map (fun k => (xs k).toInt*(ys k).toInt)).sum := by
  simp only [KernelNativeModel.instructionTotal, Micro.terms, option_sum, List.map_ofFn]
  rw [← Fin.sum_ofFn]
  congr 2
  congr 1
  funext i
  cases h : lanes i <;> simp [h]

/-- Local integer exactness for the actual scalar-add or K32 instruction.
The prefix bound is discharged structurally by operations_exact below. -/
theorem micro_exact (f : KernelNativeModel.FloatingInstructions) (op : Micro BK)
    (xs ys : Fin BK → Byte) (old : BitVec 32)
    (bound : Fits32 (old.toInt+(op.terms.map (fun k => (xs k).toInt*(ys k).toInt)).sum)) :
    (op.eval (KernelNativeModel.operations f).toOps xs ys old).toInt =
      old.toInt+(op.terms.map (fun k => (xs k).toInt*(ys k).toInt)).sum := by
  cases op with
  | scalar k =>
    have product : Fits32 ((xs k).toInt*(ys k).toInt) := by
      have h := product_bounds (xs k) (ys k)
      norm_num [Fits32]; omega
    simp only [Micro.eval, Micro.terms, List.map_cons, List.map_nil, List.sum_cons,
      List.sum_nil, add_zero] at bound ⊢
    change (old+BitVec.ofInt 32 ((xs k).toInt*(ys k).toInt)).toInt = _
    rw [native_add_exact _ _ (by simpa only [encode_exact _ product] using bound), encode_exact _ product]
  | instruction kind lanes =>
    change (KernelNativeModel.instructions.result kind _ _ old).toInt = _
    rw [instruction_exact _ _ _ _ (by simpa only [instruction_total_terms kind] using bound),
      instruction_total_terms]

theorem operations_exact (f : KernelNativeModel.FloatingInstructions) (K : Nat) (hk : K ≤ 131071)
    (a b : Nat → Byte) (index : Fin BK → Nat) (xs ys : Fin BK → Byte)
    (products : ∀ k, (xs k).toInt*(ys k).toInt = maskedProduct K a b (index k))
    (instructions : List (Micro BK)) (used : List Nat) (old : BitVec 32)
    (distinct : (used ++ (instructions.flatMap Micro.terms).map index).Nodup)
    (seed : old.toInt = (used.map (maskedProduct K a b)).sum) :
    (instructions.foldl (fun old op => op.eval (KernelNativeModel.operations f).toOps xs ys old) old).toInt =
      ((used ++ (instructions.flatMap Micro.terms).map index).map (maskedProduct K a b)).sum := by
  induction instructions generalizing used old with
  | nil => simpa using seed
  | cons op instructions ih =>
    have distinct' : ((used ++ op.terms.map index) ++ (instructions.flatMap Micro.terms).map index).Nodup := by
      simpa only [List.flatMap_cons, List.map_append, List.append_assoc] using distinct
    have bound := arbitrary_prefix_fits K hk a b (used ++ op.terms.map index)
      (List.nodup_append.mp distinct').1
    have terms : (op.terms.map (fun k => (xs k).toInt*(ys k).toInt)).sum =
        ((op.terms.map index).map (maskedProduct K a b)).sum := by
      simp only [List.map_map, Function.comp_def]
      simp_rw [products]
    have next : (op.eval (KernelNativeModel.operations f).toOps xs ys old).toInt =
        ((used ++ op.terms.map index).map (maskedProduct K a b)).sum := by
      rw [micro_exact _ _ _ _ _ (by simpa only [seed, terms, List.map_append, List.sum_append] using bound)]
      simp only [seed, terms, List.map_append, List.sum_append]
    simpa only [List.foldl_cons, List.flatMap_cons, List.map_append, List.append_assoc] using
      ih (used ++ op.terms.map index) _ distinct' next

/-- Both fused carry and independent block-then-add paths are exact; the
latter gets its own zero-seed instruction bounds before the final add. -/
theorem plan_exact (f : KernelNativeModel.FloatingInstructions) (K : Nat) (hk : K ≤ 131071)
    (a b : Nat → Byte) (index : Fin BK → Nat) (xs ys : Fin BK → Byte)
    (products : ∀ k, (xs k).toInt*(ys k).toInt = maskedProduct K a b (index k))
    (plan : DotPlan BK) (used : List Nat) (old : BitVec 32)
    (distinct : (used ++ (plan.operations.flatMap Micro.terms).map index).Nodup)
    (seed : old.toInt = (used.map (maskedProduct K a b)).sum) :
    (plan.eval (KernelNativeModel.operations f).toOps xs ys old).toInt =
      ((used ++ (plan.operations.flatMap Micro.terms).map index).map (maskedProduct K a b)).sum := by
  unfold Int8DequantMatmul.DotPlan.eval
  split_ifs
  · exact operations_exact f K hk a b index xs ys products plan.operations used old distinct seed
  · have block := operations_exact f K hk a b index xs ys products plan.operations [] 0
      (by simpa using (List.nodup_append.mp distinct).2.1) (by rfl)
    have bound := arbitrary_prefix_fits K hk a b _ distinct
    change (old + plan.operations.foldl _ 0).toInt = _
    rw [native_add_exact _ _ (by simpa only [seed, block, List.nil_append, List.map_append, List.sum_append] using bound)]
    simp only [seed, block, List.nil_append, List.map_append, List.sum_append]

theorem planCoordinates_succ (BK t : Nat) (plans : Nat → DotPlan BK) :
    planCoordinates BK (t+1) plans = planCoordinates BK t plans ++
      ((plans t).operations.flatMap Micro.terms).map (fun lane => (t,lane)) := by
  simp [planCoordinates, List.range_succ, List.flatMap_append]

/-- The concrete signed32 register after every source loop prefix equals
its exact masked integer sum, under the sole active-K guard. -/
theorem accumulator_native_exact (f : KernelNativeModel.FloatingInstructions)
    (s : Shape) (hk : s.K ≤ 131071) (c : Config) (columnMajor : Bool) (plans : Plans)
    (input : Gemm.Input Word32) (r n : Nat) (z : Fin c.S) (t : Nat) :
    Gemm.accumulator signed32 (KernelNativeModel.operations f).toOps plans (Gemm.parameters s c columnMajor) input r n z t =
      (((planCoordinates c.BK t (plans c.BK r n z)).map (fun q => sourceK c q.1 z q.2)).map
        (maskedProduct s.K (input.A r) (fun k => input.B k n))).sum := by
  induction t with
  | zero => rfl
  | succ t ih =>
    have distinct := (planCoordinates_distinct c.BK (t+1) (plans c.BK r n z)).map (source_index_injective c z)
    rw [planCoordinates_succ, List.map_append] at distinct ⊢
    simp only [List.map_map, Function.comp_def] at distinct ⊢
    have usedBound := source_prefix_fits s hk c z (input.A r) (fun k => input.B k n)
      (planCoordinates c.BK t (plans c.BK r n z)) (planCoordinates_distinct _ _ _)
    change ((plans c.BK r n z t).eval (KernelNativeModel.operations f).toOps
      (fun l => if sourceK c t z l < s.K then input.A r (sourceK c t z l) else 0)
      (fun l => if sourceK c t z l < s.K then input.B (sourceK c t z l) n else 0)
      (BitVec.ofInt 32 (Gemm.accumulator signed32 (KernelNativeModel.operations f).toOps plans
        (Gemm.parameters s c columnMajor) input r n z t))).toInt = _
    apply plan_exact f s.K hk (input.A r) (fun k => input.B k n) (fun l => sourceK c t z l)
    · intro l
      by_cases h : sourceK c t z l < s.K <;> simp [maskedProduct, h]
    · exact distinct
    · rw [ih, encode_exact _ usedBound]

#axiomsClean micro_exact
#axiomsClean operations_exact
#axiomsClean plan_exact
#axiomsClean accumulator_native_exact
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe
