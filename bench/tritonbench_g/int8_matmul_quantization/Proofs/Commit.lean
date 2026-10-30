import bench.tritonbench_g.int8_matmul_quantization.Proofs.GemmValues
import bench.tritonbench_g.rowwise_quantization_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Commit
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
variable {F ι : Type}

/-- One source store or atomic transition, retaining the current cell. -/
def step (region : RegionName) (offset : ι → Nat) (action : ι → FP.Structural.Cell F → FP.Structural.Cell F)
    (enabled : ι → Prop) [DecidablePred enabled] (s : State F) (i : ι) : State F :=
  if enabled i then s.write region (offset i) (action i (s.mem region (offset i))) else s

theorem frame (region : RegionName) (offset : ι → Nat) (action : ι → FP.Structural.Cell F → FP.Structural.Cell F)
    (enabled : ι → Prop) [DecidablePred enabled] (xs : List ι) (s : State F)
    (r : RegionName) (o : Nat)
    (outside : r ≠ region ∨ ∀ i ∈ xs, enabled i → o ≠ offset i) :
    (xs.foldl (step region offset action enabled) s).mem r o = s.mem r o := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih =>
    rw [List.foldl_cons, ih]
    · unfold step
      split_ifs with hi
      · apply State.write_other
        rcases outside with hr | ho
        · exact Or.inl hr
        · exact Or.inr (ho i (by simp) hi)
      · rfl
    · rcases outside with hr | ho
      · exact Or.inl hr
      · exact Or.inr (fun j hj => ho j (by simp [hj]))

/-- One issued event per active address. There is no associativity,
commutation, initial-zero or rollback assumption. -/
theorem readback (region : RegionName) (offset : ι → Nat) (action : ι → FP.Structural.Cell F → FP.Structural.Cell F)
    (enabled : ι → Prop) [DecidablePred enabled] (xs : List ι) (s : State F) (i : ι)
    (unique : xs.Nodup) (present : i ∈ xs) (active : enabled i)
    (injective : ∀ i j, enabled i → enabled j → offset i = offset j → i = j) :
    (xs.foldl (step region offset action enabled) s).mem region (offset i) =
      action i (s.mem region (offset i)) := by
  obtain ⟨before, after, hx⟩ := List.append_of_mem present
  rw [hx, List.nodup_append, List.nodup_cons] at unique
  obtain ⟨_, ⟨absent, _⟩, disjoint⟩ := unique
  have missBefore : ∀ j ∈ before, enabled j → offset i ≠ offset j := by
    intro j hj ha same
    have he := injective i j active ha same
    exact (disjoint j hj i (by simp)) he.symm
  have missAfter : ∀ j ∈ after, enabled j → offset i ≠ offset j := by
    intro j hj ha same
    have he := injective i j active ha same
    exact absent (he.symm ▸ hj)
  rw [hx, List.foldl_append, List.foldl_cons,
    frame region offset action enabled after _ region (offset i) (Or.inr missAfter)]
  simp only [step, if_pos active, State.write_same]
  rw [frame region offset action enabled before s region (offset i) (Or.inr missBefore)]

#axiomsClean frame
#axiomsClean readback
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Commit

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Gemm
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 6000000
variable {I F H : Type} [Inhabited F]

def source (rowScale output : Precision) (p : Parameters) : ComputeKernel :=
  match rowScale, output with
  | .half, .half => Source.gemmHalfHalf p
  | .half, .single => Source.gemmHalfSingle p
  | .single, .half => Source.gemmSingleHalf p
  | .single, .single => Source.gemmSingleSingle p

theorem source_identity (rowScale output : Precision) (p : Parameters) :
    source rowScale output p = KernelGemm.source rowScale output p := by
  cases rowScale <;> cases output <;> rfl

def action (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (atomicHalf atomicSingle : F → F → F) (rowScale output : Precision)
    (p : Parameters) (input : Input F) (s : State F) (i : KernelGemm.CIndex p) (old : FP.Structural.Cell F) : FP.Structural.Cell F :=
  let value := KernelGemm.storedValue M output (partialWord M carrier ops plans rowScale p input
    (KernelGemm.rowBlock p s*p.blockRows+i.1.val)
    (KernelGemm.columnBlock p s*p.blockColumns+i.2.1.val) (s.pids 1))
  match output with
  | .half => .mk .fp16 (if p.split = 1 then value else atomicHalf (old.read .fp16) value)
  | .single => .mk .real (if p.split = 1 then value else atomicSingle (old.read .real) value)

def writtenTarget (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (atomicHalf atomicSingle : F → F → F) (rowScale output : Precision)
    (p : Parameters) (input : Input F) (s : State F) : State F :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (Commit.step p.output (KernelGemm.outputOffset p s)
      (action M carrier ops plans atomicHalf atomicSingle rowScale output p input s)
      (KernelGemm.active p s)) s

theorem written_target (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (atomicHalf atomicSingle : F → F → F) (rowScale output : Precision)
    (p : Parameters) (input : Input F) (s : State F) (reads : Reads p rowScale input s) :
    KernelGemm.written M carrier ops plans atomicHalf atomicSingle rowScale output p s =
      writtenTarget M carrier ops plans atomicHalf atomicSingle rowScale output p input s := by
  unfold KernelGemm.written writtenTarget
  congr 1
  funext current i
  unfold Commit.step
  by_cases active : KernelGemm.active p s i
  · simp only [if_pos active, partial_source M carrier ops plans p rowScale input s reads i active]
    cases output <;> rfl
  · simp [active]

theorem output_injective (p : Parameters) (s : State F)
    (rowStride : p.outputRowStride = p.columns) (columnStride : p.outputColumnStride = 1)
    (i j : KernelGemm.CIndex p) (hi : KernelGemm.active p s i) (hj : KernelGemm.active p s j)
    (same : KernelGemm.outputOffset p s i = KernelGemm.outputOffset p s j) : i = j := by
  have address := same
  simp only [KernelGemm.outputOffset, rowStride, columnStride, one_mul] at address
  rw [Nat.mul_comm p.columns, Nat.mul_comm p.columns] at address
  obtain ⟨hr,hc⟩ := RowwiseQuantizationTriton.linear_injective p.columns _ _ _ _ hi.2 hj.2 address
  apply Prod.ext
  · exact Fin.ext (by omega)
  · exact Prod.ext (Fin.ext (by omega)) (Subsingleton.elim _ _)

/-- Full readable source executes; every active lane performs its independent
ordered transition on the actual old output cell. All other cells frame. -/
theorem program_complete (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (atomicHalf atomicSingle : F → F → F)
    (half : M.atomicAdd .fp16 = some atomicHalf) (single : M.atomicAdd .real = some atomicSingle)
    (rowScale output : Precision) (p : Parameters) (input : Input F) (s : State F)
    (reads : Reads p rowScale input s)
    (rowStride : p.outputRowStride = p.columns) (columnStride : p.outputColumnStride = 1) :
    ∃ final, KernelExecution.exec M carrier ops plans (source rowScale output p) s = some final ∧
      (∀ i : KernelGemm.CIndex p, KernelGemm.active p s i →
        final.mem p.output (KernelGemm.outputOffset p s i) =
          action M carrier ops plans atomicHalf atomicSingle rowScale output p input s i
            (s.mem p.output (KernelGemm.outputOffset p s i))) ∧
      (∀ (region : RegionName) offset,
        (region ≠ p.output ∨ ∀ i : KernelGemm.CIndex p, KernelGemm.active p s i → offset ≠ KernelGemm.outputOffset p s i) →
        final.mem region offset = s.mem region offset) := by
  obtain ⟨final, he, hm⟩ := KernelGemm.source_execution M carrier ops plans atomicHalf atomicSingle
    half single rowScale output p s
  rw [written_target M carrier ops plans atomicHalf atomicSingle rowScale output p input s reads] at hm
  refine ⟨final, by simpa [source_identity] using he, ?_, ?_⟩
  · intro i hi
    rw [hm]
    exact Commit.readback p.output (KernelGemm.outputOffset p s)
      (action M carrier ops plans atomicHalf atomicSingle rowScale output p input s)
      (KernelGemm.active p s) _ s i (TileShape.allIndices_nodup _) (TileShape.mem_allIndices _ i) hi
      (output_injective p s rowStride columnStride)
  · intro region offset outside
    rw [hm]
    apply Commit.frame
    rcases outside with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ => ho i)

#axiomsClean source_identity
#axiomsClean written_target
#axiomsClean output_injective
#axiomsClean program_complete
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Gemm
