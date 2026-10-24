import bench.repairs.tritonbench_g.matmul_triton_autotune.Contract
import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.LegacyContext
import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.Memory
import bench.tritonbench_g.matmul_triton_autotune.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel
open VeriTile.Triton FP.Structural Whole
open VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
noncomputable section
set_option maxHeartbeats 3000000

/-- Both input regions read the same physical heap, so overlapping views agree
at equal byte addresses without an assumed alias-consistency callback. -/
def initial {α : Type} (a b : Matrix) (read : Nat → α) (scratch : State α) : State α :=
  { scratch with mem := fun region offset =>
      if region = "a" then Cell.mk .fp32 (read (inputBase a+4*offset))
      else if region = "b" then Cell.mk .fp32 (read (inputBase b+4*offset))
      else scratch.mem region offset }

theorem initial_a {α : Type} [Inhabited α] (a b : Matrix) (al : MatrixLaws a)
    (activation : String) (selection : Selection) (read : Nat → α) (scratch : State α)
    (i : Fin a.rows) (k : Fin a.columns) :
    Whole.inputA (request a b activation selection) (initial a b read scratch) i.val k.val =
      logical read a i.val k.val := by
  simp only [Whole.inputA, request, initial, ↓reduceIte, Cell.read_same]
  rw [logical, input_address a al i.val k.val i.isLt k.isLt]

theorem initial_b {α : Type} [Inhabited α] (a b : Matrix) (bl : MatrixLaws b)
    (dims : a.columns = b.rows) (activation : String) (selection : Selection)
    (read : Nat → α) (scratch : State α) (k : Fin a.columns) (j : Fin b.columns) :
    Whole.inputB (request a b activation selection) (initial a b read scratch) k.val j.val =
      logical read b k.val j.val := by
  simp only [Whole.inputB, request, initial,
    show ("b" : RegionName) ≠ "a" by decide, ↓reduceIte, Cell.read_same]
  rw [logical, input_address b bl k.val j.val (by simpa [←dims] using k.isLt) j.isLt]

theorem input_allocation (x : Matrix) (xl : TensorBinding x) : 4*inputExtent x < 2^63 := by
  have bound := xl.allocationBytes
  have elements := xl.backingElements
  have cap : inputExtent x ≤ x.tensor.storage.elements := by
    unfold inputExtent
    split <;> omega
  omega

theorem dense_input (x : Matrix) (xl : MatrixLaws x)
    (contiguous : x.tensor.metadata.isContiguous = Bool.true) :
    x.rows*x.columns ≤ inputExtent x := by
  by_cases hz : x.tensor.metadata.numel = 0
  · simp [inputExtent, hz, ←numel x xl, hz]
  have cap := VeriTile.Torch.FP32.contiguous_capacity x.tensor xl.tensor contiguous (by omega)
  rw [numel x xl] at cap
  simp only [inputExtent, hz, if_false]
  omega

theorem output_allocation (a b : Matrix) (c : OutputTensor) (cl : OutputBinding c a b) :
    2*(a.rows*b.columns) < 2^63 := by
  have hn := output_numel a b c cl.toOutputLaws
  by_cases hz : c.metadata.numel = 0
  · rw [←hn, hz]
    decide
  have bound := cl.inBounds (c.metadata.numel-1) (by omega)
  rw [cl.indices _ (by omega)] at bound
  have cap := cl.allocationBytes
  omega

theorem output_alignment (a b : Matrix) (c : OutputTensor) (cl : OutputLaws c a b) :
    outputBase c % 2 = 0 := by
  by_cases hz : c.metadata.numel = 0
  · simp [outputBase, hz]
  have hp := cl.alignment (by omega)
  rw [←output_pointer a b c cl (by omega), BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt (by have hs := output_span a b c cl; omega : outputBase c < 2^64)] at hp
  exact hp

/-- Derive the retained native safety interface from actual Tensor objects,
storage/stride laws, checked admission and ordinary fresh output storage. -/
def tensorAPI {α : Type} (a b : Matrix) (al : TensorBinding a) (bl : TensorBinding b)
    (h : InputContract a b) (activation : String) (selection : Selection)
    (c : OutputTensor) (cl : OutputBinding c a b) (fresh : Fresh a b c)
    (read : Nat → α) (scratch : State α) :
    TensorAPI (request a b activation selection) (initial a b read scratch) := by
  have ha := admitted a b al.toMatrixLaws bl.toMatrixLaws h
  have hn := output_numel a b c cl.toOutputLaws
  exact {
    extentA := inputExtent a, extentB := inputExtent b
    baseA := inputBase a, baseB := inputBase b, baseC := outputBase c
    metadataA := ⟨[a.rows,a.columns],[a.strideRow,a.strideColumn],.fp32,"cuda"⟩
    metadataB := ⟨[b.rows,b.columns],[b.strideRow,b.strideColumn],.fp32,"cuda"⟩
    metadataC := ⟨[a.rows,b.columns],[max b.columns 1,1],.fp16,"cuda"⟩
    shapeA := rfl, shapeB := rfl, shapeC := rfl
    stridesA := rfl, stridesB := rfl, stridesC := rfl
    dtypeA := rfl, dtypeB := rfl, dtypeC := rfl
    deviceB := rfl, deviceC := rfl
    dimensions := ha.1, contiguousA := ha.2.1
    selected := selection.selected
    freshA := by change ("c" : RegionName) ≠ "a"; decide
    freshB := by change ("c" : RegionName) ≠ "b"; decide
    physicalFreshA := by simpa only [hn] using fresh_view a al.toMatrixLaws _ _ fresh.a
    physicalFreshB := by simpa only [hn] using fresh_view b bl.toMatrixLaws _ _ fresh.b
    alignmentA := ha.2.2.2.2.1, alignmentB := ha.2.2.2.2.2
    alignmentC := output_alignment a b c cl.toOutputLaws
    spanA := input_span a al.toMatrixLaws, spanB := input_span b bl.toMatrixLaws
    spanC := by simpa only [hn] using output_span a b c cl.toOutputLaws
    allocationA := input_allocation a al, allocationB := input_allocation b bl
    allocationC := output_allocation a b c cl
    denseA := dense_input a al.toMatrixLaws ha.2.1
    strideAM := al.strideRow, strideAK := al.strideColumn
    strideBK := bl.strideRow, strideBN := bl.strideColumn
    sameViewOrigin := by
      intro hh
      exact False.elim ((by decide : ("a" : RegionName) ≠ "b") hh)
    aliasConsistency := by
      intro row hr k hk k' hk' col hc he
      simp only [request, initial, ↓reduceIte]
      exact congrArg (Cell.mk .fp32 ∘ read) he
    inputA := fun row hr k hk => input_capacity a al.toMatrixLaws row k hr hk
    inputB := by
      intro k hk col hc
      exact input_capacity b bl.toMatrixLaws k col (by simpa [request, ha.1] using hk) hc
    tagA := by
      intro row hr k hk
      exact ⟨read (inputBase a+4*(row*a.strideRow+k*a.strideColumn)), rfl⟩
    tagB := by
      intro k hk col hc
      exact ⟨read (inputBase b+4*(k*b.strideRow+col*b.strideColumn)), rfl⟩ }

theorem output_offset (a b : Matrix) (activation : String) (selection : Selection)
    (i j : Nat) (hj : j < b.columns) :
    Whole.outputOffset (request a b activation selection) i j = i*b.columns+j := by
  have hn : 1 ≤ b.columns := by omega
  simp [Whole.outputOffset, request, max_eq_left hn, Nat.mul_comm]

/-- Actual logical FP16 Tensor addresses coincide with every active quoted
source store; the physical interpretation is tied to that store's address. -/
theorem source_store_address {α : Type} (a b : Matrix) (activation : String) (selection : Selection)
    (c : OutputTensor) (cl : OutputLaws c a b) (s : State α)
    (i : FPValue.CIndex (request a b activation selection).parameters)
    (hi : FPValue.active (request a b activation selection).parameters s i) :
    let row := FPValue.row (request a b activation selection).parameters s i.1
    let col := FPValue.col (request a b activation selection).parameters s i.2.1
    outputAddress c (row*b.columns+col) = outputBase c+
      2*safeCOffset (request a b activation selection) row col := by
  dsimp only
  have hs : safeCOffset (request a b activation selection)
      (FPValue.row (request a b activation selection).parameters s i.1)
      (FPValue.col (request a b activation selection).parameters s i.2.1) =
      Whole.outputOffset (request a b activation selection)
        (FPValue.row (request a b activation selection).parameters s i.1)
        (FPValue.col (request a b activation selection).parameters s i.2.1) := by
    have hr : FPValue.row (request a b activation selection).parameters s i.1 <
        (request a b activation selection).rows := hi.1
    have hc : FPValue.col (request a b activation selection).parameters s i.2.1 <
        (request a b activation selection).columns := hi.2
    simp [safeCOffset, safeCoordinate, Whole.outputOffset, hr, hc]
  rw [hs, output_offset a b activation selection _ _ hi.2]
  apply output_address a b c cl
  rw [output_numel a b c cl]
  exact flat_bound _ _ _ _ hi.1 hi.2

theorem tensor_result {α β : Type} [Inhabited α] (a b : Matrix)
    (activation : String) (selection : Selection) (c : OutputTensor)
    (cl : OutputLaws c a b) (fresh : Fresh a b c)
    (encode : α → Fin 2 → β) (target : Fin a.rows → Fin b.columns → α)
    (before : Nat → β) (final : State α)
    (value : ∀ i : Fin a.rows, ∀ j : Fin b.columns,
      final.mem "c" (Whole.outputOffset (request a b activation selection) i.val j.val) =
        Cell.mk .fp16 (target i j)) : TensorResult a b c encode target before final := by
  refine ⟨?_, writeback_frame c encode before final,
    (backing_frame a b c fresh encode before final).1,
    (backing_frame a b c fresh encode before final).2⟩
  intro i j byte
  rw [writeback_at a b c cl encode before final _
    (by rw [output_numel a b c cl]; exact flat_bound _ _ _ _ i.isLt j.isLt) byte]
  have hv := value i j
  rw [output_offset a b activation selection _ _ j.isLt] at hv
  rw [hv, Cell.read_same]

#axiomsClean tensorAPI
#axiomsClean source_store_address
#axiomsClean tensor_result
#specNonCircular initial avoiding [Whole.reference, Whole.logicalReference, Whole.mathematical]
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel
