import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.Tensor
import bench.tritonbench_g.matmul_leakyrelu.Proofs.ContractedWhole

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section
set_option maxHeartbeats 3000000

theorem admitted (a b : Matrix) (al : MatrixLaws a) (bl : MatrixLaws b)
    (h : InputContract (metadata a b)) (activation : String) :
    Whole.IndexDomain (request a b activation) ∧ a.columns = b.rows ∧
    a.tensor.metadata.isContiguous = Bool.true ∧
    a.tensor.metadata.isNeg = Bool.false ∧ b.tensor.metadata.isNeg = Bool.false ∧
    a.tensor.metadata.device = b.tensor.metadata.device ∧
    inputBase a%4 = 0 ∧ inputBase b%4 = 0 ∧
    inputBase a+4*(a.rows*a.strideRow+(Whole.paddedInner (request a b activation)+32)*a.strideColumn)<2^64 ∧
    inputBase b+4*((Whole.paddedInner (request a b activation)+32)*b.strideRow+b.columns*b.strideColumn)<2^64 := by
  have ix := admitted_indices (metadata a b) h activation
  rw [request_eq a b al bl activation] at ix
  have hm := h.metadata
  simp only [metadataChecks, decide_eq_true_eq] at hm
  obtain ⟨_,_,_,_,_,_,_,_,dev,dims,cont,na,nb,aa,ab,ba,bb⟩ := hm
  have alignA : inputBase a%4 = 0 := by
    rcases aa with hz | ha
    · simp [inputBase, al.emptyPointer hz]
    · exact ha
  have alignB : inputBase b%4 = 0 := by
    rcases ab with hz | hb
    · simp [inputBase, bl.emptyPointer hz]
    · exact hb
  refine ⟨ix, ?_, cont, na, nb, dev, alignA, alignB, ?_, ?_⟩
  · simpa [metadata, al.shape, bl.shape] using dims
  · simpa [metadata, MatmulLeakyrelu.request, request, inputBase,
      Whole.paddedInner, al.shape, bl.shape, VeriTile.Torch.FP32.Tensor.dataPtr] using ba
  · simpa [metadata, MatmulLeakyrelu.request, request, inputBase,
      Whole.paddedInner, al.shape, bl.shape, VeriTile.Torch.FP32.Tensor.dataPtr] using bb

def initial {α : Type} (a b : Matrix) (read : Nat → α) (scratch : State α) : State α :=
  { scratch with mem := fun region offset =>
      if region = "a" then Cell.mk .fp32 (read (inputBase a+4*offset))
      else if region = "b" then Cell.mk .fp32 (read (inputBase b+4*offset))
      else scratch.mem region offset }

theorem initial_a {α : Type} [Inhabited α] (a b : Matrix) (al : MatrixLaws a)
    (activation : String) (read : Nat → α) (scratch : State α)
    (i : Fin a.rows) (k : Fin a.columns) :
    Whole.inputA (request a b activation) (initial a b read scratch) i.val k.val =
      logical read a i.val k.val := by
  simp only [Whole.inputA, request, initial, ↓reduceIte, Cell.read_same]
  rw [logical, input_address a al i.val k.val i.isLt k.isLt]

theorem initial_b {α : Type} [Inhabited α] (a b : Matrix) (bl : MatrixLaws b)
    (dims : a.columns = b.rows) (activation : String) (read : Nat → α) (scratch : State α)
    (k : Fin a.columns) (j : Fin b.columns) :
    Whole.inputB (request a b activation) (initial a b read scratch) k.val j.val =
      logical read b k.val j.val := by
  simp only [Whole.inputB, request, initial,
    show ("b" : RegionName) ≠ "a" by decide, ↓reduceIte, Cell.read_same]
  rw [logical, input_address b bl k.val j.val (by simpa [←dims] using k.isLt) j.isLt]

theorem input_view_in_backing (x : Matrix) (xl : MatrixLaws x)
    (hn : inputExtent x ≠ 0) :
    x.tensor.storage.base ≤ inputBase x ∧
    inputBase x+4*inputExtent x ≤ x.tensor.storage.base+x.storageBytes := by
  have hz : x.tensor.metadata.numel ≠ 0 := by
    intro hz
    simp [inputExtent, hz] at hn
  have hnum : 0 < x.rows*x.columns := by rw [←numel x xl]; omega
  have hr : 0 < x.rows := by nlinarith
  have hc : 0 < x.columns := by nlinarith
  have ha := input_address x xl 0 0 hr hc
  rw [xl.indices 0 hr 0 hc] at ha
  simp only [Nat.zero_mul, Nat.zero_add, Nat.add_zero,
    VeriTile.Torch.FP32.Storage.address] at ha
  have he : 4*x.tensor.storage.elements ≤ x.storageBytes := by
    rw [←xl.backingElements]
    omega
  have hcap := input_capacity x xl 0 0 hr hc
  simp only [inputExtent, hz, if_false] at hcap ⊢
  omega

theorem fresh_view (x : Matrix) (xl : MatrixLaws x) (base size : Nat)
    (h : Whole.byteSpansDisjoint base size x.tensor.storage.base x.storageBytes) :
    Whole.byteSpansDisjoint base size (inputBase x) (4*inputExtent x) := by
  by_cases hz : inputExtent x = 0
  · simp [Whole.byteSpansDisjoint, hz]
  have hx := input_view_in_backing x xl hz
  rcases h with h | h | h | h
  · exact Or.inl h
  · unfold Whole.byteSpansDisjoint; omega
  · exact Or.inr (Or.inr (Or.inl (by omega)))
  · exact Or.inr (Or.inr (Or.inr (by omega)))

/-- The older address theorem's TensorContract is derived from actual Tensor
bindings and the exact guards. In particular aliasConsistency is proved from
one physical reader; it is not a caller-supplied output or alias premise. -/
def tensorContract {α : Type} (a b : Matrix) (al : MatrixLaws a) (bl : MatrixLaws b)
    (h : InputContract (metadata a b)) (activation : String)
    (c : OutputTensor) (cl : OutputLaws c a b) (fresh : Fresh a b c)
    (output : outputCheck (metadata a b) (outputBase c) = Bool.true)
    (read : Nat → α) (scratch : State α) :
    Whole.TensorContract (request a b activation) (initial a b read scratch) := by
  have ha := admitted a b al bl h activation
  have hn := output_numel a b c cl
  have oc : outputBase c+2*((Whole.tileRows (request a b activation)*32)*max b.columns 1+
      Whole.tileColumns (request a b activation)*32)<2^64 := by
    simpa [outputCheck, metadata, MatmulLeakyrelu.request, request,
      Whole.tileRows, Whole.tileColumns, al.shape, bl.shape] using output
  have align : outputBase c%2 = 0 := by
    by_cases hz : c.metadata.numel = 0
    · simp [outputBase, hz]
    have hp := cl.alignment (by omega)
    rw [←output_pointer a b c cl (by omega), BitVec.toNat_ofNat,
      Nat.mod_eq_of_lt (by have hs := output_span a b c cl; omega : outputBase c < 2^64)] at hp
    exact hp
  exact {
    extentA := inputExtent a, extentB := inputExtent b, extentC := c.metadata.numel
    baseA := inputBase a, baseB := inputBase b, baseC := outputBase c
    metadataA := ⟨[a.rows,a.columns],[a.strideRow,a.strideColumn],.fp32,"cuda"⟩
    metadataB := ⟨[b.rows,b.columns],[b.strideRow,b.strideColumn],.fp32,"cuda"⟩
    metadataC := ⟨[a.rows,b.columns],[max b.columns 1,1],.fp16,"cuda"⟩
    shapeA := rfl, shapeB := rfl, shapeC := rfl
    stridesA := rfl, stridesB := rfl, stridesC := rfl
    dtypeA := rfl, dtypeB := rfl, dtypeC := rfl
    deviceB := rfl, deviceC := rfl
    dimensions := ha.2.1, contiguousA := ha.2.2.1
    freshA := by change ("c" : RegionName) ≠ "a"; decide
    freshB := by change ("c" : RegionName) ≠ "b"; decide
    physicalFreshA := fresh_view a al _ _ fresh.a
    physicalFreshB := fresh_view b bl _ _ fresh.b
    alignmentA := ha.2.2.2.2.2.2.1, alignmentB := ha.2.2.2.2.2.2.2.1
    alignmentC := align
    spanA := input_span a al, spanB := input_span b bl, spanC := output_span a b c cl
    sameViewOrigin := by
      intro hh
      exact False.elim ((by decide : ("a" : RegionName) ≠ "b") hh)
    aliasConsistency := by
      intro row hr k hk k' hk' col hc he
      simp only [request, initial, ↓reduceIte]
      exact congrArg (Cell.mk .fp32 ∘ read) he
    inputA := fun row hr k hk => input_capacity a al row k hr hk
    inputB := by
      intro k hk col hc
      exact input_capacity b bl k col (by simpa [request, ha.2.1] using hk) hc
    output := by
      intro row hr col hc
      have hc' : 1 ≤ b.columns := by change col < b.columns at hc; omega
      simpa [Whole.outputOffset, request, max_eq_left hc', Nat.mul_comm, hn]
        using flat_bound a.rows b.columns row col hr hc
    tagA := by
      intro row hr k hk
      exact ⟨read (inputBase a+4*(row*a.strideRow+k*a.strideColumn)), rfl⟩
    tagB := by
      intro k hk col hc
      exact ⟨read (inputBase b+4*(k*b.strideRow+col*b.strideColumn)), rfl⟩
    byteA := ha.2.2.2.2.2.2.2.2.1
    byteB := ha.2.2.2.2.2.2.2.2.2
    byteC := oc }

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
