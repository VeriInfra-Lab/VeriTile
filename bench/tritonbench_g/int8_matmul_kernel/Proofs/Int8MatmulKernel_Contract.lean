import bench.tritonbench_g.int8_matmul_kernel.Proofs.Contract

namespace VeriTile.Bench.Repairs.Int8MatmulKernel
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001

theorem indexChecks_iff (r : Request) (out : Nat) :
    indexChecks r = true ↔ SourceBounds r.shape (hostBuffers r out) := by
  simp only [indexChecks, decide_eq_true_eq]
  constructor
  · rintro ⟨hs, ht, hg, ha, hb, hc⟩; exact ⟨hs, ht, hg, ha, hb, hc⟩
  · intro h; exact ⟨h.shape, h.strides, h.grid, h.aExpr, h.bExpr, h.outExpr⟩

theorem castCheck_iff (word : Word) : castCheck word = true ↔ DefinedCast word := by
  unfold castCheck DefinedCast
  cases h : truncateFloat word <;> simp [h]

theorem inputChecks_iff (r : Request) (out : Nat) (before : Memory Word) :
    inputChecks r before = true ↔ InputDefined r.shape (hostBuffers r out) before := by
  simp only [inputChecks, List.all_eq_true, List.mem_range, castCheck_iff,
    InputDefined, Request.shape, hostBuffers, aAddress, Buffers.aRegion]
  constructor
  · intro h row k hr hk; exact h row hr k hk
  · intro h row hr k hk; exact h row k hr hk

theorem checkInputs_iff (r : Request) (aPointer bPointer : Nat) (before : Memory Word) :
    checkInputs r aPointer bPointer before = .ok () ↔ InputContract r aPointer bPointer before := by
  have hc : InputContract r aPointer bPointer before ↔ MetadataContract r aPointer bPointer ∧
      SourceBounds r.shape (hostBuffers r 0) ∧ InputDefined r.shape (hostBuffers r 0) before := by
    constructor
    · intro h; exact ⟨h.metadata, h.indices, h.defined⟩
    · rintro ⟨hm, hi, hd⟩; exact ⟨hm, hi, hd⟩
  have hm : metadataChecks r aPointer bPointer = true ↔ MetadataContract r aPointer bPointer := by
    simp only [metadataChecks, MetadataContract, decide_eq_true_eq]
  rw [hc, ← indexChecks_iff, ← inputChecks_iff, ← hm]
  simp [checkInputs, ite_eq_iff]

theorem checkArguments_iff (args : Arguments) (before : Memory Word) :
    checkArguments args before = .ok () ↔ Admitted args before := by
  cases args <;> simp [checkArguments, Admitted, checkInputs_iff]

/-- Construct the established whole-task domain. Coverage and physical pointer
metadata are ordinary Tensor contracts; legal completed trials reflect the
unchanged source static_assert. No result value is a premise. -/
def whole_domain (r : Request) (aPointer bPointer : Nat) (before : Memory Word) (h : InputContract r aPointer bPointer before)
    (out : Nat) (trials : List (Fin 15)) (selected : Fin 15)
    (coveredA : r.a.covered) (coveredB : r.b.covered)
    (pointers : PointerBounds r.shape (hostBuffers r out))
    (selectedLegal : r.shape.legal selected) (trialsLegal : ∀ c ∈ trials, r.shape.legal c) :
    WholeDomain r (hostBuffers r out) before trials selected := by
  rcases h.metadata with ⟨ha,hb,hac,hbc,hdev,hcurrent,hshape,hcontig,har,hbr,halign,hpa,hpb⟩
  exact ⟨⟨hshape,hcontig,ha,hb,hac,hbc,hdev,hcurrent,har,hbr,coveredA,coveredB⟩,
    (indexChecks_iff r out).1 ((indexChecks_iff r 0).2 h.indices), pointers,
    (inputChecks_iff r out before).1 ((inputChecks_iff r 0 before).2 h.defined),
    selectedLegal, trialsLegal⟩

/-- The three source-dependent padded/final byte bounds are established by
input checks and the explicit allocated-output check, not Tensor provenance. -/
def OrdinaryPointers.checked {r : Request} {out aPointer bPointer : Nat}
    (ordinary : OrdinaryPointers r out aPointer bPointer)
    (before : Memory Word) (admitted : InputContract r aPointer bPointer before)
    (cPointer : Nat) (output : outputChecks r cPointer=true) :
    PointerBounds r.shape (hostBuffers r out) := by
  have h := admitted.metadata
  rcases h with ⟨_,_,_,_,_,_,_,_,_,_,_,ha,hb⟩
  refine ⟨ordinary.aBase,ordinary.bBase,cPointer,?_,?_,?_⟩
  · simpa [ordinary.aData,hostBuffers,Nat.mul_add,Nat.add_assoc] using ha
  · simpa [ordinary.bData,hostBuffers,Nat.add_assoc] using hb
  · simpa only [outputChecks,decide_eq_true_eq] using output

/-- Guarded actual FP32 data pointers make every active scalar load naturally aligned.
No wider alignment promise is made for byte-addressed B. -/
theorem active_a_aligned (r : Request) (aPointer bPointer : Nat) (before : Memory Word)
    (h : InputContract r aPointer bPointer before) (row k : Nat)
    (hr : row < r.a.rows) (hk : k < r.a.cols) :
    (aPointer+4*(row*r.a.strideRow+k*r.a.strideCol))%4=0 := by
  have positive := Nat.mul_pos (Nat.zero_lt_of_lt hr) (Nat.zero_lt_of_lt hk)
  have aligned : aPointer%4=0 := h.metadata.2.2.2.2.2.2.2.2.2.2.1.resolve_left (by omega)
  simp [Nat.add_mod,aligned]

#axiomsClean checkArguments_iff
#axiomsClean whole_domain
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.Int8MatmulKernel
