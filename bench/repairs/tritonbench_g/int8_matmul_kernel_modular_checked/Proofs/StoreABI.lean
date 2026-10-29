import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.TensorStorage

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TensorModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
open TileModel
noncomputable section

def backingStore (memory : Memory Word32) (out offset : Nat) (word : Word32) : Memory Word32 :=
  fun loc => if loc = ((out,.int32),offset) then word else memory loc
def sourceStore (memory : Backing) (offset : Nat) (word : Word32) : Backing :=
  fun loc => if loc = (outputPort,offset) then int32Cell word else memory loc

/-- One source INT32 scalar store and one backing-memory scalar store commute
through the explicit output ABI. This is a primitive memory operation proof,
not an assumed whole-kernel observation equation. -/
theorem store_commutes (out : Nat) (before : Memory Word32) (sourceMemory : Backing)
    (offset : Nat) (word : Word32) :
    decodeBacking out before (sourceStore sourceMemory offset word) =
      backingStore (decodeBacking out before sourceMemory) out offset word := by
  funext loc
  by_cases hr : loc.1 = (out,.int32)
  · by_cases ho : loc.2 = offset
    · have hl : loc = ((out,.int32),offset) := Prod.ext hr ho
      simp [decodeBacking,sourceStore,backingStore,hl,int32_read]
    · have hl : loc ≠ ((out,.int32),offset) := by intro he; exact ho (congrArg Prod.snd he)
      simp [decodeBacking,sourceStore,backingStore,hr,ho,hl]
  · have hl : loc ≠ ((out,.int32),offset) := by intro he; exact hr (congrArg Prod.fst he)
    simp [decodeBacking,backingStore,hr,hl]

theorem state_store (state : State Word32) (offset : Nat) (word : Word32) :
    flatten (state.write outputPort offset (int32Cell word)) = sourceStore (flatten state) offset word := by
  funext loc
  simp [flatten,State.write,sourceStore,Prod.ext_iff,and_comm]

theorem encode_store (request : Request) (out : Nat) (before : Memory Word32)
    (separate : out ≠ request.a.storage ∧ out ≠ request.b.storage) (offset : Nat) (word : Word32) :
    flatten (encodedState request out (backingStore before out offset word)) =
      sourceStore (flatten (encodedState request out before)) offset word := by
  funext loc
  rcases loc with ⟨r,o⟩
  by_cases ha : r = inputPort
  · subst r
    have ne : request.a.storage ≠ out := Ne.symm separate.1
    simp [flatten,encodedState,sourceStore,backingStore,inputPort,outputPort,ne]
  · by_cases hb : r = weightPort
    · subst r
      have ne : request.b.storage ≠ out := Ne.symm separate.2
      simp [flatten,encodedState,sourceStore,backingStore,inputPort,weightPort,outputPort,ne]
    · by_cases hc : r = outputPort
      · subst r
        by_cases ho : o = offset <;>
          simp [flatten,encodedState,sourceStore,backingStore,inputPort,weightPort,outputPort,ho]
      · simp [flatten,encodedState,sourceStore,backingStore,ha,hb,hc]

theorem output_word_cell (request : Request) (out : Nat) (before : Memory Word32)
    (c : Fin 15) (offset : Nat) :
    ∃ word, gridMemory (layout request) request.shape c (encodedState request out before)
      (outputPort,offset) = int32Cell word := by
  classical
  unfold gridMemory frozenStep
  rw [fixed_cell]
  split
  · exact ⟨_,rfl⟩
  · exact ⟨before ((out,.int32),offset),output_encoding request out before offset⟩

/-- The final evaluator memory is the encoding of the final common backing,
with the same input storage identities and unchanged cells outside C. Fresh
output storage is an ordinary allocation fact and is used explicitly here. -/
theorem encode_result (request : Request) (out : Nat) (before : Memory Word32)
    (separate : out ≠ request.a.storage ∧ out ≠ request.b.storage) (c : Fin 15) :
    flatten (encodedState request out (resultMemory request out before c)) =
      gridMemory (layout request) request.shape c (encodedState request out before) := by
  funext loc
  rcases loc with ⟨r,o⟩
  have outside (ne : r ≠ outputPort) :
      gridMemory (layout request) request.shape c (encodedState request out before) (r,o) =
        flatten (encodedState request out before) (r,o) :=
    grid_frame _ _ _ _ (by exact ⟨rfl,rfl⟩) _ (Or.inl ne)
  by_cases ha : r = inputPort
  · subst r
    rw [outside (by decide)]
    have ne : request.a.storage ≠ out := Ne.symm separate.1
    simp [flatten,encodedState,resultMemory,decodeBacking,inputPort,outputPort,ne]
  · by_cases hb : r = weightPort
    · subst r
      rw [outside (by decide)]
      have ne : request.b.storage ≠ out := Ne.symm separate.2
      simp [flatten,encodedState,resultMemory,decodeBacking,inputPort,weightPort,outputPort,ne]
    · by_cases hc : r = outputPort
      · subst r
        obtain ⟨word,hw⟩ := output_word_cell request out before c o
        change (encodedState request out (resultMemory request out before c)).mem outputPort o = _
        rw [output_encoding]
        simp [resultMemory,decodeBacking,hw,int32_read,BitVec.ofInt_toInt]
      · rw [outside hc]
        simp [flatten,encodedState,ha,hb,hc]

#axiomsClean store_commutes
#axiomsClean state_store
#axiomsClean encode_store
#axiomsClean output_word_cell
#axiomsClean encode_result
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TensorModel
