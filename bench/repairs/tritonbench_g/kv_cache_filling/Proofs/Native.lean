import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Execution
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Contract
import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.Bounds

namespace VeriTile.Bench.Repairs.KvCacheFilling.Native
open TensorInputs

abbrev domain {t : TensorInputs} {r : Request} (h : InputContract t r) : Domain r := h.2.2.2.2.2.2

theorem state_extents (t : TensorInputs) (r : Request) (h : InputContract t r) :
    r.tokens*r.heads*r.dk*4<2^63 ∧ r.tokens*r.heads*r.dv*4<2^63 := by
  have limits := h.2.2.2.2.2.1
  have key := limits.2.2.2 t.key (by simp [dataRoles])
  have value := limits.2.2.2 t.value (by simp [dataRoles])
  rcases h.1 with ⟨ks,vs,_⟩
  rcases h.2.1 with ⟨kt,vt,_⟩
  constructor
  · simpa [VeriTile.Torch.TensorMetadata.numel,ks,bytes,kt,Nat.mul_assoc] using key
  · simpa [VeriTile.Torch.TensorMetadata.numel,vs,bytes,vt,Nat.mul_assoc] using value

theorem cache_extents (t : TensorInputs) (r : Request) (h : InputContract t r) :
    r.pages*r.block*r.heads*r.ck*elementBytes r .key<2^63 ∧
    r.pages*r.block*r.heads*r.cv*elementBytes r .value<2^63 := by
  have limits := h.2.2.2.2.2.1
  have key := limits.2.2.2 t.keyCache (by simp [dataRoles])
  have value := limits.2.2.2 t.valueCache (by simp [dataRoles])
  have ks := h.1.2.2.1
  have vs := h.1.2.2.2.1
  have kt := h.2.1.2.2.1
  have vt := h.2.1.2.2.2.1
  constructor
  · by_cases hp : r.policy=.copy <;>
      simpa [VeriTile.Torch.TensorMetadata.numel,ks,bytes,kt,hp,elementBytes,Nat.mul_assoc] using key
  · by_cases hp : r.policy=.copy <;>
      simpa [VeriTile.Torch.TensorMetadata.numel,vs,bytes,vt,hp,elementBytes,Nat.mul_assoc] using value

def signed64 (i : Int) : Int := (BitVec.ofInt 64 i).toInt

theorem signed64_exact (i : Int) (lo : -(2^63) ≤ i) (hi : i < 2^63) : signed64 i=i :=
  BitVec.toInt_ofInt_eq_self (by decide) lo hi

/-- Host-created INT64 source/destination rows fit before device arithmetic.
All active products are bounded by the corresponding checked allocation. -/
theorem source_indices (t : TensorInputs) (r : Request) (h : InputContract t r)
    (j : Job) (hj : j∈jobs r) :
    sourceRow r j<2^63 ∧ destinationRow r j<2^63 ∧
    (∀ i,i<r.dk → sourceRow r j*r.dk+i<2^61) ∧
    (∀ i,i<r.dv → sourceRow r j*r.dv+i<2^61) := by
  have d := domain h
  have hs := sourceRow_bound r d j hj
  have hd := destinationRow_bound r d j hj
  obtain ⟨key,value⟩ := state_extents t r h
  obtain ⟨cache,_⟩ := cache_extents t r h
  have hp := (packed_widths_positive r d).1
  have eb : 0<elementBytes r .key := by
    change 0 < (if r.policy=.copy then 4 else 1)
    split_ifs <;> decide
  have src := Nat.le_mul_of_pos_right (r.tokens*r.heads) d.dk_pos
  have dst := Nat.le_mul_of_pos_right (r.pages*r.block*r.heads) hp
  have bytes := Nat.le_mul_of_pos_right (r.pages*r.block*r.heads*r.ck) eb
  have hs64 : sourceRow r j < 2^63 := by nlinarith
  have hd64 : destinationRow r j < 2^63 := by nlinarith
  refine ⟨hs64,hd64,?_,?_⟩
  · intro i hi
    have bound := flat_bound (sourceRow r j) i (r.tokens*r.heads) r.dk hs hi
    omega
  · intro i hi
    have bound := flat_bound (sourceRow r j) i (r.tokens*r.heads) r.dv hs hi
    omega

/-- All arange values fit before the explicit int64 widening. -/
theorem padded_lanes (t : TensorInputs) (r : Request) (h : InputContract t r) :
    r.ck.nextPowerOfTwo≤2^20 ∧ r.cv.nextPowerOfTwo≤2^20 := by
  have limits := h.2.2.2.2.2.1
  exact ⟨VeriTile.Bench.Repairs.EmbeddingTritonKernel.Bounds.nextPower_bound _ limits.1,
    VeriTile.Bench.Repairs.EmbeddingTritonKernel.Bounds.nextPower_bound _ limits.2.1⟩

theorem pid_cast (t : TensorInputs) (r : Request) (h : InputContract t r)
    (i : Nat) (hi : i<(jobs r).length) :
    (BitVec.ofInt 32 (i : Int)).toInt=(i : Int) := by
  have grid := h.2.2.2.2.2.1.2.2.1
  exact BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)

theorem safe_lane (n i : Nat) (hn : 0<n) : padIndex n i<n := by
  unfold padIndex
  split_ifs <;> omega

/-- Both int4 halves, the ordinary policy lanes and masked padded pointers
stay inside the selected source row. The second-half addition is int64 exact. -/
theorem source_lanes (t : TensorInputs) (r : Request) (h : InputContract t r)
    (j : Job) (hj : j∈jobs r) (i : Nat) :
    sourceRow r j*r.dk+padIndex r.ck i<2^61 ∧
    sourceRow r j*r.dv+padIndex r.cv i<2^61 ∧
    (r.policy=.int4 → sourceRow r j*r.dk+r.ck+padIndex r.ck i<2^61 ∧
      sourceRow r j*r.dv+r.cv+padIndex r.cv i<2^61) := by
  have d := domain h
  have positive := packed_widths_positive r d
  have ki := safe_lane r.ck i positive.1
  have vi := safe_lane r.cv i positive.2
  have kc : r.ck ≤ r.dk := Nat.div_le_self _ _
  have vc : r.cv ≤ r.dv := Nat.div_le_self _ _
  have bounds := source_indices t r h j hj
  refine ⟨bounds.2.2.1 _ (by omega),bounds.2.2.2 _ (by omega),?_⟩
  intro policy
  have ks := d.key_shape
  have vs := d.value_shape
  simp only [Request.factor,policy,↓reduceIte] at ks vs
  constructor
  · have b := bounds.2.2.1 (r.ck+padIndex r.ck i) (by omega)
    omega
  · have b := bounds.2.2.2 (r.cv+padIndex r.cv i) (by omega)
    omega

/-- Native address-space facts supplied by ordinary actual Tensor storage.
They describe backing capacities only, never a kernel or arithmetic result. -/
structure StorageFits (r : Request) (l : Layout r) (inputs : InputLayout r l) : Prop where
  output : ∀ role,activeRole r role → l.base role+l.size role≤2^64
  input : ∀ v,inputs.base v+inputs.size v≤2^64

def pointer (p : Nat) : Nat := (BitVec.ofNat 64 p).toNat

theorem output_pointer (r : Request) (l : Layout r) (inputs : InputLayout r l)
    (fit : StorageFits r l inputs) (a : Address) (b : Nat) (h : ValidByte r a b) :
    pointer (bytePtr r l a b)=bytePtr r l a b := by
  have bound := bytePtr_bound r l a b h
  have span := fit.output a.role h.2.2.1
  exact Nat.mod_eq_of_lt (by omega)

theorem input_pointer (r : Request) (l : Layout r) (inputs : InputLayout r l)
    (fit : StorageFits r l inputs) (v : Bool) (row col b : Nat)
    (hr : row<r.tokens*r.heads) (hc : col<stateWidth r v) (hb : b<4) :
    pointer (inputPtr r inputs v row col b)=inputPtr r inputs v row col b := by
  have bound := inputPtr_bound r inputs v row col b hr hc hb
  have span := fit.input v
  exact Nat.mod_eq_of_lt (by omega)

#axiomsClean state_extents
#axiomsClean cache_extents
#axiomsClean source_indices
#axiomsClean padded_lanes
#axiomsClean pid_cast
#axiomsClean source_lanes
#axiomsClean output_pointer
#axiomsClean input_pointer
end VeriTile.Bench.Repairs.KvCacheFilling.Native
