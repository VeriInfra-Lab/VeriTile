import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.TileDot

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 6000000

def inputWords (layout : Layout) (state : State Word32) (row : Nat) : Nat → Word32 :=
  fun k => (state.mem layout.input (row*layout.inputRowStride+k*layout.inputInnerStride)).read .real
def packedValues (layout : Layout) (state : State Word32) (column : Nat) : Nat → Nat :=
  fun q => (state.mem layout.weight
    (q*layout.weightInnerStride+column*layout.weightColumnStride)).read .nat
def packedWords (layout : Layout) (state : State Word32) (column : Nat) : Nat → Word32 :=
  fun q => BitVec.ofNat 32 (packedValues layout state column q)

def FP32Domain (layout : Layout) (s : Shape) (state : State Word32) : Prop :=
  ∀ row, row < s.m → DefinedRow s.k (inputWords layout state row)
def UInt8Domain (layout : Layout) (s : Shape) (state : State Word32) : Prop :=
  ∀ column, column < s.n → ∀ q, q < s.k/4 → packedValues layout state column q < 256

theorem aWord_stream (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (state : State Word32) (f j : Nat) (hf : f < 4) (hj : j < s.steps c)
    (i : AIndex (parameters layout s c)) :
    aWord (parameters layout s c) state f j i =
      inputWords layout state (inputRow (parameters layout s c) state i.1)
        ((f*s.steps c+j)*(configs c).bk+i.2.1.val) := by
  have masks := legal_all_masks s c legal f j i.2.1.val hf hj i.2.1.isLt
  unfold aWord
  erw [if_pos masks.1]
  unfold aOffset inputWords
  congr 2
  simp [parameters, Parameters.steps, Shape.steps, ceilDiv, Nat.add_mul, Nat.mul_assoc, Nat.add_comm, Nat.add_left_comm,
    Nat.add_assoc]

theorem admitted_tile_casts (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (state : State Word32) (mpos : 0 < s.m) (defined : FP32Domain layout s state) :
    InputDefined (parameters layout s c) state := by
  intro f hf j hj i
  rw [aWord_stream layout s c legal state f j hf hj i]
  apply defined _ (Nat.mod_lt _ mpos)
  exact (legal_all_masks s c legal f j i.2.1.val hf hj i.2.1.isLt).2.2.1

theorem packedWords_byte (layout : Layout) (state : State Word32) (column q : Nat)
    (bound : packedValues layout state column q < 256) :
    (packedByte (packedWords layout state column q)).toNat = packedValues layout state column q := by
  unfold packedByte packedWords
  simp only [BitVec.toNat_ofNat]
  norm_num
  exact bound

theorem bValue_stream (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (state : State Word32) (npos : 0 < s.n) (bytes : UInt8Domain layout s state)
    (f j : Nat) (hf : f < 4) (hj : j < s.steps c)
    (i : BIndex (parameters layout s c)) :
    bValue (parameters layout s c) state f j i =
      packedWeight (packedByte (packedWords layout state
        (inputColumn (parameters layout s c) state i.2.1) (j*(configs c).bk+i.1.val))) f := by
  have masks := legal_all_masks s c legal f j i.1.val hf hj i.1.isLt
  have byteBound := bytes (inputColumn (parameters layout s c) state i.2.1)
    (Nat.mod_lt _ npos) _ masks.2.2.2
  rw [bValue_source]
  erw [if_pos masks.2.1]
  rw [extracted_field]
  unfold packedWeight
  rw [packedWords_byte _ _ _ _ byteBound]
  unfold bOffset packedValues
  congr 4
  simp [parameters, Nat.add_mul, Nat.mul_assoc, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]

theorem wordDot_block (cast : Word32 → Byte)
    (castLeaf : ∀ word, DefinedCast word → cast word = referenceCast word)
    (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (state : State Word32) (mpos : 0 < s.m) (npos : 0 < s.n)
    (defined : FP32Domain layout s state) (bytes : UInt8Domain layout s state)
    (f j : Nat) (hf : f < 4) (hj : j < s.steps c)
    (i : CIndex (parameters layout s c)) :
    wordDot (aValue cast (parameters layout s c) state f j)
      (bValue (parameters layout s c) state f j) i =
      (modularBlock c (s.steps c) f j
        (inputWords layout state (inputRow (parameters layout s c) state i.1))
        (packedWords layout state (inputColumn (parameters layout s c) state i.2.1))).toInt := by
  rcases i with ⟨r,col,u⟩
  cases u
  rw [wordDot_sum, modularBlock_exact]
  congr 1
  rw [exactArithmetic, blockStep_integer]
  simp only [zero_add]
  rw [← List.sum_toFinset _ (List.nodup_finRange _), List.toFinset_finRange]
  rw [← Fin.sum_univ_eq_sum_range]
  congr 1
  apply Finset.sum_congr rfl
  intro q _
  rw [bValue_stream layout s c legal state npos bytes f j hf hj]
  unfold aValue
  rw [aWord_stream layout s c legal state f j hf hj]
  erw [castLeaf _ (defined (inputRow (parameters layout s c) state r) (Nat.mod_lt _ mpos) _
    (legal_all_masks s c legal f j q.val hf hj q.isLt).2.2.1)]
  congr 2
  simp [Nat.add_mul, Nat.mul_assoc, Nat.add_assoc]

#axiomsClean aWord_stream
#axiomsClean admitted_tile_casts
#axiomsClean packedWords_byte
#axiomsClean bValue_stream
#axiomsClean wordDot_block
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
