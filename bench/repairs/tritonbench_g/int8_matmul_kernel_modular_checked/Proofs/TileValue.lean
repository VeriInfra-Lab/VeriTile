import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.TileDense

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 6000000

theorem innerValues_modular (cast : Word32 → Byte)
    (castLeaf : ∀ word, DefinedCast word → cast word = referenceCast word)
    (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (state : State Word32) (mpos : 0 < s.m) (npos : 0 < s.n)
    (defined : FP32Domain layout s state) (bytes : UInt8Domain layout s state)
    (f : Nat) (hf : f < 4) (initial : CIndex (parameters layout s c) → Int)
    (i : CIndex (parameters layout s c)) (seed : Word32) (hseed : initial i = seed.toInt)
    (count : Nat) (hc : count ≤ s.steps c) :
    innerValues cast (parameters layout s c) state f initial count i =
      (foldN (fun j carry => ptxAdd32 carry (modularBlock c (s.steps c) f j
        (inputWords layout state (inputRow (parameters layout s c) state i.1))
        (packedWords layout state (inputColumn (parameters layout s c) state i.2.1)))) count seed).toInt := by
  induction count with
  | zero => exact hseed
  | succ count ih =>
    rw [innerValues]
    dsimp only
    rw [ih (by omega), wordDot_block cast castLeaf layout s c legal state mpos npos
      defined bytes f count hf (by omega) i]
    simp only [signedAdd32, BitVec.ofInt_toInt, foldN]

theorem outerValues_modular (cast : Word32 → Byte)
    (castLeaf : ∀ word, DefinedCast word → cast word = referenceCast word)
    (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (state : State Word32) (mpos : 0 < s.m) (npos : 0 < s.n)
    (defined : FP32Domain layout s state) (bytes : UInt8Domain layout s state)
    (i : CIndex (parameters layout s c)) (count : Nat) (hc : count ≤ 4) :
    outerValues cast (parameters layout s c) state count i =
      (foldN (fun f outer => foldN (fun j carry => ptxAdd32 carry
        (modularBlock c (s.steps c) f j
          (inputWords layout state (inputRow (parameters layout s c) state i.1))
          (packedWords layout state (inputColumn (parameters layout s c) state i.2.1))))
        (s.steps c) outer) count 0).toInt := by
  induction count with
  | zero => rfl
  | succ count ih =>
    rw [outerValues]
    exact innerValues_modular cast castLeaf layout s c legal state mpos npos defined bytes
      count (by omega) _ i _ (ih (by omega)) _ (by rfl)

theorem outerValues_low32 (cast : Word32 → Byte)
    (castLeaf : ∀ word, DefinedCast word → cast word = referenceCast word)
    (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (state : State Word32) (mpos : 0 < s.m) (npos : 0 < s.n)
    (defined : FP32Domain layout s state) (bytes : UInt8Domain layout s state)
    (i : CIndex (parameters layout s c)) :
    outerValues cast (parameters layout s c) state 4 i =
      (denseWordTarget s.k
        (inputWords layout state (inputRow (parameters layout s c) state i.1))
        (packedWords layout state (inputColumn (parameters layout s c) state i.2.1))).toInt := by
  rw [outerValues_modular cast castLeaf layout s c legal state mpos npos defined bytes i 4 (by decide)]
  change (modularAccumulation s c _ _).toInt = _
  exact congrArg BitVec.toInt (modularAccumulation_dense s c legal _ _
    (defined (inputRow (parameters layout s c) state i.1) (Nat.mod_lt _ mpos)))

#axiomsClean innerValues_modular
#axiomsClean outerValues_modular
#axiomsClean outerValues_low32
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
