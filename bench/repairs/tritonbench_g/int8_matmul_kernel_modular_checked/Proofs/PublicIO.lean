import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.IO
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.AccessSafety
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.KernelResults

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.PublicIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
open TileModel
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem input_binding (cast : Word32 → Byte) (rest : Algebra Word32)
    (layout : Layout) (s : Shape) (c : Fin 15) (xs : Inputs layout s)
    (initial : State Word32) (hin : (matmulIO cast rest layout s c).InputBinding xs initial)
    (m : Fin s.m) (k : Nat) (hk : k<s.k) :
    inputWords layout initial m.val k=inputRow layout s xs m k := by
  have h := hin ⟨0,by simp [matmulIO]⟩ (m,⟨k,hk⟩,PUnit.unit) trivial
  simpa [matmulIO,ProgramIO.InputBinding,mapValue,inputWords,inputRow,hk] using h

theorem weight_binding (cast : Word32 → Byte) (rest : Algebra Word32)
    (layout : Layout) (s : Shape) (c : Fin 15) (xs : Inputs layout s)
    (initial : State Word32) (hin : (matmulIO cast rest layout s c).InputBinding xs initial)
    (n : Fin s.n) (q : Nat) (hq : q<s.k/4) :
    packedValues layout initial n.val q=xs ⟨1,by simp⟩ (n,⟨q,hq⟩,PUnit.unit) := by
  exact hin ⟨1,by simp [matmulIO]⟩ (n,⟨q,hq⟩,PUnit.unit) trivial

theorem domains (cast : Word32 → Byte) (rest : Algebra Word32)
    (layout : Layout) (s : Shape) (c : Fin 15) (xs : Inputs layout s)
    (initial : State Word32) (hin : (matmulIO cast rest layout s c).InputBinding xs initial)
    (hd : Domain layout s xs) : FP32Domain layout s initial ∧ UInt8Domain layout s initial := by
  constructor
  · intro m hm k hk
    rw [input_binding cast rest layout s c xs initial hin ⟨m,hm⟩ k hk]
    simpa [inputRow,hk] using hd.1 (⟨m,hm⟩,⟨k,hk⟩,PUnit.unit)
  · intro n hn q hq
    rw [weight_binding cast rest layout s c xs initial hin ⟨n,hn⟩ q hq]
    exact hd.2 (⟨n,hn⟩,⟨q,hq⟩,PUnit.unit)

theorem source_bounds (cast : Word32 → Byte) (rest : Algebra Word32)
    (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (xs : Inputs layout s) (bounds : RegionName → Nat)
    (hb : (matmulIO cast rest layout s c).InBounds bounds xs)
    (initial : State Word32) (defined : FP32Domain layout s initial)
    (mpos : 0<s.m) (npos : 0<s.n) (hout : OutputLayout layout s) :
    TraceSafe (backend cast rest) bounds (source (parameters layout s c)) initial := by
  apply TileModel.source_safe cast rest _ initial bounds
    (admitted_tile_casts layout s c legal initial mpos defined)
  · intro f hf k hk i _
    have masks := legal_all_masks s c legal f k i.2.1.val hf hk i.2.1.isLt
    have hi := hb.2.1 ⟨0,by simp [matmulIO]⟩
      (⟨TileModel.inputRow (parameters layout s c) initial i.1,Nat.mod_lt _ mpos⟩,
        ⟨(f*s.steps c+k)*(configs c).bk+i.2.1.val,masks.2.2.1⟩,PUnit.unit) trivial
    convert hi using 1 <;>
      simp [matmulIO,aOffset,parameters,Parameters.steps,Shape.steps,ceilDiv,
        Nat.add_mul,Nat.mul_assoc,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc]
  · intro k hk i _
    have masks := legal_all_masks s c legal 0 k i.1.val (by decide) hk i.1.isLt
    have hi := hb.2.1 ⟨1,by simp [matmulIO]⟩
      (⟨inputColumn (parameters layout s c) initial i.2.1,Nat.mod_lt _ npos⟩,
        ⟨k*(configs c).bk+i.1.val,masks.2.2.2⟩,PUnit.unit) trivial
    convert hi using 1 <;>
      simp [matmulIO,bOffset,parameters,Nat.add_mul,Nat.mul_assoc,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc]
  · intro i hi
    rw [output_coordinates_source layout s c initial i hout]
    exact hb.2.2.1 ⟨0,by simp [matmulIO]⟩
      (⟨outputRow (parameters layout s c) initial i,hi.1⟩,
        ⟨outputColumn (parameters layout s c) initial i,hi.2⟩,PUnit.unit) trivial

/-- Logical formulas depend only on the finite input channels. -/
theorem dense_congr (K : Nat) (xa xb ya yb : Nat → Word32)
    (hx : ∀ k,k<K → xa k=xb k) (hy : ∀ q,q<K/4 → ya q=yb q) :
    denseWordTarget K xa ya=denseWordTarget K xb yb := by
  unfold denseWordTarget
  congr 1
  unfold denseSignedSum
  apply Finset.sum_congr rfl
  intro f hf
  apply Finset.sum_congr rfl
  intro q hq
  have hf' := Finset.mem_range.mp hf
  have hq' := Finset.mem_range.mp hq
  rw [hx _ (by have := Nat.div_mul_le_self K 4; nlinarith),hy _ hq']

theorem target_binding (cast : Word32 → Byte) (rest : Algebra Word32)
    (layout : Layout) (s : Shape) (c : Fin 15) (xs : Inputs layout s)
    (initial : State Word32) (hin : (matmulIO cast rest layout s c).InputBinding xs initial)
    (m : Fin s.m) (n : Fin s.n) :
    (matrixWordTarget layout s initial m.val n.val).toInt=
      formula layout s xs ⟨0,by simp⟩ (m,n,PUnit.unit) := by
  apply congrArg BitVec.toInt
  apply dense_congr
  · exact input_binding cast rest layout s c xs initial hin m
  · intro q hq
    unfold packedWords
    rw [weight_binding cast rest layout s c xs initial hin n q hq]
    simp [weightRow,hq]

#axiomsClean source_bounds
#axiomsClean target_binding
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.PublicIO
