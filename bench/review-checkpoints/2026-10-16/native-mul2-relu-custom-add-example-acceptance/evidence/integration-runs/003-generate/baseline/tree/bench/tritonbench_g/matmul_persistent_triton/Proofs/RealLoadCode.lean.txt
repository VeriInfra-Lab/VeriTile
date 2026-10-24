import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealProgram
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealTiles

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealLoads
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 14000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def kOffsets (p : Parameters) : Op .int [p.BK] :=
  .add .int .scalarL (.mul .int .nil (.ref .int [] "ki") (.constInt p.BK))
    (.castNatToInt (.arange p.BK))

def aPointers (p : Parameters) : Op .ptr [p.BM,p.BK] :=
  .ptrAdd .scalarL (.ptrBase (d := .real) (Region.cast p.A))
    (.castIntToNat (.add .int Broadcast.nil.consL.consR
      (.mul .int .scalarR (.expandDim ⟨1,by simp⟩ (.ref .int [p.BM] "offs_am"))
        (.castNatToInt (.constNat p.sam)))
      (.mul .int .scalarR (.expandDim ⟨0,by simp⟩ (.ref .int [p.BK] "offs_k"))
        (.castNatToInt (.constNat p.sak)))))

def bPointers (p : Parameters) : Op .ptr [p.BK,p.BN] :=
  .ptrAdd .scalarL (.ptrBase (d := .real) (Region.cast p.B))
    (.castIntToNat (.add .int Broadcast.nil.consL.consR
      (.mul .int .scalarR (.expandDim ⟨1,by simp⟩ (.ref .int [p.BK] "offs_k"))
        (.castNatToInt (.constNat p.sbk)))
      (.mul .int .scalarR (.expandDim ⟨0,by simp⟩ (.ref .int [p.BN] "offs_bn"))
        (.castNatToInt (.constNat p.sbn)))))

def remaining (p : Parameters) : Op .int [] :=
  .sub .int .nil (.castNatToInt (.constNat p.K))
    (.mul .int .nil (.ref .int [] "ki") (.castNatToInt (.constNat p.BK)))

def aLoad (p : Parameters) : Op .real [p.BM,p.BK] :=
  .load .real (.ptr (.ref .ptr [p.BM,p.BK] "a_ptrs"))
    (.maskOther
      (.remap [p.BM,p.BK] Broadcast.nil.consSame.consL.leftIndex
        (.lt .int .scalarR
          (.castNatToInt (.expandDim ⟨0,by simp⟩ (.ref .nat [p.BK] "offs_k_for_mask")))
          (remaining p)))
      (.broadcast (.const 0) [p.BM,p.BK]))

def bLoad (p : Parameters) : Op .real [p.BK,p.BN] :=
  .load .real (.ptr (.ref .ptr [p.BK,p.BN] "b_ptrs"))
    (.maskOther
      (.remap [p.BK,p.BN] Broadcast.nil.consL.consSame.leftIndex
        (.lt .int .scalarR
          (.castNatToInt (.expandDim ⟨1,by simp⟩ (.ref .nat [p.BK] "offs_k_for_mask")))
          (remaining p)))
      (.broadcast (.const 0) [p.BK,p.BN]))

def code (p : Parameters) : List Stmt :=
  [.assign .int [p.BK] "offs_k" (kOffsets p),
   .assign .ptr [p.BM,p.BK] "a_ptrs" (aPointers p),
   .assign .ptr [p.BK,p.BN] "b_ptrs" (bPointers p),
   .assign .real [p.BM,p.BK] "a" (aLoad p),
   .assign .real [p.BK,p.BN] "b" (bLoad p)]

theorem matches_source (p : Parameters) : RealProgram.loads p = code p := by
  simp only [RealProgram.loads, RealProgram.body, RealSource.kernel,
    Parameters.source, TypedSource.kernel, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.surfaceBody,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, FloatDType.toTileDType]
  simp only [List.getElem!_cons_zero, List.getElem!_cons_succ, RealProgram.loopBody, List.take, List.drop, code, kOffsets, aPointers, bPointers, aLoad, bLoad, remaining]
  congr 1 <;> norm_num

/-- Only original input addresses occur in these arrays; no output or prepared
accumulator appears in their definition. -/
def memoryInputs (p : Parameters) (s : BlockState) : RealMath.Inputs where
  a row k := s.readMem p.A (row * p.sam + k * p.sak)
  b k column := s.readMem p.B (k * p.sbk + column * p.sbn)

def aAddresses (p : Parameters) (row : Fin p.BM → Nat) (ki : Nat) : Tile .ptr [p.BM,p.BK] :=
  Tile.mat fun i k => (p.A, RealTiles.clamp p.M (row i) * p.sam + (ki*p.BK+k.val)*p.sak)
def bAddresses (p : Parameters) (column : Fin p.BN → Nat) (ki : Nat) : Tile .ptr [p.BK,p.BN] :=
  Tile.mat fun k j => (p.B, (ki*p.BK+k.val)*p.sbk + RealTiles.clamp p.N (column j)*p.sbn)

def result (p : Parameters) (s : BlockState)
    (row : Fin p.BM → Nat) (column : Fin p.BN → Nat) (ki : Nat) : BlockState :=
  let t := s.setReg "offs_k" .int [p.BK] (Tile.vec fun k => (ki*p.BK+k.val : Nat))
  let t := t.setReg "a_ptrs" .ptr [p.BM,p.BK] (aAddresses p row ki)
  let t := t.setReg "b_ptrs" .ptr [p.BK,p.BN] (bAddresses p column ki)
  let t := t.setReg "a" .real [p.BM,p.BK] (RealTiles.aValues p (memoryInputs p s) row ki)
  t.setReg "b" .real [p.BK,p.BN] (RealTiles.bValues p (memoryInputs p s) column ki)

theorem signed_mask (K BK ki lane : Nat) :
    ((lane : Int) < (K : Int) - (ki : Int)*BK) ↔ lane < K - ki*BK := by omega

/-- Loads and address calculations preserve all raw memory and program fields. -/
theorem frame (p : Parameters) (s : BlockState)
    (row : Fin p.BM → Nat) (column : Fin p.BN → Nat) (ki : Nat) :
    {result p s row column ki with regs := s.regs} = s := rfl

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealLoads
