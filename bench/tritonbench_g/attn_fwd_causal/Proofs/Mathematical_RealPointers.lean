import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealCoordinates

/-! Source index vectors and pointer expressions. Query, value and output
share the source Q strides, while the key uses stride_kn and an unscaled
head lane. The unused source K/V/Out stride parameters are not substituted. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealPointers
open VeriTile.Triton RealLoads
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def rowExpr (BM : Nat) : Op .nat [BM] :=
  .add .nat .scalarL (.mul .nat .nil (.ref .nat [] "start_m") (.constNat BM)) (.arange BM)

def rowPointerExpr (R : RegionName) (BR BD strideRow strideDim : Nat) (rows : RegName) : Op .ptr [BR,BD] :=
  .ptrAdd .scalarL (.ptrBase R)
    (.add .nat (.consR (.consL .nil))
      (.add .nat .scalarL (.ref .nat [] "qvk_offset")
        (.mul .nat .scalarR (.expandDim ⟨1, by simp⟩ (.ref .nat [BR] rows)) (.constNat strideRow)))
      (.mul .nat .scalarR (.expandDim ⟨0, by simp⟩ (.ref .nat [BD] "offs_k")) (.constNat strideDim)))

def keyExpr (K : RegionName) (BD BN strideKN : Nat) : Op .ptr [BD,BN] :=
  .ptrAdd .scalarL (.ptrBase K)
    (.add .nat (.consR (.consL .nil))
      (.add .nat .scalarL (.ref .nat [] "qvk_offset") (.expandDim ⟨1, by simp⟩ (.ref .nat [BD] "offs_k")))
      (.mul .nat .scalarR (.expandDim ⟨0, by simp⟩ (.ref .nat [BN] "offs_n")) (.constNat strideKN)))

def queryScaleExpr (QS : RegionName) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase QS) (.add .nat .nil (.ref .nat [] "q_scale_offset") (.ref .nat [] "start_m"))

def keyScaleExpr (KS : RegionName) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase KS) (.ref .nat [] "k_scale_offset")

def rows (BR start : Nat) : Tile .nat [BR] := ⟨fun i => start+i.1.val⟩
def rowPointer (R : RegionName) (BR BD base start strideRow strideDim : Nat) : Tile .ptr [BR,BD] :=
  ⟨fun i => (R.cast,base+(start+i.1.val)*strideRow+i.2.1.val*strideDim)⟩
def queryPointer (c : Config) (BM BD : Nat) : Tile .ptr [BM,BD] :=
  rowPointer c.Q BM BD c.base (c.startM*BM) c.strideM c.strideD
def outputPointer (c : Config) (Out : RegionName) (BM BD : Nat) : Tile .ptr [BM,BD] :=
  rowPointer Out BM BD c.base (c.startM*BM) c.strideM c.strideD
def queryScalePointer (c : Config) : Tile .ptr [] := Tile.scalar (c.QS.cast,c.queryScaleBase+c.startM)

theorem eval_rows (s : BlockState) (BM startM : Nat)
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar startM)) :
    evalOp (rowExpr BM) s = some (rows BM (startM*BM)) := by
  simp only [rowExpr, evalOp.eq_def, hm, bind, Option.bind]
  rfl

theorem eval_row_pointer (s : BlockState) (R : RegionName) (BR BD base start strideRow strideDim : Nat)
    (name : RegName)
    (hb : s.regs .nat [] "qvk_offset" = some (Tile.scalar base))
    (hr : s.regs .nat [BR] name = some (rows BR start))
    (hk : s.regs .nat [BD] "offs_k" = some (Tile.vec (fun i => i.val))) :
    evalOp (rowPointerExpr R BR BD strideRow strideDim name) s =
      some (rowPointer R BR BD base start strideRow strideDim) := by
  simp only [rowPointerExpr, evalOp.eq_def, hb, hr, hk, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨r,d,⟨⟩⟩
  simp [rowPointer, rows, Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, Tile.expandDim_data,
    Tile.vec, TileShape.dropInsertedIndex, NumericDType.mul, NumericDType.add, -Tile.scalar_eta]

theorem eval_key_pointer (s : BlockState) (c : Config) (BD BN : Nat)
    (hb : s.regs .nat [] "qvk_offset" = some (Tile.scalar c.base))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hk : s.regs .nat [BD] "offs_k" = some (Tile.vec (fun i => i.val))) :
    evalOp (keyExpr c.K BD BN c.strideKN) s = some (keyPointer c BD BN 0) := by
  simp only [keyExpr, evalOp.eq_def, hb, hn, hk, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨d,j,⟨⟩⟩
  simp [keyPointer, Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, Tile.expandDim_data,
    Tile.vec, TileShape.dropInsertedIndex, NumericDType.mul, NumericDType.add, -Tile.scalar_eta]

theorem eval_query_scale_pointer (s : BlockState) (c : Config)
    (hb : s.regs .nat [] "q_scale_offset" = some (Tile.scalar c.queryScaleBase))
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar c.startM)) :
    evalOp (queryScaleExpr c.QS) s = some (queryScalePointer c) := by
  simp only [queryScaleExpr, evalOp.eq_def, hb, hm, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨⟩
  simp [queryScalePointer, Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, NumericDType.add, -Tile.scalar_eta]

theorem eval_key_scale_pointer (s : BlockState) (c : Config) (BN : Nat)
    (hb : s.regs .nat [] "k_scale_offset" = some (Tile.scalar c.keyScaleBase)) :
    evalOp (keyScaleExpr c.KS) s = some (keyScalePointer c BN 0) := by
  simp only [keyScaleExpr, evalOp.eq_def, hb, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨⟩
  simp [keyScalePointer, Tile.scalar, -Tile.scalar_eta]

def writes : List RegName := ["offs_m","offs_n","offs_k","Q_ptrs","Q_scale_ptr","K_ptrs","K_scale_ptr","V_ptrs","O_block_ptr"]

structure Registers (c : Config) (Out : RegionName) (BM BN BD : Nat) (s : BlockState) : Prop where
  rows : s.regs .nat [BM] "offs_m" = some (rows BM (c.startM*BM))
  keys : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val))
  head : s.regs .nat [BD] "offs_k" = some (Tile.vec (fun i => i.val))
  query : s.regs .ptr [BM,BD] "Q_ptrs" = some (queryPointer c BM BD)
  queryScale : s.regs .ptr [] "Q_scale_ptr" = some (queryScalePointer c)
  caller : RealTwoLoops.Caller s c BN BD
  output : s.regs .ptr [BM,BD] "O_block_ptr" = some (outputPointer c Out BM BD)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealPointers
