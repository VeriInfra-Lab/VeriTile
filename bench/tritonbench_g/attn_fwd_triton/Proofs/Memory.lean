import bench.tritonbench_g.attn_fwd_triton.Proofs.SourceFragments
import bench.tritonbench_g.attn_fwd_triton.Proofs.Real

/-! Changed memory bindings: each role has independent strides, every load has
an explicit zero alternative, and helper advances use the actual row strides.
These are exact Real execution lemmas, with no undefined-memory premise. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Memory
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealTiles (finite)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 800000
set_option linter.unusedSimpArgs false

structure BaseOffsets where
  query : Nat := 0
  key : Nat := 0
  value : Nat := 0
  output : Nat := 0
  queryScale : Nat := 0
  keyScale : Nat := 0

structure Config where
  p : Parameters
  (z h startM : Nat)
  offsets : BaseOffsets := {}

def rows (B start : Nat) : Tile .int [B] := ⟨fun i => ((start+i.1.val : Nat) : Int)⟩

def qbase (c : Config) := c.z*c.p.queryBatch+c.h*c.p.queryHead+c.offsets.query
def kbase (c : Config) := c.z*c.p.keyBatch+c.h*c.p.keyHead+c.offsets.key
def vbase (c : Config) := c.z*c.p.valueBatch+c.h*c.p.valueHead+c.offsets.value
def obase (c : Config) := c.z*c.p.outputBatch+c.h*c.p.outputHead+c.offsets.output
def qsbase (c : Config) (BM : Nat) := (c.z*c.p.heads+c.h)*((c.p.context+BM-1)/BM)+c.offsets.queryScale
def ksbase (c : Config) (BN : Nat) := (c.z*c.p.heads+c.h)*((c.p.context+BN-1)/BN)+c.offsets.keyScale

def queryPointer (c : Config) (BM BD : Nat) : Tile .ptr [BM,BD] :=
  ⟨fun i => (c.p.Q.cast,qbase c+(c.startM*BM+i.1.val)*c.p.queryRow+i.2.1.val*c.p.queryDim)⟩
def keyPointer (c : Config) (BD BN start : Nat) : Tile .ptr [BD,BN] :=
  ⟨fun i => (c.p.K.cast,kbase c+i.1.val*c.p.keyDim+(start+i.2.1.val)*c.p.keyRow)⟩
def valuePointer (c : Config) (BN BD start : Nat) : Tile .ptr [BN,BD] :=
  ⟨fun i => (c.p.V.cast,vbase c+(start+i.1.val)*c.p.valueRow+i.2.1.val*c.p.valueDim)⟩
def outputPointer (c : Config) (BM BD : Nat) : Tile .ptr [BM,BD] :=
  ⟨fun i => (c.p.Out.cast,obase c+(c.startM*BM+i.1.val)*c.p.outputRow+i.2.1.val*c.p.outputDim)⟩
def queryScalePointer (c : Config) (BM : Nat) : Tile .ptr [] :=
  Tile.scalar (c.p.QS.cast,qsbase c BM+c.startM)
def keyScalePointer (c : Config) (BN start : Nat) : Tile .ptr [] :=
  Tile.scalar (c.p.KS.cast,ksbase c BN+start/BN)

def qvalue (s : BlockState) (c : Config) (BM BD : Nat) (i : TileIndex [BM,BD]) : ℝ :=
  if c.startM*BM+i.1.val < c.p.context ∧ i.2.1.val < c.p.headDim then
    s.readMem c.p.Q (queryPointer c BM BD |>.data i |>.2) else 0
def kvalue (s : BlockState) (c : Config) (BD BN start : Nat) (i : TileIndex [BD,BN]) : ℝ :=
  if start+i.2.1.val < c.p.context ∧ i.1.val < c.p.headDim then
    s.readMem c.p.K (keyPointer c BD BN start |>.data i |>.2) else 0
def vvalue (s : BlockState) (c : Config) (BN BD start : Nat) (i : TileIndex [BN,BD]) : ℝ :=
  if start+i.1.val < c.p.context ∧ i.2.1.val < c.p.headDim then
    s.readMem c.p.V (valuePointer c BN BD start |>.data i |>.2) else 0

def keyMaskTile (BD BN N D start : Nat) : Tile .bool [BD,BN] :=
  ⟨fun i => decide (start+i.2.1.val < N ∧ i.1.val < D)⟩
def valueMaskTile (BN BD N D start : Nat) : Tile .bool [BN,BD] :=
  ⟨fun i => decide (start+i.1.val < N ∧ i.2.1.val < D)⟩
def queryMaskTile (BM BD N D startM : Nat) : Tile .bool [BM,BD] :=
  ⟨fun i => decide (startM*BM+i.1.val < N ∧ i.2.1.val < D)⟩

theorem eval_key_mask (s : BlockState) (BD BN N D start : Nat)
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hctx : s.regs .int [] "n_ctx" = some (Tile.scalar N)) :
    evalOp (keyMask BD BN D) s = some (keyMaskTile BD BN N D start) := by
  simp only [keyMask, tail, evalOp.eq_def, hn, hs, hctx, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨d,j,⟨⟩⟩
  simp [keyMaskTile, Tile.cop_data, Tile.bop_data, Tile.expandDim_data, Tile.vec, Tile.scalar,
    TileShape.dropInsertedIndex, ComparableDType.lt, NumericDType.sub,
    VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealExpressions.tail_mask_iff,ControlView.signed_tail, -Tile.scalar_eta]
  change (decide ((j.val : Int)<(N : Int)-(start : Int)) && decide (d.val<D)) = _
  rw [ControlView.signed_tail_add]

theorem eval_value_mask (s : BlockState) (BN BD N D start : Nat)
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hctx : s.regs .int [] "n_ctx" = some (Tile.scalar N)) :
    evalOp (valueMask BN BD D) s = some (valueMaskTile BN BD N D start) := by
  simp only [valueMask, evalOp.eq_def, hn, hs, hctx, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨j,d,⟨⟩⟩
  simp [valueMaskTile, Tile.cop_data, Tile.bop_data, Tile.expandDim_data, Tile.vec, Tile.scalar,
    TileShape.dropInsertedIndex, ComparableDType.lt, NumericDType.sub,
    VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealExpressions.tail_mask_iff,ControlView.signed_tail, -Tile.scalar_eta]
  change (decide ((j.val : Int)<(N : Int)-(start : Int)) && decide (d.val<D)) = _
  rw [ControlView.signed_tail_add]

theorem eval_query_mask (s : BlockState) (BM BD N D startM : Nat)
    (hm : s.regs .int [BM] "offs_m" = some ⟨fun i => startM*BM+i.1.val⟩)
    (hctx : s.regs .int [] "n_ctx" = some (Tile.scalar N)) :
    evalOp (queryMask BM BD D) s = some (queryMaskTile BM BD N D startM) := by
  simp only [queryMask, evalOp.eq_def, hm, hctx, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨r,d,⟨⟩⟩
  simp [queryMaskTile, Tile.cop_data, Tile.bop_data, Tile.expandDim_data, Tile.vec, Tile.scalar,
    TileShape.dropInsertedIndex, ComparableDType.lt, -Tile.scalar_eta]
  change (decide ((startM : Int)*(BM : Int)+(r.val : Int)<(N : Int)) && decide (d.val<D)) = _
  simp only [←Int.natCast_mul,←Int.natCast_add]
  congr 1
  simp
  change (startM : Int)*(BM : Int)+(r.val : Int)<(N : Int) ↔ startM*BM+r.val<N
  simp only [←Int.natCast_mul,←Int.natCast_add]
  omega

theorem key_load (s : BlockState) (c : Config) (BD BN start : Nat) (pointer : RegName)
    (hp : s.regs .ptr [BD,BN] pointer = some (keyPointer c BD BN start))
    (hm : s.regs .bool [BD,BN] "k_mask" = some (keyMaskTile BD BN c.p.context c.p.headDim start)) :
    evalOp (.load .real (.ptr (.ref .ptr [BD,BN] pointer))
      (.maskOther (.ref .bool [BD,BN] "k_mask") (.broadcast (.const 0.0) [BD,BN]))) s =
      some (finite (kvalue s c BD BN start)) := by
  simp only [evalOp.eq_def, hp, hm, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨d,j,⟨⟩⟩
  simp [keyMaskTile, keyPointer, kvalue, finite, Region.cast, Tile.scalar, -Tile.scalar_eta]
  split <;> norm_num

theorem value_load (s : BlockState) (c : Config) (BN BD start : Nat) (pointer : RegName)
    (hp : s.regs .ptr [BN,BD] pointer = some (valuePointer c BN BD start))
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hctx : s.regs .int [] "n_ctx" = some (Tile.scalar c.p.context)) :
    evalOp (.load .real (.ptr (.ref .ptr [BN,BD] pointer))
      (.maskOther (valueMask BN BD c.p.headDim) (.broadcast (.const 0.0) [BN,BD]))) s =
      some (finite (vvalue s c BN BD start)) := by
  have hm := eval_value_mask s BN BD c.p.context c.p.headDim start hn hs hctx
  rw [evalOp.eq_def]
  simp only [evalOp_ref, hp, hm, evalOp.eq_def, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨j,d,⟨⟩⟩
  simp [valueMaskTile, valuePointer, vvalue, finite, Region.cast, Tile.scalar, -Tile.scalar_eta]
  split <;> norm_num

theorem query_load (s : BlockState) (c : Config) (BM BD : Nat)
    (hp : s.regs .ptr [BM,BD] "Q_ptrs" = some (queryPointer c BM BD))
    (hm : s.regs .int [BM] "offs_m" = some ⟨fun i => c.startM*BM+i.1.val⟩)
    (hctx : s.regs .int [] "n_ctx" = some (Tile.scalar c.p.context)) :
    evalOp (.load .real (.ptr (.ref .ptr [BM,BD] "Q_ptrs"))
      (.maskOther (queryMask BM BD c.p.headDim) (.broadcast (.const 0.0) [BM,BD]))) s =
      some (finite (qvalue s c BM BD)) := by
  have hmask := eval_query_mask s BM BD c.p.context c.p.headDim c.startM hm hctx
  rw [evalOp.eq_def]
  simp only [evalOp_ref, hp, hmask, evalOp.eq_def, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨j,d,⟨⟩⟩
  simp [queryMaskTile, queryPointer, qvalue, finite, Region.cast, Tile.scalar, -Tile.scalar_eta]
  split <;> norm_num

theorem scale_load (s : BlockState) (r : RegionName) (offset : Nat) (pointer : RegName)
    (hp : s.regs .ptr [] pointer = some (Tile.scalar (r.cast,offset))) :
    evalOp (.load .real (.ptr (.ref .ptr [] pointer)) .none) s =
      some (Tile.scalar (some (s.readMem r offset))) := by
  simp only [evalOp.eq_def, hp, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨⟩
  simp [Tile.scalar, Region.cast, -Tile.scalar_eta]

theorem key_advance (s : BlockState) (c : Config) (BN BD start : Nat) (name : RegName)
    (hp : s.regs .ptr [BD,BN] name = some (keyPointer c BD BN start))
    (hr : s.regs .int [] "stride_kn64" = some (Tile.scalar c.p.keyRow)) :
    evalOp (.ptrAdd .scalarR (.ref .ptr [BD,BN] name)
      (.castIntToNat (.mul .int .nil (wide BN) (.ref .int [] "stride_kn64")))) s =
      some (keyPointer c BD BN (start+BN)) := by
  simp only [wide,evalOp.eq_def, hp, hr, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨d,j,⟨⟩⟩
  simp [keyPointer, Tile.scalar, NumericDType.mul, -Tile.scalar_eta]
  change _ + Int.toNat ((BN : Int)*(c.p.keyRow : Int)) = _
  simp only [←Int.natCast_mul,Int.toNat_natCast]
  ring

theorem value_advance (s : BlockState) (c : Config) (BN BD start : Nat) (name : RegName)
    (hp : s.regs .ptr [BN,BD] name = some (valuePointer c BN BD start))
    (hr : s.regs .int [] "stride_vk64" = some (Tile.scalar c.p.valueRow)) :
    evalOp (.ptrAdd .scalarR (.ref .ptr [BN,BD] name)
      (.castIntToNat (.mul .int .nil (wide BN) (.ref .int [] "stride_vk64")))) s =
      some (valuePointer c BN BD (start+BN)) := by
  simp only [wide,evalOp.eq_def, hp, hr, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨j,d,⟨⟩⟩
  simp [valuePointer, Tile.scalar, NumericDType.mul, -Tile.scalar_eta]
  change _ + Int.toNat ((BN : Int)*(c.p.valueRow : Int)) = _
  simp only [←Int.natCast_mul,Int.toNat_natCast]
  ring

theorem scale_advance (s : BlockState) (c : Config) (BN start : Nat) (hN : 0 < BN) (name : RegName)
    (hp : s.regs .ptr [] name = some (keyScalePointer c BN start)) :
    evalOp (.ptrAdd .nil (.ref .ptr [] name) (.constNat 1)) s =
      some (keyScalePointer c BN (start+BN)) := by
  simp only [evalOp.eq_def, hp, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨⟩
  simp [keyScalePointer, Tile.scalar, Nat.add_div_right start hN, Nat.add_assoc, -Tile.scalar_eta]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Memory
