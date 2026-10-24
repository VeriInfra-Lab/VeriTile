import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.LogicalInputs
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.KeyPartition
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.Initialization
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealLoopContext

/-! The actual two-loop closed carry normalizes to the independent causal
prefix formula, with arbitrary context tails and logical feature dimensions. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.CausalFormula
open VeriTile.Triton Memory LogicalInputs
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open RealLoopFormula (Carry rowSeed closed_row)
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def pair (mem : BlockState) (c : Config) (row : Fin 128) (lane : Nat) (key : Nat) : ℝ × ℝ :=
  (RealProofs.score c.p.heads c.p.context c.p.headDim c.z c.h (c.startM*128+row.val) key (inputs mem c.p c.offsets),
    (inputs mem c.p c.offsets).value c.z c.h key lane)

def blockPairs (mem : BlockState) (c : Config) (BD start : Nat) (diagonal : Bool)
    (row : Fin 128) (lane : Fin BD) : List (ℝ × ℝ) :=
  RealTiles.pairs (Preparation.score mem c 128 64 BD start row)
    (fun key => vvalue mem c 64 BD start (key,lane,PUnit.unit))
    (Masks.keep 128 64 c.p.context c.startM start diagonal row)

def blocks (mem : BlockState) (c : Config) (BD lo count : Nat) (diagonal : Bool)
    (row : Fin 128) (lane : Fin BD) : List (List (ℝ × ℝ)) :=
  (List.range count).map (fun i => blockPairs mem c BD (Loop.cursor lo 64 i) diagonal row lane)

theorem block_global (mem : BlockState) (c : Config) (BD start : Nat) (diagonal : Bool)
    (row : Fin 128) (lane : Fin BD) (hD : c.p.headDim ≤ BD)
    (hr : c.startM*128+row.val < c.p.context) (hd : lane.val < c.p.headDim) (ha : start%64=0) :
    blockPairs mem c BD start diagonal row lane =
      ((List.range' start 64).filter (KeyPartition.keep c.p.context (c.startM*128+row.val) diagonal)).map
        (pair mem c row lane.val) := by
  rw [← RealLoopContext.finRange_global,List.filter_map,List.map_map]
  unfold blockPairs RealTiles.pairs
  have hm : Masks.keep 128 64 c.p.context c.startM start diagonal row =
      fun key : Fin 64 => KeyPartition.keep c.p.context (c.startM*128+row.val) diagonal (start+key.val) := by
    rfl
  simp only [Function.comp_def]
  rw [hm]
  apply List.map_congr_left
  intro key hkey
  have hk : start+key.val < c.p.context := by
    have h := (List.mem_filter.mp hkey).2
    cases diagonal <;> simp [KeyPartition.keep] at h <;> tauto
  exact Prod.ext (score_at mem c BD start row key hD hr hk ha) (value_at mem c BD start key lane hk hd)

theorem blocks_flatten (mem : BlockState) (c : Config) (BD lo count : Nat) (diagonal : Bool)
    (row : Fin 128) (lane : Fin BD) (hD : c.p.headDim ≤ BD)
    (hr : c.startM*128+row.val < c.p.context) (hd : lane.val < c.p.headDim) (ha : lo%64=0) :
    (blocks mem c BD lo count diagonal row lane).flatten =
      (KeyPartition.keys lo count c.p.context (c.startM*128+row.val) diagonal).map (pair mem c row lane.val) := by
  have hb (i : Nat) := block_global mem c BD (Loop.cursor lo 64 i) diagonal row lane hD hr hd
    (by simp only [Loop.cursor,VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealLoop.cursor]; omega)
  unfold blocks
  simp_rw [hb]
  change (List.range count).flatMap (fun i =>
    ((List.range' (lo+i*64) 64).filter (KeyPartition.keep c.p.context (c.startM*128+row.val) diagonal)).map
      (pair mem c row lane.val)) = _
  rw [← List.map_flatMap]
  rfl

theorem after_row (mem : BlockState) (c : Config) (BD lo count : Nat) (second diagonal : Bool)
    (seed : Carry 128 BD) (row : Fin 128) (lane : Fin BD) :
    let carry := Loop.after mem c 128 64 BD lo second diagonal seed count
    rowSeed carry row (carry.numerator (row,lane,PUnit.unit)) =
      RealFormula.result (rowSeed seed row (seed.numerator (row,lane,PUnit.unit)))
        (blocks mem c BD lo count (second || diagonal) row lane) := by
  dsimp only
  rw [Loop.after,closed_row]
  rfl

theorem both_row (mem : BlockState) (c : Config) (BD : Nat) (row : Fin 128) (lane : Fin BD) :
    let carry := TwoLoops.bothResult mem c 128 64 BD Bool.false (Initialization.initial 128 BD)
    rowSeed carry row (carry.numerator (row,lane,PUnit.unit)) =
      RealFormula.result RealFormula.initial
        (blocks mem c BD 0 (Loop.count 0 (c.startM*128) 64) Bool.false row lane ++
         blocks mem c BD (c.startM*128) (Loop.count (c.startM*128) (min ((c.startM+1)*128) c.p.context) 64)
           Bool.true row lane) := by
  dsimp only
  rw [TwoLoops.bothResult,after_row,TwoLoops.firstResult,after_row,
    VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealLoopFormula.result_append]
  rfl

theorem normalized (mem : BlockState) (c : Config) (BD : Nat) (row : Fin 128) (lane : Fin BD)
    (hD : c.p.headDim ≤ BD) (hr : c.startM*128+row.val < c.p.context) (hd : lane.val < c.p.headDim) :
    let carry := TwoLoops.bothResult mem c 128 64 BD Bool.false (Initialization.initial 128 BD)
    carry.numerator (row,lane,PUnit.unit)/carry.denominator row =
      RealProofs.expected c.p.heads c.p.context c.p.headDim c.z c.h (c.startM*128+row.val) lane.val (inputs mem c.p c.offsets) := by
  dsimp only
  have h := both_row mem c BD row lane
  have hn := congrArg RealFormula.RowState.numerator h
  have hl := congrArg RealFormula.RowState.denominator h
  simp only [rowSeed] at hn hl
  rw [hn,hl,RealFormula.initialized_result_normalized,List.flatten_append,
    blocks_flatten mem c BD 0 _ Bool.false row lane hD hr hd (by omega),
    blocks_flatten mem c BD (c.startM*128) _ Bool.true row lane hD hr hd (by omega),
    ← List.map_append,KeyPartition.partition c.startM c.p.context row.val row.isLt hr]
  simp only [RealFormula.unshiftedSum,RealProofs.expected,RealProofs.pairs,mul_one]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.CausalFormula
