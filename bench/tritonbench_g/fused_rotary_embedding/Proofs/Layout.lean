import bench.tritonbench_g.fused_rotary_embedding.Proofs.Execution
import bench.tritonbench_g.fused_rotary_embedding.Contract
import bench.tritonbench_g.fused_rotary_embedding.Proofs.Source

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
abbrev Mem (α : Type) := RegionName→Nat→Cell α

/-- Exact canonical contiguous strides passed by the unchanged wrapper.
Old/new K-cache layout selection is retained explicitly. -/
def parameters (s : Shape) : Parameters where
  x:=if s.newLayout then s.split else s.width
  q_token_stride:=s.headsQ*s.width
  q_head_stride:=s.width
  q_dim_stride:=1
  k_token_stride:=s.headsK*s.width
  k_head_stride:=s.width
  k_dim_stride:=1
  v_token_stride:=s.headsK*s.width
  v_head_stride:=s.width
  v_dim_stride:=1
  cos_token_stride:=s.cosWidth
  cos_stride:=1
  sin_token_stride:=s.sinWidth
  sin_stride:=1
  length_stride:=1
  kcb_stride:=if s.newLayout then s.headsK*(s.width/s.split)*s.blockSize*s.split else s.headsK*s.blockSize*s.width
  kch_stride:=if s.newLayout then (s.width/s.split)*s.blockSize*s.split else s.blockSize*s.width
  kcsplit_x_stride:=if s.newLayout then s.blockSize*s.split else 0
  kcs_stride:=if s.newLayout then s.split else s.width
  kcd_stride:=1
  vcb_stride:=s.headsK*s.blockSize*s.width
  vch_stride:=s.blockSize*s.width
  vcs_stride:=s.width
  vcd_stride:=1
  bts_stride:=s.tableWidth
  btb_stride:=1
  block_size:=s.blockSize
  KV_GROUP_NUM:=s.groups
  HEAD_DIM:=s.width
  BLOCK_HALF:=s.halfBlock
  BLOCK_DIM:=s.fullBlock

def region (r : FusedRotaryEmbedding.Region) : RegionName :=
  match r with
  | .q=>"q" | .k=>"k" | .v=>"v" | .cos=>"cos" | .sin=>"sin"
  | .kcache=>"k_cache" | .vcache=>"v_cache" | .lengths=>"context_lengths" | .table=>"BLOCK_TABLES"
  | .other _=>"other"
def decoded (r : RegionName) : FusedRotaryEmbedding.Region :=
  if r="q" then .q else if r="k_cache" then .kcache else if r="v_cache" then .vcache else
  if r="k" then .k else if r="v" then .v else if r="cos" then .cos else if r="sin" then .sin else
  if r="context_lengths" then .lengths else if r="BLOCK_TABLES" then .table else .other 0

def lift {α : Type} (c : Config) (m : Memory α) : Mem α := fun r a=>
  if r="context_lengths" then .mk .int (c.metadata.lengths a : Int)
  else if r="BLOCK_TABLES" then .mk .int (c.metadata.table a : Int)
  else .mk .real (m (decoded r) a)
def state {α : Type} [Inhabited α] (job : Program) (memory : Mem α) : State α where
  mem:=memory
  regs:=fun _ _ _=>none
  pids:=fun ax=>if ax=0 then job.1 else if ax=1 then job.2 else 0
  numPids:=fun _=>1
  undef:=fun dtype _ _=>defaultValue dtype

theorem lift_write {α : Type} (c : Config) (m : Memory α) (kind : Kind) (a : Nat) (v : α) :
    lift c (Stores.store m ((kind.output,a),v))=
      fun r off=>if r=region kind.output ∧ off=a then .mk .real v else lift c m r off := by
  funext r off
  cases kind <;> simp only [Kind.output,region,lift,decoded,Stores.store,Prod.mk.injEq]
  all_goals split_ifs <;> simp_all

theorem lift_commit {α ι : Type} (c : Config) (m : Memory α) (kind : Kind) (xs : List ι)
    (off : ι→Nat) (value : ι→α) (s : State α) (memory : s.mem=lift c m) :
    (xs.foldl (fun t i=>t.write (region kind.output) (off i) (.mk .real (value i))) s).mem=
      lift c (Stores.commit (xs.map (fun i=>((kind.output,off i),value i))) m) := by
  induction xs generalizing m s with
  | nil=>exact memory
  | cons i xs ih=>
    simp only [List.foldl_cons,List.map_cons,Stores.commit,List.foldl_cons]
    apply ih
    rw [lift_write]
    simp only [State.write,memory]

theorem lanes_map {β : Type} (n : Nat) (f : Nat→β) :
    (TileShape.allIndices [n]).map (fun i=>f i.1.val)=(List.range n).map f := by
  calc
    _ = ((List.finRange n).map Fin.val).map f := by
      simp only [TileShape.allIndices,List.flatMap_nil,List.map_nil,List.map_cons,
        List.map_map,List.flatMap_cons,List.nil_append,List.append_nil,
        ←List.map_eq_flatMap,Function.comp_def]
    _ = _ := by rw [List.map_coe_finRange_eq_range]

theorem active_lanes {β : Type} (n limit : Nat) (bound : limit≤n) (f : Nat→β) :
    ((TileShape.allIndices [n]).filter (fun i=>decide (i.1.val<limit))).map (fun i=>f i.1.val)=
      (List.range limit).map f := by
  calc
    _ = (((TileShape.allIndices [n]).map (fun i=>i.1.val)).filter (fun i=>decide (i<limit))).map f := by
      simp [List.filter_map,List.map_map,Function.comp_def]
    _ = _ := by
      have lanes : (TileShape.allIndices [n]).map (fun i=>i.1.val)=List.range n := by
        simpa using lanes_map n id
      rw [lanes,masked_range n limit bound]

#axiomsClean lift_write
#axiomsClean lift_commit
#axiomsClean active_lanes
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
