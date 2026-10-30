import bench.tritonbench_g.fused_rotary_embedding.Proofs.Execution

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

abbrev HalfLane (p : Parameters) := TileIndex [p.BLOCK_HALF]
abbrev FullLane (p : Parameters) := TileIndex [p.BLOCK_DIM]
def quot (a b : Int) : Int := if a<0 then -((-a)/b) else a/b
def rem (a b : Int) : Int := a-b*quot a b
def dimension (p : Parameters) (high : Bool) (i : Nat) : Int :=
  i + if high then quot p.HEAD_DIM 2 else 0
def active (p : Parameters) (i : HalfLane p) : Prop := (i.1.val : Int)<quot p.HEAD_DIM 2
instance (p : Parameters) : DecidablePred (active p) := fun _=>inferInstanceAs (Decidable (_<_))
def head (p : Parameters) (s : State α) (key : Bool) : Int :=
  if key then quot (s.pids 0) p.KV_GROUP_NUM else s.pids 0
def dataOffset (p : Parameters) (s : State α) (key high : Bool) (i : HalfLane p) : Nat :=
  ((s.pids 1 : Int)*(if key then p.k_token_stride else p.q_token_stride)+
    head p s key*(if key then p.k_head_stride else p.q_head_stride)+
    dimension p high i.1.val*(if key then p.k_dim_stride else p.q_dim_stride)).toNat
def coefficientOffset (p : Parameters) (s : State α) (sine : Bool) (i : HalfLane p) : Nat :=
  ((s.pids 1 : Int)*(if sine then p.sin_token_stride else p.cos_token_stride)+
    (i.1.val : Int)*(if sine then p.sin_stride else p.cos_stride)).toNat
def zero (rest : Algebra α) : α := rest.literal (some .fp32) .real 0
def load (rest : Algebra α) (p : Parameters) (s : State α) (r : RegionName)
    (offset : HalfLane p→Nat) (i : HalfLane p) : α :=
  if active p i then (s.mem r (offset i)).read .real else zero rest
def pair (rest : Algebra α) (ops : Ops α) (p : Parameters) (s : State α)
    (key high : Bool) (i : HalfLane p) : Cell α :=
  .mk .real (component (sourcePair ops
    (load rest p s (if key then "k" else "q") (dataOffset p s key Bool.false) i)
    (load rest p s (if key then "k" else "q") (dataOffset p s key Bool.true) i)
    (load rest p s "cos" (coefficientOffset p s Bool.false) i)
    (load rest p s "sin" (coefficientOffset p s Bool.true) i)) high)
def past (p : Parameters) (s : State α) : Int :=
  (s.mem "context_lengths" ((s.pids 1 : Int)*p.length_stride).toNat).read .int-1
def block (p : Parameters) (s : State α) : Int :=
  (s.mem "BLOCK_TABLES" ((s.pids 1 : Int)*p.bts_stride+quot (past p s) p.block_size*p.btb_stride).toNat).read .int
def slot (p : Parameters) (s : State α) : Int := rem (past p s) p.block_size
def cacheOffset (p : Parameters) (s : State α) (high : Bool) (i : HalfLane p) : Nat :=
  (block p s*p.kcb_stride+head p s Bool.true*p.kch_stride+slot p s*p.kcs_stride+
    quot (dimension p high i.1.val) p.x*p.kcsplit_x_stride+rem (dimension p high i.1.val) p.x*p.kcd_stride).toNat
def valueOffset (p : Parameters) (s : State α) (i : FullLane p) : Nat :=
  ((s.pids 1 : Int)*p.v_token_stride+head p s Bool.true*p.v_head_stride+(i.1.val : Int)*p.v_dim_stride).toNat
def valueCacheOffset (p : Parameters) (s : State α) (i : FullLane p) : Nat :=
  (block p s*p.vcb_stride+head p s Bool.true*p.vch_stride+slot p s*p.vcs_stride+(i.1.val : Int)*p.vcd_stride).toNat
def scatter {ι : Type} (lanes : List ι) (r : RegionName) (mask : ι→Prop) [DecidablePred mask]
    (offset : ι→Nat) (values : ι→Cell α) (s : State α) : State α :=
  lanes.foldl (fun t i=>if mask i then t.write r (offset i) (values i) else t) s
def written (rest : Algebra α) (ops : Ops α) (p : Parameters) (s : State α) : State α :=
  let q0:=scatter (TileShape.allIndices [p.BLOCK_HALF]) "q" (active p)
    (dataOffset p s Bool.false Bool.false) (pair rest ops p s Bool.false Bool.false) s
  let q1:=scatter (TileShape.allIndices [p.BLOCK_HALF]) "q" (active p)
    (dataOffset p s Bool.false Bool.true) (pair rest ops p s Bool.false Bool.true) q0
  if s.pids 0%p.KV_GROUP_NUM=0 then
    let k0:=scatter (TileShape.allIndices [p.BLOCK_HALF]) "k_cache" (active p)
      (cacheOffset p s Bool.false) (pair rest ops p s Bool.true Bool.false) q1
    let k1:=scatter (TileShape.allIndices [p.BLOCK_HALF]) "k_cache" (active p)
      (cacheOffset p s Bool.true) (pair rest ops p s Bool.true Bool.true) k0
    scatter (TileShape.allIndices [p.BLOCK_DIM]) "v_cache" (fun i=>i.1.val<p.HEAD_DIM)
      (valueCacheOffset p s) (fun i=>.mk .real ((s.mem "v" (valueOffset p s i)).read .real)) k1
  else q1

@[simp] theorem scatter_read_other {ι : Type} (lanes : List ι) (r other : RegionName)
    (mask : ι→Prop) [DecidablePred mask] (offset : ι→Nat) (values : ι→Cell α)
    (s : State α) (off : Nat) (different : other≠r) :
    (lanes.foldl (fun t i=>if mask i then t.write r (offset i) (values i) else t) s).mem other off=
      s.mem other off := State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl different) _

/-- The complete quotation's memory effect equals the independent masked
Q/K/V vector stores. Metadata is loaded from the entry memory, not supplied. -/
theorem execution_memory (rest : Algebra α) (ops : Ops α) (p : Parameters) (s : State α) :
    (execution rest ops p s).val.mem=(written rest ops p s).mem := by
  have nonnegative (n : Nat) : ¬(n : Int)<0 := by omega
  by_cases leader : s.pids 0%p.KV_GROUP_NUM=0
  all_goals
    simp (config := {contextual := Bool.true, maxSteps := 1000000}) [execution,written,scatter,active,pair,load,zero,
      dataOffset,coefficientOffset,cacheOffset,valueOffset,valueCacheOffset,past,block,slot,head,dimension,
      sourcePair,component,leader,bop,Broadcast.leftIndex,Broadcast.rightIndex,
      integralFloorDiv,integralMod,quot,rem,nonnegative,masked_scatter_memory]

#axiomsClean execution_memory
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
