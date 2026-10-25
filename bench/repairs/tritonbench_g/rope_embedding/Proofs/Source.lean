import bench.repairs.tritonbench_g.rope_embedding.Kernels
import bench.tritonbench_g.rotary_emb_nopad.Proofs.Pair
import VeriTile.Triton.Float.Control
namespace VeriTile.Bench.Repairs.RopeEmbedding.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

structure Parameters where
  (input cosine sine : RegionName)
  (rowStride cosineStride sineStride length dimension heads width : Nat)

/-- The unchanged four-head body, including both loaded halves, both stores,
the backwards sign, load-to-sine casts and FP32 output conversion. -/
def source (p : Parameters) (backward : Bool) : ComputeKernel :=
  Kernels.rope p.input p.cosine p.sine
    p.rowStride p.cosineStride p.sineStride p.length p.dimension p.heads p.width backward
def prelude (p : Parameters) (backward : Bool) := (source p backward).surfaceBody.take 10
def body (p : Parameters) (backward : Bool) : List ComputeStmt :=
  match (source p backward).surfaceBody.drop 10 with
  | [.forRangeDyn _ _ _ _ stmts] => stmts
  | _ => []

theorem decomposition (p : Parameters) (backward : Bool) :
    (source p backward).surfaceBody = prelude p backward ++
      [.forRangeDyn "k" (.ref .nat [] "head_start") (.ref .nat [] "head_end") (.constNat 1)
        (body p backward)] := by
  cases backward <;> rfl

abbrev Lane (p : Parameters) := TileIndex [p.width]
def active (p : Parameters) (i : Lane p) : Prop := i.1.val < p.dimension/2
instance (p : Parameters) : DecidablePred (active p) := fun i => inferInstanceAs (Decidable (i.1.val < p.dimension/2))
def offset (p : Parameters) (row head : Nat) (second : Bool) (i : Lane p) :=
  row*p.rowStride+head*p.dimension+i.1.val+(if second then p.dimension/2 else 0)
def coefficient {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (region : RegionName) (stride : Nat) (i : Lane p) : α :=
  M.fp32Load (if active p i then (s.mem region (s.pids 0%p.length*stride+i.1.val)).read .real
  else M.literal (some .fp32) .real 0)

structure Context {α : Type} (p : Parameters) (row : Nat) (cs ss : Lane p → α) (s : State α) : Prop where
  row : s.regs .nat [] "row_position" = some (fun _ => row)
  columns : s.regs .nat [p.width] "col_offsets" = some (fun i => i.1.val)
  half : s.regs .nat [] "half_head_dim" = some (fun _ => p.dimension/2)
  mask : s.regs .bool [p.width] "mask" = some (fun i => decide (active p i))
  cosine : s.regs .real [p.width] "cos1" = some cs
  sine : s.regs .real [p.width] "sin1" = some ss

def prepared {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (backward : Bool) (s : State α) :
    {v : State α // run M (prelude p backward) s=some v} := by
  cases backward <;>
    (refine ⟨?_,?_⟩
     swap
     simp [prelude,source,Kernels.rope,ComputeKernel.surfaceBody,
       run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
       Region.cast,Option.bind,integralFloorDiv,integralMod]
     try rfl)

theorem prepared_facts {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (backward : Bool) (s : State α) :
    Context p (s.pids 0) (coefficient M p s p.cosine p.cosineStride)
      (fun i => if backward then M.unary (some .fp32) (.neg .real) (coefficient M p s p.sine p.sineStride i)
        else coefficient M p s p.sine p.sineStride i) (prepared M p backward s).val ∧
    (prepared M p backward s).val.mem=s.mem ∧
    (prepared M p backward s).val.regs .nat [] "head_start"=some (fun _ => s.pids 1*4) ∧
    (prepared M p backward s).val.regs .nat [] "head_end"=some (fun _ => min (s.pids 1*4+4) p.heads) := by
  cases backward <;> refine ⟨?_,rfl,?_,?_⟩
  all_goals try constructor
  all_goals simp [prepared,coefficient,active,State.setReg,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,integralMod,min_def_lt]
  all_goals rfl

def value {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (row head : Nat) (cs ss : Lane p → α) (s : State α) (second : Bool) (i : Lane p) : α :=
  let a := M.cast (some .fp32) .real .real ((s.mem p.input (offset p row head Bool.false i)).read .real)
  let b := M.cast (some .fp32) .real .real ((s.mem p.input (offset p row head Bool.true i)).read .real)
  let v := if second then M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul b (cs i)) (M.binary (some .fp32) .real .mul a (ss i))
    else M.binary (some .fp32) .real .sub
      (M.binary (some .fp32) .real .mul a (cs i)) (M.binary (some .fp32) .real .mul b (ss i))
  M.cast (some .fp32) .real .real v

def written {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (row head : Nat) (cs ss : Lane p → α) (s : State α) : State α :=
  RotaryEmbNopad.Pair.written p.input (active p) (offset p row head)
    (fun second i => .mk .real (value M p row head cs ss s second i)) s

def oneHead {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (backward : Bool)
    (row head : Nat) (cs ss : Lane p → α) (s : State α) (h : Context p row cs ss s) :
    {v : State α // run M (body p backward) (s.setReg "k" .nat [] (fun _ => head))=some v} := by
  cases backward <;>
    (refine ⟨?_,?_⟩
     swap
     simp [body,source,Kernels.rope,ComputeKernel.surfaceBody,
       run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
       Region.cast,Option.bind,State.setReg,h.row,h.columns,h.half,h.mask,h.cosine,h.sine]
     try rfl)

theorem oneHead_facts {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (backward : Bool)
    (row head : Nat) (cs ss : Lane p → α) (s : State α) (h : Context p row cs ss s) :
    Context p row cs ss (oneHead M p backward row head cs ss s h).val ∧
    (oneHead M p backward row head cs ss s h).val.mem=(written M p row head cs ss s).mem := by
  constructor
  · cases backward <;> constructor <;> simp [oneHead,State.setReg,h.row,h.columns,h.half,h.mask,h.cosine,h.sine]
  · cases backward <;> simp +contextual [oneHead,written,RotaryEmbNopad.Pair.written,
      RotaryEmbNopad.Pair.halfScatter,masked_scatter_memory,value,offset,active,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,State.setReg,Region.cast,
      h.row,h.columns,h.half,h.mask,h.cosine,h.sine]

theorem written_mem_congr {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (row head : Nat) (cs ss : Lane p → α) (s t : State α) (h : s.mem=t.mem) :
    (written M p row head cs ss s).mem=(written M p row head cs ss t).mem := by
  simp only [written,RotaryEmbNopad.Pair.written,RotaryEmbNopad.Pair.halfScatter,
    masked_scatter_memory,value,h]

def heads {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (row : Nat) (cs ss : Lane p → α) (start : Nat) : Nat → State α → State α
  | 0,s => s
  | count+1,s => heads M p row cs ss (start+1) count (written M p row start cs ss s)

theorem heads_mem_congr {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (row : Nat) (cs ss : Lane p → α) (start count : Nat) (s t : State α) (h : s.mem=t.mem) :
    (heads M p row cs ss start count s).mem=(heads M p row cs ss start count t).mem := by
  induction count generalizing start s t with
  | zero => exact h
  | succ count ih => exact ih _ _ _ (written_mem_congr M p row start cs ss s t h)

theorem loop_executes {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (backward : Bool)
    (row : Nat) (cs ss : Lane p → α) (start count : Nat) (s : State α) (h : Context p row cs ss s) :
    ∃ final, range M "k" start (start+count) 1 (body p backward) s=some final ∧
      Context p row cs ss final ∧ final.mem=(heads M p row cs ss start count s).mem := by
  induction count generalizing start s with
  | zero => exact ⟨s,range_done M _ _ _ _ _ _ (by omega),h,rfl⟩
  | succ count ih =>
    let v := (oneHead M p backward row start cs ss s h).val
    have hv := (oneHead M p backward row start cs ss s h).property
    have hf := oneHead_facts M p backward row start cs ss s h
    obtain ⟨final,he,hc,hm⟩ := ih (start+1) v hf.1
    refine ⟨final,?_,hc,?_⟩
    · rw [range_step M _ _ _ _ _ _ (by decide) (by omega),hv]
      simpa only [Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using he
    · exact hm.trans (heads_mem_congr M p row cs ss (start+1) count v
        (written M p row start cs ss s) hf.2)

#axiomsClean loop_executes
#axiomsClean oneHead_facts
end
end VeriTile.Bench.Repairs.RopeEmbedding.Source
