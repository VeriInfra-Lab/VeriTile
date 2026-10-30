import bench.tritonbench_g.kv_cache_copy.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.Integer
import VeriTile.Triton.Float.Control
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.KvCacheCopy.KernelFP
open VeriTile.Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def position {α : Type} [Inhabited α] (s : State α) : Int :=
  (s.mem "lengths" (s.pids 0)).read .int - 1
def selectedBlock {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Int :=
  (s.mem "table" (s.pids 0*p.strideBts+((position s).tdiv p.blockSize*p.strideBtb).toNat)).read .int

def sourceOffset (p : Parameters) (ids : Nat → Nat) (key : Bool) (q : Nat)
    (i : TileIndex [p.width]) : Nat :=
  if key then ids 0*p.strideKt+ids 1*p.strideKh+(q*p.width+i.1.val)*p.strideKd
  else ids 0*p.strideVt+ids 1*p.strideVh+(q*p.width+i.1.val)*p.strideVd

def targetInt {α : Type} [Inhabited α] (p : Parameters) (s : State α) (key : Bool)
    (q : Nat) (i : TileIndex [p.width]) : Int :=
  if key then selectedBlock p s*p.strideKcb+(s.pids 1:Int)*p.strideKch+
    (q:Int)*p.strideKcsplit+(position s).tmod p.blockSize*p.strideKcs+i.1.val
  else selectedBlock p s*p.strideVcb+(s.pids 1:Int)*p.strideVch+
    (position s).tmod p.blockSize*p.strideVcs+(q*p.width+i.1.val)*p.strideVcd

def outputRegion (key : Bool) : RegionName := if key then "keyCache" else "valueCache"
def inputRegion (key : Bool) : RegionName := if key then "key" else "value"

def scatterOne {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (key : Bool) (q : Nat) (t : State α) : State α :=
  (TileShape.allIndices [p.width]).foldl (fun z i =>
    z.write (outputRegion key) (targetInt p s key q i).toNat
      (.mk .real (M.fp32Load ((s.mem (inputRegion key) (sourceOffset p s.pids key q i)).read .real)))) t

def copyStep {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (q : Nat) (t : State α) : State α :=
  scatterOne M p s Bool.false q (scatterOne M p s Bool.true q t)

def copied {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : Nat → State α
  | 0 => s
  | q+1 => copyStep M p s q (copied M p s q)

def preludeState {α : Type} [Inhabited α] (p : Parameters) (s : State α) : State α :=
  let z := s.setReg "cur_seq_idx" .nat [] (fun _ => s.pids 0)
  let z := z.setReg "cur_kv_head_idx" .nat [] (fun _ => s.pids 1)
  let z := z.setReg "past_kv_seq_len" .int [] (fun _ => position s)
  let z := z.setReg "last_bt_block_idx" .int [] (fun _ => (position s).tdiv p.blockSize)
  let z := z.setReg "block_table_ptr" .ptr [] (fun _ => ("table",s.pids 0*p.strideBts))
  let z := z.setReg "block_id" .int [] (fun _ => selectedBlock p s)
  let z := z.setReg "offsets_in_last_block" .int [] (fun _ => (position s).tmod p.blockSize)
  let z := z.setReg "range_x" .nat [p.width] (fun i => i.1.val)
  z.setReg "offsets_dmodel_x_partition" .nat [p.width] (fun i => i.1.val)

structure Controls {α : Type} [Inhabited α] (p : Parameters) (s t : State α) : Prop where
  seq : t.regs .nat [] "cur_seq_idx"=some (fun _ => s.pids 0)
  head : t.regs .nat [] "cur_kv_head_idx"=some (fun _ => s.pids 1)
  block : t.regs .int [] "block_id"=some (fun _ => selectedBlock p s)
  slot : t.regs .int [] "offsets_in_last_block"=some (fun _ => (position s).tmod p.blockSize)
  lanes : t.regs .nat [p.width] "range_x"=some (fun i => i.1.val)

def Frozen {α : Type} (s t : State α) : Prop := t.mem "key"=s.mem "key" ∧ t.mem "value"=s.mem "value"

def preparedState {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (q : Nat) (t : State α) : State α :=
  let z := t.setReg "split_x" .nat [] (fun _ => q)
  let z := z.setReg "offsets_dmodel_x_partition" .nat [p.width] (fun i => q*p.width+i.1.val)
  let z := z.setReg "offsets_k" .nat [p.width] (sourceOffset p s.pids Bool.true q)
  let z := z.setReg "k" .real [p.width] (fun i => M.fp32Load ((s.mem "key" (sourceOffset p s.pids Bool.true q i)).read .real))
  let z := z.setReg "offsets_v" .nat [p.width] (sourceOffset p s.pids Bool.false q)
  let z := z.setReg "v" .real [p.width] (fun i => M.fp32Load ((s.mem "value" (sourceOffset p s.pids Bool.false q i)).read .real))
  z.setReg "offsets_kcache" .int [p.width] (targetInt p s Bool.true q)

def finalStep {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (q : Nat) (t : State α) : State α :=
  let z := scatterOne M p s Bool.true q (preparedState M p s q t)
  let z := z.setReg "offsets_vcache" .int [p.width] (targetInt p s Bool.false q)
  scatterOne M p s Bool.false q z

def prelude (p : Parameters) := (originalKernel p).surfaceBody.take 9
def body (v : Bool) (p : Parameters) :=
  match (kernel v p).surfaceBody[9]? with
  | some (.forLoop _ _ body) => body
  | _ => []

theorem decomposition (v : Bool) (p : Parameters) :
    (kernel v p).surfaceBody=prelude p ++ [.forLoop "split_x" (p.dims/p.width) (body v p)] := by
  cases v <;> rfl

theorem prelude_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    run M (prelude p) s=some (preludeState p s) := by
  simp [prelude,originalKernel,kernel,quotedOriginal,quotedCommuted,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,numeric,numericLt,bop,Region.cast,Option.bind,
    FP.Structural.eval_truncDiv,FP.Structural.eval_truncMod,preludeState,position,selectedBlock,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals try dsimp only [bop,Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.sameShape]
  all_goals simp [integralFloorDiv,←IntegerExpr.tdiv_formula,Int.tmod_def,
    Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.same_left,IntegerExpr.same_right]
  all_goals congr 20 <;> funext <;> try simp [bop,Broadcast.leftIndex,Broadcast.rightIndex]

theorem body_run {α : Type} [Inhabited α] (M : Algebra α) (v : Bool) (p : Parameters)
    (s t : State α) (q : Nat) (hc : Controls p s t) (hf : Frozen s t) :
    run M (body v p) (t.setReg "split_x" .nat [] (fun _ => q))=some (finalStep M p s q t) := by
  rcases hc with ⟨hs,hh,hb,hslot,hl⟩
  rcases hf with ⟨hk,hv⟩
  cases v <;> simp [body,kernel,quotedOriginal,quotedCommuted,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,store,numeric,bop,ComputeDType.eraseDType,Region.cast,Option.bind,
    finalStep,preparedState,scatterOne,sourceOffset,targetInt,outputRegion,inputRegion,
    hs,hh,hb,hslot,hl,hk,hv,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals try dsimp only [bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]
  all_goals simp [Nat.add_comm,Nat.add_left_comm,Nat.add_assoc]
  all_goals congr 20
  all_goals funext
  all_goals try simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,sourceOffset,targetInt,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc]
  all_goals try ring
  all_goals try simp


theorem scatter_memory {α : Type} [Inhabited α] {I : Type} (xs : List I)
    (t : State α) (r : RegionName) (offset : I → Nat) (value : I → Cell α) :
    (xs.foldl (fun z i => z.write r (offset i) (value i)) t).mem =
      xs.foldl (fun m i => fun r' o => if r'=r ∧ o=offset i then value i else m r' o) t.mem := by
  induction xs generalizing t with
  | nil => rfl
  | cons i xs ih => exact ih (t.write r (offset i) (value i))

theorem finalStep_memory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (q : Nat) :
    (finalStep M p s q t).mem=(copyStep M p s q t).mem := by
  simp only [finalStep,copyStep,scatterOne,scatter_memory,State.setReg_mem,preparedState]
  all_goals rfl

theorem copyStep_congr {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t u : State α) (q : Nat) (hm : t.mem=u.mem) :
    (copyStep M p s q t).mem=(copyStep M p s q u).mem := by
  simp only [copyStep,scatterOne,scatter_memory,hm]

theorem scatterOne_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (key : Bool) (q : Nat) (r : RegionName) (o : Nat)
    (h : r≠outputRegion key ∨ ∀ i : TileIndex [p.width],o≠(targetInt p s key q i).toNat) :
    (scatterOne M p s key q t).mem r o=t.mem r o := by
  apply State.scatter_frame
  exact h.imp id (fun hh i _ => hh i)

theorem copyStep_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (q : Nat) (r : RegionName) (o : Nat)
    (h : ∀ key,r≠outputRegion key ∨ ∀ i : TileIndex [p.width],o≠(targetInt p s key q i).toNat) :
    (copyStep M p s q t).mem r o=t.mem r o := by
  unfold copyStep
  rw [scatterOne_frame M p s _ Bool.false q r o (h Bool.false),
    scatterOne_frame M p s _ Bool.true q r o (h Bool.true)]

theorem finalStep_controls {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (q : Nat) (h : Controls p s t) : Controls p s (finalStep M p s q t) := by
  rcases h with ⟨hs,hh,hb,hslot,hl⟩
  constructor <;> simp [finalStep,scatterOne,preparedState,hs,hh,hb,hslot,hl]

theorem finalStep_frozen {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (q : Nat) (h : Frozen s t) : Frozen s (finalStep M p s q t) := by
  constructor
  · funext o
    rw [finalStep_memory,copyStep_frame M p s t q "key" o (by intro key;cases key <;> exact Or.inl (by decide))]
    exact congrFun h.1 o
  · funext o
    rw [finalStep_memory,copyStep_frame M p s t q "value" o (by intro key;cases key <;> exact Or.inl (by decide))]
    exact congrFun h.2 o

/-- Complete static-loop execution, with one shared snapshot memory result for
both actual source versions. Successful execution is part of the conclusion. -/
theorem execute {α : Type} [Inhabited α] (M : Algebra α) (v : Bool)
    (p : Parameters) (s : State α) :
    ∃ final,exec M (kernel v p) s=some final ∧
      final.mem=(copied M p s (p.dims/p.width)).mem := by
  let P := fun q t => Controls p s t ∧ Frozen s t ∧ t.mem=(copied M p s q).mem
  have initial : P 0 (preludeState p s) := by
    refine ⟨?_,⟨rfl,rfl⟩,rfl⟩
    constructor <;> simp [preludeState]
  have step : ∀ q t,q<p.dims/p.width → P q t →
      ∃ u,run M (body v p) (t.setReg "split_x" .nat [] (fun _ => q))=some u ∧ P (q+1) u := by
    intro q t _ ht
    refine ⟨finalStep M p s q t,body_run M v p s t q ht.1 ht.2.1,?_,?_,?_⟩
    · exact finalStep_controls M p s t q ht.1
    · exact finalStep_frozen M p s t q ht.2.1
    · exact (finalStep_memory M p s t q).trans (copyStep_congr M p s t _ q ht.2.2)
  obtain ⟨final,he,hp⟩ := forLoop_invariant M initial step
  refine ⟨final,?_,hp.2.2⟩
  change run M (kernel v p).surfaceBody s=some final
  rw [decomposition,run_append,prelude_run]
  simp [run,he]

theorem copied_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (n : Nat) (r : RegionName) (o : Nat)
    (h : ∀ key q,q<n → r≠outputRegion key ∨
      ∀ i : TileIndex [p.width],o≠(targetInt p s key q i).toNat) :
    (copied M p s n).mem r o=s.mem r o := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [copied,copyStep_frame M p s _ n r o (fun key => h key n (by omega))]
    exact ih (fun key q hq => h key q (by omega))

#axiomsClean prelude_run
#axiomsClean body_run
#axiomsClean execute
#axiomsClean copied_frame
end
end VeriTile.Bench.Repairs.KvCacheCopy.KernelFP
