import bench.repairs.tritonbench_g.embedding_triton_kernel.Kernels
import VeriTile.Triton.Float.Control
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
import bench.tritonbench_g.embedding_triton_kernel.Proofs.Reference

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.Execution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

variable {α : Type} [Inhabited α]

/-- Only the source's FP32 load and literal +0 are interpreted here. Every
other floating callback is retained but unused by this memory-copy source. -/
def backend (rest : Algebra α) (zero : α) : Algebra α :=
  { copyBackend rest with literal := fun _ _ _ => zero }

def prefixCode (index : IndexType) (p : Parameters) : List ComputeStmt :=
  (source index p).surfaceBody.take 3

def body (index : IndexType) (p : Parameters) : List ComputeStmt :=
  match (source index p).surfaceBody.drop 3 with
  | [.forRange _ _ _ _ statements] => statements
  | _ => []

theorem decomposition (index : IndexType) (p : Parameters) :
    (source index p).surfaceBody=prefixCode index p ++ [.forRange "start_nn" 0 64 1 (body index p)] := by
  cases index <;> rfl

structure Context (p : Parameters) (s : State α) : Prop where
  rows : s.regs .int [1] "offs_nn"=some (fun j : TileIndex [1] => Int.ofNat (s.pids 0*64+j.1.val))
  columns : s.regs .int [p.width] "offs_d"=some (fun j=>(j.1.val : Int))

def row (s : State α) (start : Nat) : Nat := s.pids 0*64+start
def token (p : Parameters) (s : State α) (start : Nat) : Int :=
  (s.mem p.ids (row s start)).read .int
abbrev valid (p : Parameters) (s : State α) (start : Nat) : Prop :=
  row s start < p.n ∧ p.first ≤ token p s start ∧ token p s start < p.last
abbrev active (p : Parameters) (s : State α) (start : Nat) (i : TileIndex [1,p.width]) : Prop :=
  row s start < p.n ∧ i.2.1.val < p.hidden
def offset (p : Parameters) (s : State α) (start : Nat) (i : TileIndex [1,p.width]) : Nat :=
  row s start*p.outputStride+i.2.1.val
def value (zero : α) (p : Parameters) (s : State α) (start d : Nat) : α :=
  if valid p s start then
    (s.mem p.weight ((token p s start-p.first).toNat*p.weightStride+d)).read .real
  else zero

def copied (zero : α) (p : Parameters) (s : State α) (start : Nat) : State α :=
  (TileShape.allIndices [1,p.width]).foldl
    (fun t i=>if active p s start i then
      t.write p.output (offset p s start i) (.mk .real (value zero p s start i.2.1.val)) else t) s

def chunk (rest : Algebra α) (zero : α) (index : IndexType) (p : Parameters)
    (s : State α) (start : Nat) (context : Context p s) :
    {t : State α // run (backend rest zero) (body index p)
      (s.setReg "start_nn" .nat [] (fun _=>start))=some t} := by
  rcases context with ⟨hr,hc⟩
  cases index
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [body,source,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,evalOffsets,evalDeltas,Option.bind,copyBackend,backend,
      ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,bop,ofFloat,toFloat,
      Region.cast,FP.Structural.store,TileShape.dropInsertedIndex,hr,hc]
    try rfl

theorem nonnegative_offset (id first : Int) (stride d : Nat) (h : first ≤ id) :
    ((id-first)*(stride : Int)+(d : Int)).toNat=(id-first).toNat*stride+d := by
  have hn : 0 ≤ id-first := by omega
  rw [Int.toNat_add (Int.mul_nonneg hn (Int.natCast_nonneg _)) (Int.natCast_nonneg _),
    Int.toNat_mul hn (Int.natCast_nonneg _)]
  simp

theorem chunk_memory (rest : Algebra α) (zero : α) (index : IndexType) (p : Parameters)
    (s : State α) (start : Nat) (context : Context p s) :
    (chunk rest zero index p s start context).val.mem=(copied zero p s start).mem := by
  rcases context with ⟨hr,hc⟩
  cases index
  all_goals
    simp (config := {contextual := Bool.true}) [chunk,copied,masked_scatter_memory,
      active,offset,value,valid,token,row,bop,Broadcast.leftIndex,Broadcast.rightIndex,
      Region.cast,Fin.eq_zero,Nat.add_comm,nonnegative_offset]
    have pid_cast : (s.pids 0 : Int) * 64 = ((s.pids 0 * 64 : Nat) : Int) := by
      norm_cast
    simp only [pid_cast, ← Int.natCast_add, ← Int.natCast_mul, Int.ofNat_lt,
      Int.toNat_natCast, Nat.add_comm]

theorem chunk_context (rest : Algebra α) (zero : α) (index : IndexType) (p : Parameters)
    (s : State α) (start : Nat) (context : Context p s) :
    Context p (chunk rest zero index p s start context).val ∧
      (chunk rest zero index p s start context).val.pids=s.pids := by
  rcases context with ⟨hr,hc⟩
  cases index
  all_goals
    refine ⟨⟨?_,?_⟩,?_⟩
    all_goals simp [chunk,masked_scatter_regs,hr,hc,State.masked_scatter_pids]

theorem chunk_frame (rest : Algebra α) (zero : α) (index : IndexType) (p : Parameters)
    (s : State α) (start : Nat) (context : Context p s) (r : RegionName) (o : Nat)
    (miss : r≠p.output ∨ ∀ i,active p s start i → o≠offset p s start i) :
    (chunk rest zero index p s start context).val.mem r o=s.mem r o := by
  rw [chunk_memory]
  unfold copied
  exact State.masked_scatter_frame _ _ _ _ _ r o (miss.imp id (fun h i _ hi=>h i hi)) s

theorem chunk_readback (rest : Algebra α) (zero : α) (index : IndexType) (p : Parameters)
    (s : State α) (start : Nat) (context : Context p s)
    (i : TileIndex [1,p.width]) (hi : active p s start i) :
    (chunk rest zero index p s start context).val.mem p.output (offset p s start i)=
      .mk .real (value zero p s start i.2.1.val) := by
  rw [chunk_memory]
  apply masked_readback_active _ _ _ _ _ i hi
  intro a b _ _ eq
  apply Prod.ext (Subsingleton.elim _ _)
  apply Prod.ext _ (Subsingleton.elim _ _)
  apply Fin.ext
  exact Nat.add_left_cancel eq

def prefixRun (rest : Algebra α) (zero : α) (index : IndexType) (p : Parameters)
    (s : State α) : {t : State α // run (backend rest zero) (prefixCode index p) s=some t} := by
  cases index
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [prefixCode,source,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,numeric,bop,ComputeDType.eraseDType]
    try rfl

theorem prefix_properties (rest : Algebra α) (zero : α) (index : IndexType) (p : Parameters)
    (s : State α) :
    Context p (prefixRun rest zero index p s).val ∧
    (prefixRun rest zero index p s).val.pids=s.pids ∧
    (prefixRun rest zero index p s).val.mem=s.mem := by
  cases index
  all_goals
    refine ⟨⟨?_,?_⟩,?_,?_⟩
    all_goals simp [prefixRun,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals funext i; rfl

structure Layout (p : Parameters) : Prop where
  ids_distinct : p.ids ≠ p.output
  weight_distinct : p.weight ≠ p.output
  width_covers : p.hidden ≤ p.width
  stride_covers : p.hidden ≤ p.outputStride

theorem address_unique (p : Parameters) (layout : Layout p)
    (a b d e : Nat) (hd : d<p.hidden) (he : e<p.hidden)
    (eq : a*p.outputStride+d=b*p.outputStride+e) : a=b ∧ d=e := by
  have h := VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.Reference.row_major_injective
    p.outputStride p.width p.hidden a b ⟨d,lt_of_lt_of_le hd layout.width_covers⟩
    ⟨e,lt_of_lt_of_le he layout.width_covers⟩
    (le_trans (min_le_right _ _) layout.stride_covers) hd he eq
  exact ⟨h.1,congrArg Fin.val h.2⟩

structure Invariant (zero : α) (p : Parameters) (initial : State α)
    (k : Nat) (s : State α) : Prop where
  context : Context p s
  pids : s.pids=initial.pids
  bound : k≤64
  frame : ∀ r o, (r≠p.output ∨ ∀ a,a<k → row initial a<p.n →
    ∀ d,d<p.hidden → o≠row initial a*p.outputStride+d) → s.mem r o=initial.mem r o
  readback : ∀ a,a<k → row initial a<p.n → ∀ d,d<p.hidden →
    s.mem p.output (row initial a*p.outputStride+d)=.mk .real (value zero p initial a d)

theorem invariant_step (rest : Algebra α) (zero : α) (index : IndexType) (p : Parameters)
    (layout : Layout p) (initial s : State α) (k : Nat) (hk : k<64)
    (inv : Invariant zero p initial k s) :
    Invariant zero p initial (k+1) (chunk rest zero index p s k inv.context).val := by
  have rows : ∀ a,row s a=row initial a := fun a=>by simp [row,inv.pids]
  have sameValue : ∀ a d,value zero p s a d=value zero p initial a d := by
    intro a d
    have hi : ∀ o,s.mem p.ids o=initial.mem p.ids o := fun o=>inv.frame _ _ (Or.inl layout.ids_distinct)
    have hw : ∀ o,s.mem p.weight o=initial.mem p.weight o := fun o=>inv.frame _ _ (Or.inl layout.weight_distinct)
    simp only [value,valid,token,rows,hi,hw]
  obtain ⟨context,pids⟩ := chunk_context rest zero index p s k inv.context
  refine ⟨context,pids.trans inv.pids,by omega,?_,?_⟩
  · intro r o miss
    rw [chunk_frame]
    · apply inv.frame
      exact miss.imp id (fun h a ha hr d hd=>h a (by omega) hr d hd)
    · rcases miss with h|h
      · exact Or.inl h
      · right
        intro i hi
        simpa [offset,rows] using h k (by omega) (by simpa [active,rows] using hi.1)
          i.2.1.val hi.2
  · intro a ha hr d hd
    by_cases he : a=k
    · subst a
      let i : TileIndex [1,p.width] := (⟨0,by omega⟩,⟨d,lt_of_lt_of_le hd layout.width_covers⟩,())
      have hi : active p s k i := ⟨by simpa [rows] using hr,hd⟩
      have h := chunk_readback rest zero index p s k inv.context i hi
      simpa [offset,rows,i,sameValue] using h
    · have hak : a<k := by omega
      rw [chunk_frame]
      · exact inv.readback a hak hr d hd
      · right
        intro i hi eq
        have unique := address_unique p layout (row initial a) (row s k) d i.2.1.val hd hi.2 eq
        have equal : a=k := by have := unique.1; simp only [rows,row] at this; omega
        exact he equal

theorem program_correct (rest : Algebra α) (zero : α) (index : IndexType) (p : Parameters)
    (layout : Layout p) (s : State α) :
    ∃ t,FP.Structural.exec (backend rest zero) (source index p) s=some t ∧
      t.pids=s.pids ∧
      (∀ a,a<64 → row s a<p.n → ∀ d,d<p.hidden →
        t.mem p.output (row s a*p.outputStride+d)=.mk .real (value zero p s a d)) ∧
      (∀ r o,(r≠p.output ∨ ∀ a,a<64 → row s a<p.n →
        ∀ d,d<p.hidden → o≠row s a*p.outputStride+d) → t.mem r o=s.mem r o) := by
  obtain ⟨context,pids,mem⟩ := prefix_properties rest zero index p s
  have base : Invariant zero p s 0 (prefixRun rest zero index p s).val := by
    refine ⟨context,pids,by omega,?_,?_⟩
    · intro r o _; exact congrFun (congrFun mem r) o
    · intro a ha; omega
  obtain ⟨last,t,run,hlast,inv⟩ := range_invariant (backend rest zero) (idx:="start_nn")
    (stop:=64) (stride:=1) (body:=body index p) (P:=Invariant zero p s) (by omega)
    (fun k t hk inv=>⟨(chunk rest zero index p t k inv.context).val,
      (chunk rest zero index p t k inv.context).property,
      invariant_step rest zero index p layout s t k hk inv⟩)
    0 (prefixRun rest zero index p s).val base
  have eq : last=64 := by have := inv.bound; omega
  subst last
  refine ⟨t,?_,inv.pids,inv.readback,inv.frame⟩
  change FP.Structural.run (backend rest zero) (source index p).surfaceBody s=some t
  rw [decomposition,run_append,(prefixRun rest zero index p s).property]
  simpa [FP.Structural.run,step] using run

#axiomsClean decomposition
#axiomsClean chunk
#axiomsClean chunk_memory
#axiomsClean chunk_context
#axiomsClean chunk_frame
#axiomsClean chunk_readback
#axiomsClean prefix_properties
#axiomsClean invariant_step
#axiomsClean program_correct
end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.Execution
