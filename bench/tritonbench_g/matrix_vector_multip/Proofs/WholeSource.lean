import bench.tritonbench_g.matrix_vector_multip.Proofs.TypedIO
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeSource
open VeriTile.Triton FP.Structural Typed
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

abbrev Acc (α : Type) (p : Parameters) := Values α .real [p.blockRows,p.blockColumns]
def source (p : Parameters) : ComputeKernel := Typed.kernel .fp32 p
def codePrefix (p : Parameters) : List ComputeStmt := (source p).surfaceBody.take 7
def body (p : Parameters) : List ComputeStmt :=
  match ((source p).surfaceBody[7]? : Option ComputeStmt) with
  | some (.forRange _ _ _ _ code) => code
  | _ => []
def suffix (p : Parameters) : List ComputeStmt := (source p).surfaceBody.drop 8

theorem complete_source (p : Parameters) :
    (source p).surfaceBody = codePrefix p ++
      [.forRange "m" 0 p.columns p.blockColumns (body p)] ++ suffix p := by rfl

def row (p : Parameters) (pid i : Nat) := pid*p.blockRows+i
def aOffset (p : Parameters) (pid off i j : Nat) :=
  row p pid i*p.strideRow+j*p.strideColumn+off*p.strideColumn
def bOffset (p : Parameters) (off j : Nat) := j*p.strideVector+off*p.strideVector

def prefixState {α : Type} (A : Algebra α) (p : Parameters) (pid : Nat) (s : State α) : State α :=
  let s := s.setReg "pid" .nat [] (fun _ => pid)
  let s := s.setReg "offset_n" .nat [p.blockRows,1] (fun i => row p pid i.1.val)
  let s := s.setReg "offset_m" .nat [1,p.blockColumns] (fun i => i.2.1.val)
  let s := s.setReg "n_mask" .bool [p.blockRows,1] (fun i => decide (row p pid i.1.val<p.rows))
  let s := s.setReg "A_ptrs" .ptr [p.blockRows,p.blockColumns]
    (fun i => (p.a,aOffset p pid 0 i.1.val i.2.1.val))
  let s := s.setReg "B_ptrs" .ptr [1,p.blockColumns] (fun i => (p.b,bOffset p 0 i.2.1.val))
  s.setReg "acc" .real [p.blockRows,p.blockColumns] (fun _ => A.literal (some .fp32) .real 0)

theorem prefix_runs {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters) (s : State α) :
    run A (codePrefix p) s=some (prefixState A p (s.pids 0) s) := by
  rcases p with ⟨a,b,o,n,m,ar,ac,bc,oc,bn,bm⟩
  simp [codePrefix,source,Typed.kernel,Typed.source,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,natLt,bop,
    Region.cast,prefixState,row,aOffset,bOffset,Option.bind]
  congr 10 <;> (first | rfl | (funext i; simp [bop,Broadcast.leftIndex,Broadcast.rightIndex]))

def Context {α : Type} (p : Parameters) (pid off : Nat)
    (memory : RegionName → Nat → Cell α) (acc : Acc α p) (s : State α) : Prop :=
  s.regs .nat [p.blockRows,1] "offset_n"=some (fun i => row p pid i.1.val) ∧
  s.regs .nat [1,p.blockColumns] "offset_m"=some (fun i => i.2.1.val) ∧
  s.regs .bool [p.blockRows,1] "n_mask"=some (fun i => decide (row p pid i.1.val<p.rows)) ∧
  s.regs .ptr [p.blockRows,p.blockColumns] "A_ptrs"=
    some (fun i => (p.a,aOffset p pid off i.1.val i.2.1.val)) ∧
  s.regs .ptr [1,p.blockColumns] "B_ptrs"=some (fun i => (p.b,bOffset p off i.2.1.val)) ∧
  s.regs .real [p.blockRows,p.blockColumns] "acc"=some acc ∧ s.mem=memory

def load {α : Type} [Inhabited α] (A : Algebra α) (memory : RegionName → Nat → Cell α)
    (region : RegionName) (offset : Nat) (active : Prop) [Decidable active] : α :=
  A.cast (some .fp32) .fp32 .real (if active then (memory region offset).read .fp32
    else A.cast (some .fp32) .real .fp32 (A.literal (some .fp32) .real 0))

def aChunk {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters) (pid off : Nat)
    (memory : RegionName → Nat → Cell α) : Acc α p :=
  fun i => load A memory p.a (aOffset p pid off i.1.val i.2.1.val)
    (row p pid i.1.val<p.rows ∧ off+i.2.1.val<p.columns)
def bChunk {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters) (off : Nat)
    (memory : RegionName → Nat → Cell α) : Values α .real [1,p.blockColumns] :=
  fun i => load A memory p.b (bOffset p off i.2.1.val) (off+i.2.1.val<p.columns)
def nextAcc {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters) (pid off : Nat)
    (memory : RegionName → Nat → Cell α) (acc : Acc α p) : Acc α p :=
  fun i => A.binary (some .fp32) .real .add (acc i)
    (A.binary (some .fp32) .real .mul (aChunk A p pid off memory i)
      (bChunk A p off memory (⟨0,by omega⟩,i.2.1,PUnit.unit)))

def bodyState {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters) (pid off : Nat)
    (memory : RegionName → Nat → Cell α) (acc : Acc α p) (s : State α) : State α :=
  let s := s.setReg "m" .nat [] (fun _ => off)
  let s := s.setReg "m_mask" .bool [1,p.blockColumns] (fun i => decide (off+i.2.1.val<p.columns))
  let s := s.setReg "a" .real [p.blockRows,p.blockColumns] (aChunk A p pid off memory)
  let s := s.setReg "b" .real [1,p.blockColumns] (bChunk A p off memory)
  let s := s.setReg "acc" .real [p.blockRows,p.blockColumns] (nextAcc A p pid off memory acc)
  let s := s.setReg "A_ptrs" .ptr [p.blockRows,p.blockColumns]
    (fun i => (p.a,aOffset p pid (off+p.blockColumns) i.1.val i.2.1.val))
  s.setReg "B_ptrs" .ptr [1,p.blockColumns]
    (fun i => (p.b,bOffset p (off+p.blockColumns) i.2.1.val))

theorem body_runs {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters)
    (pid off : Nat) (memory : RegionName → Nat → Cell α) (acc : Acc α p) (s : State α)
    (ctx : Context p pid off memory acc s) :
    run A (body p) (s.setReg "m" .nat [] (fun _ => off))=
      some (bodyState A p pid off memory acc s) := by
  rcases ctx with ⟨hn,hm,hmask,ha,hb,hacc,hmem⟩
  simp [body,source,Typed.kernel,Typed.source,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,natLt,bop,
    Region.cast,bodyState,aChunk,bChunk,nextAcc,load,Option.bind,hn,hm,hmask,ha,hb,hacc,hmem,
    ofFloat,toFloat,ComputeDType.eraseDType]
  congr 12 <;> (first | rfl | (funext ⟨i,j,k⟩; cases k; norm_num [aOffset,bOffset,row,nextAcc,aChunk,bChunk,load,bop,Broadcast.leftIndex,Broadcast.rightIndex]; try ring))

theorem body_context {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters)
    (pid off : Nat) (memory : RegionName → Nat → Cell α) (acc : Acc α p) (s : State α)
    (ctx : Context p pid off memory acc s) :
    Context p pid (off+p.blockColumns) memory (nextAcc A p pid off memory acc)
      (bodyState A p pid off memory acc s) := by
  rcases ctx with ⟨hn,hm,hmask,ha,hb,hacc,hmem⟩
  simp [Context,bodyState,hn,hm,hmask,hmem,State.setReg]

theorem prefix_context {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters) (s : State α) :
    Context p (s.pids 0) 0 s.mem (fun _ => A.literal (some .fp32) .real 0)
      (prefixState A p (s.pids 0) s) := by simp [Context,prefixState]

/-- Complete ordered source recurrence, including all tail-mask operations.
No scalar arithmetic or native contraction law is used here. -/
def accumulated {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters) (pid : Nat)
    (memory : RegionName → Nat → Cell α) (off : Nat) (acc : Acc α p) : Acc α p :=
  if h : off<p.columns ∧ 0<p.blockColumns then
    accumulated A p pid memory (off+p.blockColumns) (nextAcc A p pid off memory acc)
  else acc
termination_by p.columns-off

theorem range_runs {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters)
    (width : 0<p.blockColumns) (pid off : Nat) (memory : RegionName → Nat → Cell α)
    (acc : Acc α p) (s : State α) (ctx : Context p pid off memory acc s) :
    ∃ finalOff t, range A "m" off p.columns p.blockColumns (body p) s=some t ∧
      Context p pid finalOff memory (accumulated A p pid memory off acc) t := by
  by_cases h : off<p.columns
  · have cb := body_context A p pid off memory acc s ctx
    obtain ⟨last,t,ht,hc⟩ := range_runs A p width pid (off+p.blockColumns) memory
      (nextAcc A p pid off memory acc) _ cb
    refine ⟨last,t,?_,?_⟩
    · rw [range]
      simp only [Nat.ne_of_gt width,if_false,h,if_true,body_runs A p pid off memory acc s ctx,Option.bind_some]
      exact ht
    · rw [accumulated]; simpa only [h,width,and_self,if_true] using hc
  · refine ⟨off,s,?_,?_⟩
    · rw [range]; simp [h]
    · rw [accumulated]; simpa only [h,false_and,if_false] using ctx
termination_by p.columns-off

def outputValues {α : Type} (A : Algebra α) (p : Parameters) (acc : Acc α p) :
    Values α .real [p.blockRows] :=
  A.reduceSum (some .fp32) ⟨1,by simp⟩ Bool.false acc

def storedValue {α : Type} (A : Algebra α) (p : Parameters) (acc : Acc α p)
    (i : Fin p.blockRows) : α :=
  A.cast none .real .fp32 (outputValues A p acc (i,PUnit.unit))

def suffixState {α : Type} (A : Algebra α) (p : Parameters) (pid : Nat)
    (acc : Acc α p) (s : State α) : State α :=
  let s := s.setReg "acc" .real [p.blockRows] (outputValues A p acc)
  let s := s.setReg "C_ptrs" .ptr [p.blockRows,1]
    (fun i => (p.out,row p pid i.1.val*p.strideOut))
  (TileShape.allIndices [p.blockRows,1]).foldl (fun t i =>
    if row p pid i.1.val<p.rows then t.write p.out (row p pid i.1.val*p.strideOut)
      (.mk .fp32 (storedValue A p acc i.1)) else t) s

theorem suffix_runs {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters)
    (pid off : Nat) (memory : RegionName → Nat → Cell α) (acc : Acc α p) (s : State α)
    (ctx : Context p pid off memory acc s) :
    run A (suffix p) s=some (suffixState A p pid acc s) := by
  rcases ctx with ⟨hn,hm,hmask,ha,hb,hacc,hmem⟩
  simp [suffix,source,Typed.kernel,Typed.source,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,natLt,bop,store,
    Region.cast,suffixState,outputValues,storedValue,Option.bind,hn,hmask,hacc,
    ofFloat,toFloat,ComputeDType.eraseDType]
  congr 8 <;> (first | rfl | (funext i; simp [bop,Broadcast.leftIndex,Broadcast.rightIndex]))

theorem row_injective (p : Parameters) (pid : Nat)
    (stride : p.strideOut=1) :
    Function.Injective (fun i : TileIndex [p.blockRows,1] => row p pid i.1.val*p.strideOut) := by
  rintro ⟨a,b,u⟩ ⟨c,d,v⟩ he
  have hac : a=c := Fin.ext (by simpa [row,stride] using he)
  subst c
  have hbd : b=d := Subsingleton.elim _ _
  subst d
  cases u; cases v; rfl

theorem suffix_value {α : Type} (A : Algebra α) (p : Parameters) (pid : Nat)
    (stride : p.strideOut=1) (acc : Acc α p) (s : State α)
    (i : Fin p.blockRows) (hi : row p pid i.val<p.rows) :
    (suffixState A p pid acc s).mem p.out (row p pid i.val*p.strideOut)=
      .mk .fp32 (storedValue A p acc i) := by
  apply State.masked_scatter_readback p.out
    (fun i : TileIndex [p.blockRows,1] => row p pid i.1.val*p.strideOut)
    (fun i => Cell.mk .fp32 (storedValue A p acc i.1))
    (fun i => row p pid i.1.val<p.rows) _ (i,⟨0,by omega⟩,PUnit.unit) hi (row_injective p pid stride)

theorem suffix_frame {α : Type} (A : Algebra α) (p : Parameters) (pid : Nat)
    (acc : Acc α p) (s : State α) (r : RegionName) (o : Nat)
    (outside : r≠p.out ∨ ∀ i : Fin p.blockRows, row p pid i.val<p.rows →
      o≠row p pid i.val*p.strideOut) :
    (suffixState A p pid acc s).mem r o=s.mem r o := by
  apply State.masked_scatter_frame
  rcases outside with hr | ho
  · exact Or.inl hr
  · exact Or.inr fun i _ hi => ho i.1 hi

def tileValue {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters) (pid : Nat)
    (memory : RegionName → Nat → Cell α) (i : Fin p.blockRows) : α :=
  storedValue A p (accumulated A p pid memory 0 (fun _ => A.literal (some .fp32) .real 0)) i

/-- Every original source statement runs, all active output words have the
explicit ordered source value, and every other region/address is preserved. -/
theorem tile_result {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters)
    (width : 0<p.blockColumns) (stride : p.strideOut=1) (s : State α) :
    ∃ final, exec A (source p) s=some final ∧
      (∀ i : Fin p.blockRows, row p (s.pids 0) i.val<p.rows →
        final.mem p.out (row p (s.pids 0) i.val)=.mk .fp32 (tileValue A p (s.pids 0) s.mem i)) ∧
      (∀ r o, (r≠p.out ∨ ∀ i : Fin p.blockRows,
        row p (s.pids 0) i.val<p.rows → o≠row p (s.pids 0) i.val) → final.mem r o=s.mem r o) := by
  obtain ⟨last,middle,loopRan,ctx⟩ := range_runs A p width (s.pids 0) 0 s.mem
    (fun _ => A.literal (some .fp32) .real 0) _ (prefix_context A p s)
  let acc := accumulated A p (s.pids 0) s.mem 0 (fun _ => A.literal (some .fp32) .real 0)
  refine ⟨suffixState A p (s.pids 0) acc middle,?_,?_,?_⟩
  · change run A (source p).surfaceBody s=_
    rw [complete_source,run_append,run_append,prefix_runs]
    simp only [Option.bind_some,List.singleton_append,run,step,loopRan]
    exact suffix_runs A p (s.pids 0) last s.mem acc middle ctx
  · intro i hi
    simpa only [stride,Nat.mul_one] using suffix_value A p (s.pids 0) stride acc middle i hi
  · intro r o ho
    have hm : middle.mem=s.mem := ctx.2.2.2.2.2.2
    rw [← hm]
    apply suffix_frame
    simpa only [stride,Nat.mul_one] using ho

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeSource
