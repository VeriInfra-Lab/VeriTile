import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceForwardResult
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.KernelForward

/-! The per-program region environment reads the same actual Tensor backing
and pointer as the admitted input. Launch execution checks every source
program and commits the cells those programs produced to the fresh outputs.
The graph is used only as an independent specification of those cells. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceForwardStorage
open VeriTile.Triton FP.Structural SourceModel
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
open KernelMemory
noncomputable section
open scoped Classical

def initial [Inhabited α] (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Forward M N) (memory : World Store) (r : Nat) : State α where
  mem := fun region i => .mk .real (
    if region="X" then inputValue api neg c.x memory i
    else if region="W" then inputValue api neg c.w memory i
    else if region="Bias" then inputValue api neg c.bias memory i
    else if region="Mean" then api.read (memory (c.buffers.output 1).owner) ((c.buffers.output 1).address i)
    else if region="Rstd" then api.read (memory (c.buffers.output 0).owner) ((c.buffers.output 0).address i)
    else if region="Y" then api.read (memory (c.buffers.output 2).owner) ((c.buffers.output 2).address i)
    else default)
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then r else 0
  numPids := fun ax => if ax=0 then M else 1
  undef := fun dtype _ _ => defaultValue dtype

@[simp] theorem initial_row [Inhabited α] (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Forward M N) (memory : World Store) (r : Nat) :
    SourceForwardGraph.row (initial api neg c memory r) N=
      KernelForward.row api neg c.x memory N r := by
  funext j
  simp [SourceForwardGraph.row,initial,KernelForward.row]

@[simp] theorem initial_weight [Inhabited α] (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Forward M N) (memory : World Store) (r : Nat) :
    SourceForwardGraph.vector (initial api neg c memory r) "W"=inputValue api neg c.w memory := by
  funext j
  simp [SourceForwardGraph.vector,initial]

@[simp] theorem initial_bias [Inhabited α] (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Forward M N) (memory : World Store) (r : Nat) :
    SourceForwardGraph.vector (initial api neg c memory r) "Bias"=inputValue api neg c.bias memory := by
  funext j
  simp [SourceForwardGraph.vector,initial]

theorem input_binding [Inhabited α] (api : Memory.API α Store Byte) (neg : α → α)
    (c : KernelContract.Forward M N) (memory : World Store) (r j : Nat) (hr : r<M) (hj : j<N) :
    SourceForwardGraph.row (initial api neg c memory r) N j=
      logical api neg memory c.x.dense.tensor (r*N+j) ∧
    SourceForwardGraph.row (initial api neg c memory r) N j=
      api.read (memory c.x.dense.owner) (c.x.metadata.dataPtr.toNat+4*(r*N+j)) := by
  rw [initial_row]
  exact KernelForward.row_input_binding api neg c memory r j hr hj

def runRow [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ)
    (api : Memory.API α Store Byte) (neg : α → α) (c : KernelContract.Forward M N)
    (memory : World Store) (r : Nat) : Option (State α) :=
  FP.Structural.exec (engine model)
    (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" N N B eps)
    (initial api neg c memory r)

def resultRow [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ)
    (api : Memory.API α Store Byte) (neg : α → α) (c : KernelContract.Forward M N)
    (memory : World Store) (r : Nat) : State α :=
  (runRow model B eps api neg c memory r).getD (initial api neg c memory r)

/-- Actual source cells, read after interpretation. -/
def sourceValues [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ)
    (api : Memory.API α Store Byte) (neg : α → α) (c : KernelContract.Forward M N)
    (memory : World Store) : ∀ k : Fin 3,Fin (c.buffers.output k).count → α
  | ⟨0,_⟩,i => ((resultRow model B eps api neg c memory i.val).mem "Rstd" i.val).read .real
  | ⟨1,_⟩,i => ((resultRow model B eps api neg c memory i.val).mem "Mean" i.val).read .real
  | ⟨2,_⟩,i => ((resultRow model B eps api neg c memory (i.val/N)).mem "Y" i.val).read .real

def graphValues [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ)
    (api : Memory.API α Store Byte) (neg : α → α) (c : KernelContract.Forward M N)
    (memory : World Store) : ∀ k : Fin 3,Fin (c.buffers.output k).count → α
  | ⟨0,_⟩,i => SourceForwardGraph.inverse model N B eps (KernelForward.row api neg c.x memory N i.val)
  | ⟨1,_⟩,i => SourceForwardGraph.mean model N B (KernelForward.row api neg c.x memory N i.val)
  | ⟨2,_⟩,i => SourceForwardGraph.output model N B eps (KernelForward.row api neg c.x memory N (i.val/N))
      (inputValue api neg c.w memory) (inputValue api neg c.bias memory) (i.val%N)

theorem row_success [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ)
    (api : Memory.API α Store Byte) (neg : α → α) (c : KernelContract.Forward M N)
    (memory : World Store) (hn : 0<N) (hb : N≤B) (r : Nat) :
    ∃ t,runRow model B eps api neg c memory r=some t :=
  SourceForward.source_executes model N N B eps hn hb _

theorem sourceValues_eq [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ)
    (api : Memory.API α Store Byte) (neg : α → α) (c : KernelContract.Forward M N)
    (memory : World Store) (hn : 0<N) (hb : N≤B) :
    sourceValues model B eps api neg c memory=graphValues model B eps api neg c memory := by
  have hr (r : Nat) := SourceForwardResult.executes model N N B eps hn hb (initial api neg c memory r)
  funext k i
  fin_cases k
  · obtain ⟨t,ht,_,_,_,_,hi,_⟩ := hr i.val
    simp only [sourceValues,graphValues,resultRow,runRow,ht,Option.getD_some]
    simpa [initial] using congrArg (fun cell : Cell α => cell.read .real) hi
  · obtain ⟨t,ht,_,_,_,hm,_,_⟩ := hr i.val
    simp only [sourceValues,graphValues,resultRow,runRow,ht,Option.getD_some]
    simpa [initial] using congrArg (fun cell : Cell α => cell.read .real) hm
  · obtain ⟨t,ht,_,_,hy,_,_,_⟩ := hr (i.val/N)
    have ho := congrArg (fun cell : Cell α => cell.read .real) (hy (i.val%N) (Nat.mod_lt _ hn))
    simp only [sourceValues,graphValues,resultRow,runRow,ht,Option.getD_some]
    have index : i.val/N*N+i.val%N=i.val := by simpa [Nat.mul_comm] using Nat.div_add_mod i.val N
    simp only [initial_row,initial_weight,initial_bias,Cell.read_same] at ho
    simpa [initial,index] using ho

/-- Dispatch executes no program for either zero dimension. For a nonempty
launch, every interpreted source result is checked before any commit. -/
def exec [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ)
    (api : Memory.API α Store Byte) (neg : α → α) (c : KernelContract.Forward M N)
    (memory : World Store) : Option (World Store) :=
  if KernelContract.launched M N then
    if ∀ r,r<M → (runRow model B eps api neg c memory r).isSome then
      some (KernelMemory.run c.buffers api (sourceValues model B eps api neg c memory) memory)
    else none
  else some memory

def final [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ)
    (api : Memory.API α Store Byte) (neg : α → α) (c : KernelContract.Forward M N)
    (memory : World Store) : World Store :=
  if KernelContract.launched M N then
    KernelMemory.run c.buffers api (graphValues model B eps api neg c memory) memory
  else memory

def candidateRunRow [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ)
    (api : Memory.API α Store Byte) (neg : α → α) (c : KernelContract.Forward M N)
    (memory : World Store) (r : Nat) : Option (State α) :=
  FP.Structural.exec (engine model) (SourceCandidate.forward N N B eps)
    (initial api neg c memory r)

theorem candidate_row_eq [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ)
    (api : Memory.API α Store Byte) (neg : α → α) (c : KernelContract.Forward M N)
    (memory : World Store) (hn : 0<N) (hb : N≤B) (r : Nat) :
    candidateRunRow model B eps api neg c memory r=runRow model B eps api neg c memory r :=
  SourceCandidate.forward_equiv (engine model) N N B eps hn hb _

theorem executes [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ)
    (api : Memory.API α Store Byte) (neg : α → α) (c : KernelContract.Forward M N)
    (memory : World Store) (hb : N≤B) :
    ∃ result,exec model B eps api neg c memory=some result ∧
      result=final model B eps api neg c memory ∧
      (KernelContract.launched M N=Bool.true → ∀ k i,
        logical api neg result (c.buffers.output k).tensor i.val=graphValues model B eps api neg c memory k i) ∧
      (KernelContract.launched M N=Bool.false → result=memory) ∧
      (∀ key,KernelMemory.Outside c.buffers key → observe api result key=observe api memory key) ∧
      (∀ key,caller [c.x.dense.tensor,c.w.dense.tensor,c.bias.dense.tensor] key →
        observe api result key=observe api memory key) := by
  by_cases hl : KernelContract.launched M N=Bool.true
  · have hn := ((KernelContract.launched_iff M N).mp hl).2
    have hs : ∀ r,r<M → (runRow model B eps api neg c memory r).isSome := by
      intro r _
      obtain ⟨t,ht⟩ := row_success model B eps api neg c memory hn hb r
      simp [ht]
    have hv := sourceValues_eq model B eps api neg c memory hn hb
    refine ⟨KernelMemory.run c.buffers api (graphValues model B eps api neg c memory) memory,?_,?_,?_,?_,?_,?_⟩
    · simp only [exec,hl,if_true,hv]
      rw [if_pos hs]
    · simp only [final,hl,if_true]
    · intro _ k i
      exact KernelMemory.run_read c.buffers api neg _ memory k i
    · intro hf; simp [hl] at hf
    · exact KernelMemory.run_outside c.buffers api _ memory
    · exact KernelMemory.run_frame c.buffers api _ memory
  · have hf := Bool.eq_false_iff.mpr hl
    exact ⟨memory,by simp [exec,hf],by simp [final,hf],fun h => False.elim (hl h),
      fun _ => rfl,fun _ _ => rfl,fun _ _ => rfl⟩

#axiomsClean input_binding
#axiomsClean executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceForwardStorage
