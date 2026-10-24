import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelIndex
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativePlan

namespace VeriTile.Bench.Repairs.LayerNormLiger.KernelContract
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (withSum)
noncomputable section

/-- Raw FP32 input cells enter the graph as words. The graph retains source
casts, epsilon/count literals, rsqrt, every reduction leaf and selected FMA. -/
def words (memory : Nat → FP.Value .fp32) : Nat → NativePlan.Expr :=
  fun a => .word (memory a)

def evaluateMemory (P : NativePlan.Primitives) (memory : Nat → NativePlan.Expr) : Nat → FP.Value .fp32 :=
  fun a => (memory a).eval P

/-- Independent ordered scalar graph, built from actual Tensor logical reads.
Forward.reference is a separately defined scalar expression and memory update;
it neither contains nor executes the source AST. -/
def forwardGraph (P : NativePlan.Primitives) (contraction : NativePlan.Contraction) (trees : NativePlan.Trees)
    (c : ForwardCall) (pointers : RegionName → Nat) (memory : Nat → FP.Value .fp32) : Nat → FP.Value .fp32 :=
  evaluateMemory P (commit pointers c.writes (words memory)
    (Forward.reference (NativePlan.algebra contraction trees) (NativePlan.sums contraction trees) c.parameters
      (logicalEntry c.inputs (words memory) (entry pointers (words memory) c.pid))))

/-- Independent ordered backward recurrence. Its loop processes exactly the
source row interval and includes both partial accumulators and all DX stores. -/
def backwardGraph (P : NativePlan.Primitives) (contraction : NativePlan.Contraction) (trees : NativePlan.Trees)
    (c : BackwardCall) (pointers : RegionName → Nat) (memory : Nat → FP.Value .fp32) : Nat → FP.Value .fp32 :=
  evaluateMemory P (commit pointers c.writes (words memory)
    (Backward.reference (NativePlan.algebra contraction trees) (NativePlan.sums contraction trees) c.parameters
      (logicalEntry c.inputs (words memory) (entry pointers (words memory) c.pid))))

/-- Tensor observations of all forward outputs at the actual source addresses. -/
def ForwardNative (c : ForwardCall) (pointers : RegionName → Nat)
    (graph final : Nat → FP.Value .fp32) : Prop :=
  c.mean.logical final c.pid = graph (address pointers (c.mean.region,c.pid)) ∧
  c.rstd.logical final c.pid = graph (address pointers (c.rstd.region,c.pid)) ∧
  ∀ j, j < c.columns → c.y.logical final (c.pid*c.columns+j) =
    graph (address pointers (c.y.region,c.pid*c.columns+j))

/-- Tensor observations of both partial rows and every DX row this program owns. -/
def BackwardNative (c : BackwardCall) (pointers : RegionName → Nat)
    (graph final : Nat → FP.Value .fp32) : Prop :=
  (∀ j, j < c.columns → c.dw.logical final (c.pid*c.columns+j) =
    graph (address pointers (c.dw.region,c.pid*c.columns+j))) ∧
  (∀ j, j < c.columns → c.db.logical final (c.pid*c.columns+j) =
    graph (address pointers (c.db.region,c.pid*c.columns+j))) ∧
  ∀ row, c.first ≤ row → row < c.last → ∀ j, j < c.columns →
    c.dx.logical final (row*c.columns+j) = graph (address pointers (c.dx.region,row*c.columns+j))

 theorem forward_native_outputs [Inhabited (FP.Value .fp32)] (c : ForwardCall) (h : ForwardDomain c pointers)
    (graph final : Nat → FP.Value .fp32) (equal : final = graph) : ForwardNative c pointers graph final := by
  subst final
  refine ⟨?_,?_,?_⟩
  · exact logical_read h.mean (h.bound _ (by simp [ForwardCall.buffers])) graph _ (vector_bound h.mean _ h.pid_bound)
  · exact logical_read h.rstd (h.bound _ (by simp [ForwardCall.buffers])) graph _ (vector_bound h.rstd _ h.pid_bound)
  · intro j hj
    exact logical_read h.y (h.bound _ (by simp [ForwardCall.buffers])) graph _ (input_bound h.y _ _ h.pid_bound hj)

 theorem backward_native_outputs [Inhabited (FP.Value .fp32)] (c : BackwardCall) (h : BackwardDomain c pointers)
    (graph final : Nat → FP.Value .fp32) (equal : final = graph) : BackwardNative c pointers graph final := by
  subst final
  refine ⟨?_,?_,?_⟩
  · intro j hj
    exact logical_read h.dw (h.bound _ (by simp [BackwardCall.buffers])) graph _ (input_bound h.dw _ _ h.pid_bound hj)
  · intro j hj
    exact logical_read h.db (h.bound _ (by simp [BackwardCall.buffers])) graph _ (input_bound h.db _ _ h.pid_bound hj)
  · intro row _ hr j hj
    exact logical_read h.dx (h.bound _ (by simp [BackwardCall.buffers])) graph _
      (input_bound h.dx _ _ (lt_of_lt_of_le hr (min_le_right _ _)) hj)

 theorem fp32_logical (b : Buffer) (h : Valid b shape) (memory : Nat → FP.Value .fp32) (i : Nat) :
    b.logical memory i = Torch.FP32.logicalValue memory b.tensor h.tensor i := by
  simp [Buffer.logical,Torch.FP32.logicalValue,h.noNeg]

 theorem forward_fp (P : NativePlan.Primitives) (contraction : NativePlan.Contraction) (trees : NativePlan.Trees)
    (c : ForwardCall) (h : ForwardDomain c pointers) (memory : Nat → FP.Value .fp32) :
    ∃ t, exec (withSum (NativePlan.algebra contraction trees) (NativePlan.sums contraction trees))
        (Forward.kernel c.parameters) (entry pointers (words memory) c.pid) = some t ∧
      evaluateMemory P (commit pointers c.writes (words memory) t) = forwardGraph P contraction trees c pointers memory ∧
      (∀ l, ¬ c.writes l → t.mem l.1 l.2 = (entry pointers (words memory) c.pid).mem l.1 l.2) ∧
      (∀ a, (∀ l, c.writes l → address pointers l ≠ a) →
        evaluateMemory P (commit pointers c.writes (words memory) t) a = memory a) := by
  obtain ⟨t,ran,mem⟩ := Forward.complete_run (NativePlan.algebra contraction trees) (NativePlan.sums contraction trees)
    c.parameters (entry pointers (words memory) c.pid)
  refine ⟨t,ran,?_,?_,?_⟩
  · unfold forwardGraph
    rw [forward_logicalEntry c h]
    rw [commit_mem_congr _ _ _ _ _ mem]
  · intro l hl
    rw [mem]
    exact Forward.reference_frame _ _ _ _ _ _ (forward_outside c h _ l hl)
  · intro a ha
    unfold evaluateMemory
    rw [commit_frame _ _ _ _ a ha]
    rfl

 theorem backward_fp (P : NativePlan.Primitives) (contraction : NativePlan.Contraction) (trees : NativePlan.Trees)
    (c : BackwardCall) (h : BackwardDomain c pointers) (memory : Nat → FP.Value .fp32) :
    ∃ t, exec (withSum (NativePlan.algebra contraction trees) (NativePlan.sums contraction trees))
        (Backward.kernel c.parameters) (entry pointers (words memory) c.pid) = some t ∧
      evaluateMemory P (commit pointers c.writes (words memory) t) = backwardGraph P contraction trees c pointers memory ∧
      (∀ l, ¬ c.writes l → t.mem l.1 l.2 = (entry pointers (words memory) c.pid).mem l.1 l.2) ∧
      (∀ a, (∀ l, c.writes l → address pointers l ≠ a) →
        evaluateMemory P (commit pointers c.writes (words memory) t) a = memory a) := by
  obtain ⟨t,ran,mem⟩ := Backward.complete_run (NativePlan.algebra contraction trees) (NativePlan.sums contraction trees)
    c.parameters (entry pointers (words memory) c.pid)
  refine ⟨t,ran,?_,?_,?_⟩
  · unfold backwardGraph
    rw [backward_logicalEntry c h]
    rw [commit_mem_congr _ _ _ _ _ mem]
  · intro l hl
    rw [mem]
    exact Backward.reference_frame _ _ _ _ _ _ (backward_outside c h _ l hl)
  · intro a ha
    unfold evaluateMemory
    rw [commit_frame _ _ _ _ a ha]
    rfl

 theorem backward_fp_input_frame (P : NativePlan.Primitives) (c : BackwardCall)
    (h : BackwardDomain c pointers) (memory : Nat → FP.Value .fp32) (t : State NativePlan.Expr)
    (b : Buffer) (hb : b ∈ c.inputs) (i : Nat) (hi : i < b.size) :
    b.logical (evaluateMemory P (commit pointers c.writes (words memory) t)) i = b.logical memory i := by
  have same := backward_input_frame c h (words memory) t b hb i hi
  unfold Buffer.logical at same ⊢
  unfold evaluateMemory
  rw [same]
  rfl

#guard_msgs (drop info) in
#specNonCircular forwardGraph avoiding [Forward.kernel,Forward.source,FP.Structural.exec]
#guard_msgs (drop info) in
#specNonCircular backwardGraph avoiding [Backward.kernel,Backward.source,FP.Structural.exec]
end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.LayerNormLiger.KernelContract
