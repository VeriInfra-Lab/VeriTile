import bench.tritonbench_g.triton_attention.Proofs.BackwardAccessGeometry
import bench.tritonbench_g.triton_attention.Proofs.FPBackwardFinish
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.MemoryAgreement
open VeriTile Triton FP.Structural Source
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 2000000

/-- A proof relation for transporting finite inputs. It constrains neither
the initial memory nor the public input domain. -/
def Agree (limit : Nat) (a b : Memory α) : Prop :=
  ∀ r o,o<limit → a r o=b r o

theorem write (limit : Nat) (a b : Memory α) (h : Agree limit a b)
    (region : RegionName) (off : Nat) (value : Cell α) :
    Agree limit (writeMemory a region off value) (writeMemory b region off value) := by
  intro r o ho
  simp only [writeMemory]
  split_ifs
  · rfl
  · exact h r o ho

theorem scatter {ι : Type} (limit : Nat) (a b : Memory α) (h : Agree limit a b)
    (region : RegionName) (off : ι→Nat) (value : ι→Cell α) (work : List ι) :
    Agree limit (work.foldl (fun m i=>writeMemory m region (off i) (value i)) a)
      (work.foldl (fun m i=>writeMemory m region (off i) (value i)) b) := by
  induction work generalizing a b with
  | nil => exact h
  | cons i work ih => exact ih _ _ (write limit a b h region (off i) (value i))

theorem maskedScatter {ι : Type} (limit : Nat) (a b : Memory α) (h : Agree limit a b)
    (region : RegionName) (off : ι→Nat) (value : ι→Cell α)
    (mask : ι→Prop) [DecidablePred mask] (work : List ι) :
    Agree limit (work.foldl (fun m i=>if mask i then writeMemory m region (off i) (value i) else m) a)
      (work.foldl (fun m i=>if mask i then writeMemory m region (off i) (value i) else m) b) := by
  induction work generalizing a b with
  | nil => exact h
  | cons i work ih =>
    simp only [List.foldl_cons]
    by_cases active : mask i
    · simp only [active,ite_true]
      exact ih _ _ (write limit a b h region (off i) (value i))
    · simp only [active,ite_false]
      exact ih _ _ h

theorem mapped {β : Type} (f : α→β) (limit : Nat) (a b : Memory α)
    (h : Agree limit a b) :
    Agree limit (fun r o=>mapCell f (a r o)) (fun r o=>mapCell f (b r o)) :=
  fun r o ho=>congrArg (mapCell f) (h r o ho)

theorem scatterState {ι : Type} (limit : Nat) (a b : State α) (h : Agree limit a.mem b.mem)
    (region : RegionName) (off : ι→Nat) (value : ι→Cell α) (work : List ι) :
    Agree limit (work.foldl (fun s i=>s.write region (off i) (value i)) a).mem
      (work.foldl (fun s i=>s.write region (off i) (value i)) b).mem := by
  rw [scatterMemoryReference,scatterMemoryReference]
  exact scatter limit a.mem b.mem h region off value work

theorem maskedScatterState {ι : Type} (limit : Nat) (a b : State α) (h : Agree limit a.mem b.mem)
    (region : RegionName) (off : ι→Nat) (value : ι→Cell α)
    (mask : ι→Prop) [DecidablePred mask] (work : List ι) :
    Agree limit (work.foldl (fun s i=>if mask i then s.write region (off i) (value i) else s) a).mem
      (work.foldl (fun s i=>if mask i then s.write region (off i) (value i) else s) b).mem := by
  rw [maskedScatterMemoryReference,maskedScatterMemoryReference]
  exact maskedScatter limit a.mem b.mem h region off value mask work

theorem row_capacity (c : Shape) : c.groups*c.rows≤c.elements := by
  change c.groups*c.rows≤c.groups*c.rows*c.dim
  exact Nat.le_mul_of_pos_right _ c.dim_pos

theorem forward_query (c : Shape) (scale : ℝ) (pid : Grid.Program)
    (legal : Grid.Legal (Grid.forwardGrid c) pid) (a b : Memory α) (h : Agree c.elements a b) :
    forwardQ (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid a)=
      forwardQ (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid b) := by
  funext i
  apply congrArg (fun cell : Cell α=>cell.read .fp32)
  exact h "Q" _ (SourceAccess.forward_query_bound c scale (Grid.entry (Grid.forwardGrid c) pid a) legal i)

theorem forward_kv (c : Shape) (scale : ℝ) (pid : Grid.Program)
    (legal : Grid.Legal (Grid.forwardGrid c) pid) (a b : Memory α) (h : Agree c.elements a b)
    (region : RegionName) (block : Nat)
    (live : block<forwardBlocks (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid a)) :
    forwardKV (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid a) region block=
      forwardKV (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid b) region block := by
  funext i
  unfold forwardKV
  have pointer : forwardKVPointer (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid a) region block=
      forwardKVPointer (forwardParameters c scale) (Grid.entry (Grid.forwardGrid c) pid b) region block := rfl
  rw [pointer]
  split_ifs
  · apply congrArg (fun cell : Cell α=>cell.read .fp32)
    exact h region _ (SourceAccess.forward_kv_bound c scale (Grid.entry (Grid.forwardGrid c) pid a) legal block live i)
  · rfl

theorem backward_read (c : Shape) (scale : ℝ) (pid : Grid.Program)
    (legal : Grid.Legal (Grid.backwardGrid c) pid) (initial : Memory α)
    (a b : Memory α) (h : Agree c.elements a b) (region : RegionName)
    (block : Nat) (live : block<c.blocks) :
    backwardRead (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid initial) a region block=
      backwardRead (backwardParameters c scale) (Grid.entry (Grid.backwardGrid c) pid initial) b region block := by
  funext i
  unfold backwardRead
  split_ifs
  · apply congrArg (fun cell : Cell α=>cell.read .fp32)
    exact h region _ (SourceAccess.backward_element_bound c scale
      (Grid.entry (Grid.backwardGrid c) pid initial) legal.1 block live i)
  · rfl

#axiomsClean forward_query
#axiomsClean forward_kv
#axiomsClean backward_read
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.MemoryAgreement
