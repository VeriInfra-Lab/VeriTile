import bench.repairs.tritonbench_g.matmul_kernel.Proofs.Physical
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.Bounds
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.Real
/- The source launcher selects a.device through the standard CUDA device
context. Formal device/stream observations refer to that effective owning
device; leaving the context restores the ambient caller device. -/
namespace VeriTile.Bench.Repairs.MatmulKernel.Host
inductive Return where | none deriving DecidableEq, Repr
structure Result (α : Type) where
  memory : Memory α
  returned : Return
  outputIdentity : Region
/-- The wrapper has no output allocation and returns Python None. Validated
empty M/N calls do not launch. K=0 with nonempty output still launches. -/
def call (c : Config) (p : Native.Instructions) (m : Physical.Bytes) : Result Physical.Byte :=
  ⟨if c.M=0 ∨ c.N=0 then m else Physical.native c p m,.none,.c⟩
noncomputable def realCall (c : Config) (m : Memory ℝ) : Result ℝ :=
  ⟨if c.M=0 ∨ c.N=0 then m else source c realOps m,.none,.c⟩

theorem memory_native (c : Config) (p : Native.Instructions) (m : Physical.Bytes) :
    (call c p m).memory=Physical.native c p m := by
  unfold call
  dsimp only
  split_ifs with h
  · simp [Physical.native,schedule_empty c h,Physical.byteExecute]
  · rfl
theorem real_memory (c : Config) (m : Memory ℝ) : (realCall c m).memory=source c realOps m := by
  unfold realCall
  dsimp only
  split_ifs with h
  · simp [source_run,run,schedule_empty c h,execute]
  · rfl
theorem returns_none (c : Config) (p : Native.Instructions) (m : Physical.Bytes) :
    (call c p m).returned=.none ∧ (call c p m).outputIdentity=.c := by exact ⟨rfl,rfl⟩
theorem caller_bytes (c : Config) (hc : c.Valid) (p : Native.Instructions) (m : Physical.Bytes)
    (w : Work) (hw : c.Active w) (i : Fin 4) :
    (call c p m).memory .c (4*c.C.addr w.1 w.2+i.val)=
      Physical.encode (Native.reference c p (Physical.load m) w.1 w.2) i := by
  rw [memory_native]
  exact Physical.native_bytes c hc p m w hw i
theorem caller_frame (c : Config) (p : Native.Instructions) (m : Physical.Bytes) (r : Region) (b : Nat)
    (outside : ∀ w,c.Active w → ¬Writes c w r (b/4)) : (call c p m).memory r b=m r b := by
  rw [memory_native]
  exact Physical.native_frame c p m r b outside
theorem caller_mathematical (c : Config) (hc : c.Valid) (m : Memory ℝ) (w : Work) (hw : c.Active w) :
    (realCall c m).memory .c (c.C.addr w.1 w.2)=mathematical c m w.1 w.2 := by
  rw [real_memory]
  exact source_mathematical c hc m w hw
theorem caller_local_transport (c : Config) (actual target : Native.Instructions)
    (h : Native.Obligations actual target) (m : Physical.Bytes) :
    (call c actual m).memory=(call c target m).memory := by
  rw [memory_native,memory_native]
  exact Physical.native_local_transport c actual target h m

theorem empty_no_launch (c : Config) (p : Native.Instructions) (m : Physical.Bytes)
    (h : c.M=0 ∨ c.N=0) : (call c p m).memory=m := by simp [call,h]
theorem zero_k_value (c : Config) (hc : c.Valid) (p : Native.Instructions) (m : Memory Native.Word)
    (h : c.K=0) (w : Work) : value c (Native.wordOps p) m w=p.widen (p.narrow 0) := by
  have ht : c.steps=0 := by
    unfold Config.steps ceil
    rw [h]
    simp only [Nat.zero_add]
    exact Nat.div_eq_of_lt (by have := hc.bk; omega)
  simp [value,ht,Native.wordOps]
#axiomsClean caller_bytes
#axiomsClean caller_frame
#axiomsClean caller_mathematical
#axiomsClean caller_local_transport
#axiomsClean zero_k_value
end VeriTile.Bench.Repairs.MatmulKernel.Host
