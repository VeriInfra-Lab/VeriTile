import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.Lowering
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.Values

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Native
open VeriTile Triton FP.Structural
open Source
open VeriTile.Bench.Repairs.AttentionLlama NativeSchedule
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Input observations only: the source cells contain the caller's FP16 words
or signed-byte encodings at its actual strided coordinates. -/
structure Binding (c : Config) (s : FP.Structural.State (Term α)) (r : Fin 64)
    (input : Target.Inputs α c.D) : Prop where
  query : ∀ d, (s.mem c.Q (base c s c.sqz c.sqh+rows s r*c.sqm+d.val*c.sqk)).read .fp16=
    .leaf (input.query d)
  halfKey : c.fp8=Bool.false → ∀ k d, k<c.N →
    (s.mem c.K (base c s c.skz c.skh+k*c.skn+d.val*c.skk)).read .fp16=.leaf (input.halfKey k d)
  halfValue : c.fp8=Bool.false → ∀ k d, k<c.N →
    (s.mem c.V (base c s c.svz c.svh+k*c.svk+d.val*c.svn)).read .fp16=.leaf (input.halfValue k d)
  byteKey : c.fp8=Bool.true → ∀ k d, k<c.N → UInt8.ofFin (Float8E5Bits.ofInt
    ((s.mem c.K (base c s c.skz c.skh+k*c.skn+d.val*c.skk)).read .int))=input.byteKey k d
  byteValue : c.fp8=Bool.true → ∀ k d, k<c.N → UInt8.ofFin (Float8E5Bits.ofInt
    ((s.mem c.V (base c s c.svz c.svh+k*c.svk+d.val*c.svn)).read .int))=input.byteValue k d

theorem query_binding (A : Target.Ops α) (c : Config) (s : FP.Structural.State (Term α))
    (r : Fin 64) (input : Target.Inputs α c.D) (h : Binding c s r input)
    (hr : rows s r<c.H) (d : Fin c.D) (l : Fin 4) :
    lower A c.fp8 c.D l (query c s (r,d,()))=input.query d := by
  simp [query,hr,h.query,lower]

theorem pointers_eq (c : Config) (s : FP.Structural.State (Term α))
    (R : RegionName) (sz sh sr sd n : Nat) (k : Fin 64) (d : Fin c.D) :
    pointers c s R sz sh sr sd n (k,d,())=
      (R,base c s sz sh+(n*64+k.val)*sr+d.val*sd) := by
  simp [pointers,Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem pointer_offset (c : Config) (s : FP.Structural.State (Term α))
    (R : RegionName) (sz sh sr sd n : Nat) (k : Fin 64) (d : Fin c.D) :
    (pointers c s R sz sh sr sd n (k,d,())).2=
      base c s sz sh+(n*64+k.val)*sr+d.val*sd :=
  congrArg Prod.snd (pointers_eq c s R sz sh sr sd n k d)

theorem key_binding (A : Target.Ops α) (c : Config) (s : FP.Structural.State (Term α))
    (r : Fin 64) (input : Target.Inputs α c.D) (h : Binding c s r input)
    (n : Nat) (k : Fin 64) (d : Fin c.D) (l : Fin 4) :
    lower A c.fp8 c.D l (loaded c s c.K c.skz c.skh c.skn c.skk n (k,d,()))=
      Target.key A input c.fp8 c.N (n*64+k.val) d := by
  unfold loaded
  rw [pointer_offset c s c.K c.skz c.skh c.skn c.skk n k d]
  cases hf : c.fp8 <;> by_cases hk : n*64+k.val<c.N
  all_goals simp [hf,hk,h.halfKey,h.byteKey,Target.key,lower,literal]
  all_goals rfl

theorem value_binding (A : Target.Ops α) (c : Config) (s : FP.Structural.State (Term α))
    (r : Fin 64) (input : Target.Inputs α c.D) (h : Binding c s r input)
    (n : Nat) (k : Fin 64) (d : Fin c.D) (l : Fin 4) :
    lower A c.fp8 c.D l (loaded c s c.V c.svz c.svh c.svk c.svn n (k,d,()))=
      Target.value A input c.fp8 c.N (n*64+k.val) d := by
  unfold loaded
  rw [pointer_offset c s c.V c.svz c.svh c.svk c.svn n k d]
  cases hf : c.fp8 <;> by_cases hk : n*64+k.val<c.N
  all_goals simp [hf,hk,h.halfValue,h.byteValue,Target.value,lower,literal]
  all_goals rfl

theorem scores_binding (A : Target.Ops α) (C : ScalarABI.Conversions α)
    (arg : ScalarABI.Argument) (c : Config) (s : FP.Structural.State (Term α))
    (r : Fin 64) (input : Target.Inputs α c.D) (h : Binding c s r input)
    (hr : rows s r<c.H) (n : Nat) (k : Fin 64) (l : Fin 4) :
    lower A c.fp8 c.D l (scores c s (argument C arg) n r k)=
      Target.score A C input arg c.fp8 c.causal c.N c.SP (rows s r) n k := by
  cases hc : c.causal <;> by_cases hk : n*64+k.val<c.N <;>
    by_cases hv : n*64+k.val+c.SP≤rows s r
  all_goals simp [scores,hc,hk,hv,lower,literal,scale_binding,
    query_binding A c s r input h hr,key_binding A c s r input h,
    Target.score,NativeGraph.padded]
  all_goals rfl

#axiomsClean query_binding
#axiomsClean key_binding
#axiomsClean value_binding
#axiomsClean scores_binding
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Native
