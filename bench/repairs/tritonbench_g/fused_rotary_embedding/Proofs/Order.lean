import bench.repairs.tritonbench_g.fused_rotary_embedding.Proofs.NativeHost
import bench.repairs.tritonbench_g.fused_rotary_embedding.Proofs.Scheduling

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding.Binding
open Checked Arena
open Classical
noncomputable section

theorem bytes_injective : Function.Injective Physical.load := by
  intro a b eq
  funext r n
  have h := congrArg (fun words => Physical.encode (words r (n/4)) ⟨n%4,Nat.mod_lt _ (by decide)⟩) eq
  dsimp only at h
  rw [Physical.original_bytes,Physical.original_bytes] at h
  simpa only [show 4*(n/4)+n%4=n by omega] using h

theorem byte_order (c : Config) (valid : c.Valid) (p : Native.Instructions) (m : Physical.Bytes)
    (order : List Work) (complete : (schedule c.shape).Perm order) :
    Physical.byteExecute c p order m=Physical.native c p m := by
  apply bytes_injective
  rw [Physical.byteExecute_load]
  change Native.wordExecute c p order (Physical.load m)=Physical.load (Physical.byteExecute c p (schedule c.shape) m)
  rw [Physical.byteExecute_load]
  have h := execute_permutation c valid (Native.ops p) order (schedule c.shape) complete.symm
    (fun w hw => schedule_active c.shape w (complete.mem_iff.mpr hw))
    ((List.Perm.nodup_iff complete).mp (schedule_nodup c.shape)) (Native.liftMemory (Physical.load m))
  have hp := congrArg Native.project h
  rw [Native.wordExecute_project,Native.wordExecute_project] at hp
  exact hp

theorem order_mirror (c : Call) (p : Native.Instructions) (m : Arena.Memory)
    (checked : check c (mirror c m)=true) (order : List Work) (complete : (schedule c.shape).Perm order) :
    mirror c (Arena.execute c p order m)=mirror c (Arena.run c p m) := by
  have hc := config_valid c (mirror c m) checked
  rw [mirror_execute c p order m checked
    (fun w hw => schedule_active c.shape w (complete.mem_iff.mpr hw)),mirror_run c p m checked,
    Physical.dynamic_eq]
  exact (Physical.dynamicExecute_eq (Physical.configuration c.shape (mirror c m)) p order _ rfl).trans
    (byte_order _ hc p _ order complete)

/-- Arbitrary complete unique work order has the same finite memory, including
all bytes outside tensor windows. This lifts the retained source race proof. -/
theorem execute_eq_run (c : Call) (p : Native.Instructions) (m : Arena.Memory)
    (checked : check c (mirror c m)=true) (order : List Work) (complete : (schedule c.shape).Perm order) :
    Arena.execute c p order m=Arena.run c p m := by
  have hm := order_mirror c p m checked order complete
  funext a
  by_cases inside : ∃ r b,b<4*extent c r ∧ a=address c r b
  · rcases inside with ⟨r,b,hb,rfl⟩
    have h := congrFun (congrFun hm r) b
    simpa only [mirror,hb,ite_true] using h
  · have outside : Outside c (mirror c m) a := by
      intro kind t h d ht hh hd conflict
      have bound := logical_extent c (mirror c m) checked kind t h d ht hh hd
      apply inside
      refine ⟨kind.output,a.val-(c.tensor kind.output).base,by omega,?_⟩
      apply Fin.ext
      have off : (c.tensor kind.output).base+(a.val-(c.tensor kind.output).base)=a.val := by omega
      simp only [address,off,Nat.mod_eq_of_lt a.isLt]
    rw [Arena.execute_frame c p order m checked
      (fun w hw => schedule_active c.shape w (complete.mem_iff.mpr hw)) a outside,
      Arena.run_frame c p m checked a outside]

def invokeOrder (c : Call) (callerDevice : Nat) (p : Native.Instructions) (order : List Work)
    (m : Arena.Memory) : Option Result :=
  if check c (mirror c m) then some ⟨none,c.tensor,callerDevice,Arena.execute c p order m⟩ else none

theorem ordered_complete (c : Call) (callerDevice : Nat) (p : Native.Instructions)
    (m : Arena.Memory) (checked : check c (mirror c m)=true)
    (order : List Work) (legal : (schedule c.shape).Perm order) :
    ∃ result,invokeOrder c callerDevice p order m=some result ∧ NativeResult c callerDevice p p m result := by
  have eq : invokeOrder c callerDevice p order m=invoke c callerDevice p m := by
    simp only [invokeOrder,invoke,execute_eq_run c p m checked order legal]
  rw [eq]
  exact Binding.complete c callerDevice p m checked

#axiomsClean execute_eq_run
#axiomsClean ordered_complete
end
end VeriTile.Bench.Repairs.FusedRotaryEmbedding.Binding
