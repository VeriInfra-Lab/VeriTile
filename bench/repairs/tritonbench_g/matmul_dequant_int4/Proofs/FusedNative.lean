import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.Native
import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.Autotune
namespace VeriTile.Bench.Repairs.MatmulDequantInt4.FusedNative
open Physical
set_option maxHeartbeats 4000000

/-- Finite autotune records retain each configuration's selected fixed ISA and
data-independent physical-lane fusion map. They contain no tensor values. -/
abbrev Trial := Library.MMAKind × Autotune.FusedTrial
def trials (s : Shape) (p : Native.Instructions) (ng : Bool) : List Trial → State Half → State Half
  | [],m => m
  | (kind,trial)::rest,m => trials s p ng rest
      (fusedRun s trial.tile (Native.fused p kind) ng trial.fusion trial.order m)
def trialsBytes (s : Shape) (p : Native.Instructions) (ng : Bool) : List Trial → Bytes → Bytes
  | [],m => m
  | (kind,trial)::rest,m => trialsBytes s p ng rest
      (Physical.run halfCodec .output s.n (fusedValue s trial.tile (Native.fused p kind) ng trial.fusion) trial.order m)
theorem trials_readonly (s : Shape) (p : Native.Instructions) (ng : Bool) (ts : List Trial)
    (initial current : State Half) (same : Writes.Readonly .output initial current) :
    Writes.Readonly .output initial (trials s p ng ts current) := by
  induction ts generalizing current with
  | nil => exact same
  | cons trial rest ih =>
    apply ih
    rw [fused_run_writes]
    exact Writes.run_readonly _ _ _ _ initial current same
theorem trials_load (s : Shape) (p : Native.Instructions) (ng : Bool) (ts : List Trial) (initial : Bytes) :
    load halfCodec (trialsBytes s p ng ts initial)=trials s p ng ts (load halfCodec initial) := by
  induction ts generalizing initial with
  | nil => rfl
  | cons trial rest ih =>
    change load halfCodec (trialsBytes s p ng rest _)=_
    rw [ih,Physical.run_load,←fused_run_writes];rfl
def call (s : Shape) (p : Native.Instructions) (ng : Bool) (ts : List Trial)
    (t : Tile) (kind : Library.MMAKind) (fusion : Nat → Nat → Fusion) (order : List Work) (initial : Bytes) :=
  Physical.run halfCodec .output s.n (fusedValue s t (Native.fused p kind) ng fusion) order (trialsBytes s p ng ts initial)
theorem call_load (s : Shape) (p : Native.Instructions) (ng : Bool) (ts : List Trial)
    (t : Tile) (kind : Library.MMAKind) (fusion : Nat → Nat → Fusion) (order : List Work) (initial : Bytes) :
    load halfCodec (call s p ng ts t kind fusion order initial)=
      fusedRun s t (Native.fused p kind) ng fusion order (trials s p ng ts (load halfCodec initial)) := by
  unfold call;rw [Physical.run_load,trials_load,←fused_run_writes]
theorem result (s : Shape) (p : Native.Instructions) (ng : Bool) (ts : List Trial)
    (t : Tile) (kind : Library.MMAKind) (fusion : Nat → Nat → Fusion) (g : Grid s.m s.n)
    (order : List Work) (complete : g.Complete order) (initial : Bytes)
    (r n : Nat) (hr : r<s.m) (hn : n<s.n) (i : Fin 2) :
    call s p ng ts t kind fusion order initial (.value .output) (2*(r*s.n+n)+i.val)=
      encode16 (fusedValue s t (Native.fused p kind) ng fusion (load halfCodec initial) ⟨r,n⟩) i := by
  change call s p ng ts t kind fusion order initial (.value .output) (halfCodec.width*(r*s.n+n)+i.val)=_
  rw [←original_byte halfCodec (call s p ng ts t kind fusion order initial) .output (r*s.n+n) i,
    call_load,fused_complete s t (Native.fused p kind) ng fusion g order complete _ r n hr hn,
    fused_stable s t (Native.fused p kind) ng fusion (load halfCodec initial) _
      (trials_readonly s p ng ts _ _ (Writes.reflexive _ _))]
  rfl
theorem trials_frame (s : Shape) (p : Native.Instructions) (ng : Bool) (ts : List Trial)
    (initial : Bytes) (r : Allocation) (b : Nat) (outside : r≠.value .output) :
    trialsBytes s p ng ts initial r b=initial r b := by
  induction ts generalizing initial with
  | nil => rfl
  | cons trial rest ih =>
    change trialsBytes s p ng rest _ r b=_
    rw [ih];apply Physical.frame
    intro w hw h;exact outside h.1
theorem frame (s : Shape) (p : Native.Instructions) (ng : Bool) (ts : List Trial)
    (t : Tile) (kind : Library.MMAKind) (fusion : Nat → Nat → Fusion) (order : List Work) (initial : Bytes)
    (r : Allocation) (b : Nat) (outside : r≠.value .output) :
    call s p ng ts t kind fusion order initial r b=initial r b := by
  unfold call
  rw [Physical.frame halfCodec .output s.n _ order _ r b (fun _ _ h => outside h.1),trials_frame s p ng ts initial r b outside]

theorem trials_frame_all (s : Shape) (p : Native.Instructions) (ng : Bool) (ts : List Trial)
    (active : ∀ trial∈ts,∀ w∈trial.2.order,w.Active s.m s.n)
    (initial : Bytes) (r : Allocation) (b : Nat) (outside : r≠.value .output ∨ s.m*s.n≤b/2) :
    trialsBytes s p ng ts initial r b=initial r b := by
  induction ts generalizing initial with
  | nil => rfl
  | cons trial rest ih =>
    change trialsBytes s p ng rest _ r b=_
    rw [ih (fun t ht => active t (by simp [ht]))]
    apply Physical.frame
    intro w hw ⟨hr,ha⟩
    have h := active trial (by simp) w hw
    have hb := rowMajor_bound _ _ _ _ h.1 h.2
    change b/2=w.row*s.n+w.col at ha
    rcases outside with h|h
    · exact h hr
    · omega
theorem frame_all (s : Shape) (p : Native.Instructions) (ng : Bool) (ts : List Trial)
    (active : ∀ trial∈ts,∀ w∈trial.2.order,w.Active s.m s.n)
    (t : Tile) (kind : Library.MMAKind) (fusion : Nat → Nat → Fusion) (g : Grid s.m s.n)
    (order : List Work) (complete : g.Complete order) (initial : Bytes)
    (r : Allocation) (b : Nat) (outside : r≠.value .output ∨ s.m*s.n≤b/2) :
    call s p ng ts t kind fusion order initial r b=initial r b := by
  unfold call
  rw [Physical.complete_frame halfCodec .output _ g order complete _ r b outside,
    trials_frame_all s p ng ts active initial r b outside]

#axiomsClean trials_readonly
#axiomsClean trials_load
#axiomsClean call_load
#axiomsClean result
#axiomsClean frame
#axiomsClean frame_all
end VeriTile.Bench.Repairs.MatmulDequantInt4.FusedNative
