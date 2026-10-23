import bench.repairs.tritonbench_g.int4_matmul.Proofs.DirectData
import bench.repairs.tritonbench_g.int4_matmul.Proofs.Source

namespace VeriTile.Bench.Repairs.Int4Matmul.Direct
open Host Physical
noncomputable section
set_option maxHeartbeats 2000000

structure Invocation (s : Shape) where
  tile : Tile
  member : tile∈sourceTiles
  schedule : List Work
  complete : Complete s tile schedule

def Invocation.valid (c : Invocation s) : c.tile.Valid := all_tiles_valid _ c.member

theorem readonly_trans (a b c : State α) (ab : Readonly a b) (bc : Readonly b c) : Readonly a c :=
  ⟨fun r i ne=>(bc.1 r i ne).trans (ab.1 r i ne),bc.2.trans ab.2⟩
theorem readonly_self (m : State α) : Readonly m m := ⟨fun _ _ _=>rfl,rfl⟩
theorem call_readonly (s : Shape) (p : Ops α) (c : Invocation s) (m : State α) : Readonly m (call s c.tile p c.schedule m) :=
  execute_readonly _ _ _ _ _ _ (zero_readonly _ _ _)

def trialsRun (s : Shape) (p : Ops α) : List (Invocation s) → State α → State α
  | [],m=>m
  | c::cs,m=>trialsRun s p cs (call s c.tile p c.schedule m)
theorem trials_readonly (s : Shape) (p : Ops α) (cs : List (Invocation s)) (m : State α) : Readonly m (trialsRun s p cs m) := by
  induction cs generalizing m with
  | nil=>exact readonly_self _
  | cons c cs ih=>exact readonly_trans _ _ _ (call_readonly s p c m) (ih _)
def wrapperRun (s : Shape) (p : Ops α) (trials : List (Invocation s)) (chosen : Invocation s) (m : State α) :=
  call s chosen.tile p chosen.schedule (trialsRun s p trials (zeroOutput s p m))

theorem referencePartial_current (s : Shape) (t : Tile) (p : Ops α) (before current : State α)
    (same : Readonly before current) (w : Work) : referencePartial s t p current w=referencePartial s t p before w := by
  have aa : current.values .a=before.values .a:=funext (fun i=>same.1 .a i (by decide))
  have ss : current.values .scale=before.values .scale:=funext (fun i=>same.1 .scale i (by decide))
  simp only [referencePartial,dequantized,logicalA,logicalQ,logicalZ,logicalScale,aa,ss,same.2]
theorem referenceOutput_current (s : Shape) (t : Tile) (p : Ops α) (before current : State α)
    (same : Readonly before current) (row col : Nat) (ws : List Work) (seed : α) :
    referenceOutput s t p current row col ws seed=referenceOutput s t p before row col ws seed := by
  induction ws generalizing seed with
  | nil=>rfl
  | cons w ws ih=>simp only [referenceOutput,referencePartial_current s t p before current same,ih]
theorem mathematical_current (s : Shape) (before current : State ℝ) (same : Readonly before current) (row col : Nat) :
    mathematical s current row col=mathematical s before row col := by
  have aa : current.values .a=before.values .a:=funext (fun i=>same.1 .a i (by decide))
  have ss : current.values .scale=before.values .scale:=funext (fun i=>same.1 .scale i (by decide))
  simp only [mathematical,logicalA,logicalQ,logicalZ,logicalScale,aa,ss,same.2]

theorem wrapper_real (s : Shape) (hs : s.Valid) (trials : List (Invocation s)) (chosen : Invocation s)
    (initial current : State ℝ) (same : Readonly initial current) (row col : Nat) (hr : row<s.m) (hc : col<s.n) :
    (wrapperRun s realOps trials chosen current).values .output (cAddress s row col)=mathematical s initial row col := by
  unfold wrapperRun
  rw [whole_real s chosen.tile hs chosen.valid _ chosen.schedule chosen.complete row col hr hc]
  exact mathematical_current s initial _ (readonly_trans _ _ _ same
    (readonly_trans _ _ _ (zero_readonly _ _ _) (trials_readonly _ _ _ _))) row col

theorem trials_frame (s : Shape) (hs : s.Valid) (p : Ops α) (cs : List (Invocation s))
    (m : State α) (r : Region) (i : Nat) (outside : r≠.output ∨ s.m*s.n≤ i) :
    (trialsRun s p cs m).values r i=m.values r i := by
  induction cs generalizing m with
  | nil=>rfl
  | cons c cs ih=>exact (ih _).trans (call_frame s c.tile hs c.valid p c.schedule c.complete m r i outside)
theorem wrapper_frame (s : Shape) (hs : s.Valid) (p : Ops α) (cs : List (Invocation s)) (chosen : Invocation s)
    (m : State α) (r : Region) (i : Nat) (outside : r≠.output ∨ s.m*s.n≤ i) :
    (wrapperRun s p cs chosen m).values r i=m.values r i := by
  unfold wrapperRun
  rw [call_frame s chosen.tile hs chosen.valid p chosen.schedule chosen.complete _ r i outside,trials_frame s hs p cs _ r i outside]
  rcases outside with ne|bound
  · simp [zeroOutput,ne]
  · simp [zeroOutput,not_lt_of_ge bound]

structure NativeInvocation (s : Shape) extends Invocation s where
  isa : Native.ISA
  parsedProduction : String
  sourceIdentity : parsedProduction=SourceBinding.productionAST

def nativeTrials (s : Shape) (p : Native.Instructions) : List (NativeInvocation s) → Bytes → Bytes
  | [],m=>m
  | c::cs,m=>nativeTrials s p cs (byteCall s c.tile p c.isa c.schedule m)
def nativeWrapper (s : Shape) (p : Native.Instructions) (trials : List (NativeInvocation s))
    (chosen : NativeInvocation s) (m : Bytes) : Bytes :=
  byteCall s chosen.tile p chosen.isa chosen.schedule (nativeTrials s p trials (byteZero s m))
theorem nativeTrials_readonly (s : Shape) (p : Native.Instructions) (cs : List (NativeInvocation s)) (m : Bytes) :
    Readonly (load m) (load (nativeTrials s p cs m)) := by
  induction cs generalizing m with
  | nil=>exact readonly_self _
  | cons c cs ih=>
    apply readonly_trans _ _ _ ?_ (ih _)
    rw [call_load]
    exact call_readonly s (Native.ops p c.isa) c.toInvocation _
theorem native_result (s : Shape) (hs : s.Valid) (p : Native.Instructions) (trials : List (NativeInvocation s))
    (chosen : NativeInvocation s) (m : Bytes) (row col : Nat) (hr : row<s.m) (hc : col<s.n) (i : Fin 4) :
    nativeWrapper s p trials chosen m (.value .output) (4*cAddress s row col+i.val)=
      encode (referenceOutput s chosen.tile (Native.ops p chosen.isa) (load m) row col chosen.schedule 0) i := by
  unfold nativeWrapper
  rw [call_bytes s chosen.tile hs chosen.valid p chosen.isa chosen.schedule chosen.complete _ row col hr hc]
  congr 1
  apply referenceOutput_current
  apply readonly_trans _ _ _ ?_ (nativeTrials_readonly s p trials (byteZero s m))
  rw [zero_load s p chosen.isa]
  exact zero_readonly _ _ _
theorem nativeTrials_frame (s : Shape) (hs : s.Valid) (p : Native.Instructions) (cs : List (NativeInvocation s))
    (m : Bytes) (r : Physical.Allocation) (i : Nat) (outside : r≠.value .output ∨ s.m*s.n≤ i/4) :
    nativeTrials s p cs m r i=m r i := by
  induction cs generalizing m with
  | nil=>rfl
  | cons c cs ih=>exact (ih _).trans (call_frame_bytes s c.tile hs c.valid p c.isa c.schedule c.complete m r i outside)
theorem native_frame (s : Shape) (hs : s.Valid) (p : Native.Instructions) (trials : List (NativeInvocation s))
    (chosen : NativeInvocation s) (m : Bytes) (r : Physical.Allocation) (i : Nat) (outside : r≠.value .output ∨ s.m*s.n≤ i/4) :
    nativeWrapper s p trials chosen m r i=m r i := by
  unfold nativeWrapper
  rw [call_frame_bytes s chosen.tile hs chosen.valid p chosen.isa chosen.schedule chosen.complete _ r i outside,
    nativeTrials_frame s hs p trials _ r i outside]
  rcases outside with ne|bound
  · simp [byteZero,ne]
  · simp [byteZero,not_lt_of_ge bound]

#axiomsClean wrapper_real
#axiomsClean wrapper_frame
#axiomsClean native_result
#axiomsClean native_frame
end
end VeriTile.Bench.Repairs.Int4Matmul.Direct
