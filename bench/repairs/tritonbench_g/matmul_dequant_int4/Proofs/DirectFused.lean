import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.DirectQuant

namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
open Host Physical
noncomputable section
open Classical
set_option maxHeartbeats 5000000

def fusedConfig (choice : Fin 17) : Source.FusedConfig := Source.fusedConfigs[choice.val]'(by
  simpa [Source.fusedConfigs] using choice.isLt)
theorem fusedConfig_mem (choice : Fin 17) : fusedConfig choice∈Source.fusedConfigs := List.getElem_mem _
def rawGrid (s : Shape) (hs : s.Valid) (choice : Fin 17) :=
  fusedGrid s (fusedConfig choice).tile hs (Source.fused_valid _ (fusedConfig_mem choice))
def rawOrder (s : Shape) (hs : s.Valid) (choice : Fin 17) := (rawGrid s hs choice).schedule
structure RawChoice where
  config : Fin 17
  instruction : Library.MMAKind
  fusion : Nat → Nat → Fusion
def rawTrial (s : Shape) (hs : s.Valid) (choice : RawChoice) : FusedNative.Trial :=
  ⟨choice.instruction,⟨(fusedConfig choice.config).tile,choice.fusion,rawOrder s hs choice.config⟩⟩
def rawTrials (s : Shape) (hs : s.Valid) (choices : List RawChoice) := choices.map (rawTrial s hs)
theorem raw_trials_active (s : Shape) (hs : s.Valid) (choices : List RawChoice) :
    ∀ trial∈rawTrials s hs choices,∀ w∈trial.2.order,w.Active s.m s.n := by
  intro trial member w used
  obtain ⟨choice,_,rfl⟩ := List.mem_map.mp member
  exact (Grid.schedule_mem _ w).mp used

/-- Explicit canonical raw ABI envelope, including masked M lanes and the final
K pointer advance. This is a raw-kernel precondition, not an invented Python host. -/
def RawIndexGuard (s : Shape) : Prop :=
  let pm := ceil s.m 128*128
  let pk := s.k+128
  max pm (max s.n (max pk s.group))<2^31 ∧
  pm*s.k+pk<2^31 ∧ (pk/8+1)*s.n+s.n<2^31 ∧
  (pk/s.group+1)*s.n+s.n<2^31 ∧ (pk/s.group+1)*(s.n/8)+s.n/8<2^31 ∧
  pm*s.n+s.n<2^31 ∧ pk*s.n+s.n<2^31 ∧ pm*s.n<2^31

def rawDomain (s : Shape) (ng : Bool) (source : Source.FusedRawScope s ng)
    (guard : RawIndexGuard s) : MachineDomain.Fused s ng :=
  ⟨source,⟨ceil s.m 128*128,s.n,s.k+128,ceil_cover _ _ (by decide),le_refl _,by omega,
    guard.1,guard.2.1,guard.2.2.1,guard.2.2.2.1,guard.2.2.2.2.1,guard.2.2.2.2.2.1,guard.2.2.2.2.2.2.1⟩,
    le_refl _,le_refl _,guard.2.2.2.2.2.2.2⟩

structure RawInfo where
  input : Info
  weight : Info
  scale : Info
  zero : Info
  output : Info
  group : Nat
  groupIsInt : Bool
  noGroups : Bool
def rawShape (p : RawInfo) : Shape := ⟨p.input.tensor.rows,p.weight.tensor.columns,p.input.tensor.columns,p.group⟩
def RawAccepted (p : RawInfo) : Prop :=
  let s := rawShape p
  p.groupIsInt=true ∧ Source.FusedRawScope s p.noGroups ∧ RawIndexGuard s ∧
  Ordinary p.input .fp16 s.m s.k p.input.tensor.device ∧
  Ordinary p.weight .int32 (s.k/8) s.n p.input.tensor.device ∧
  Ordinary p.scale .fp16 (s.k/s.group) s.n p.input.tensor.device ∧
  Ordinary p.zero .int32 (s.k/s.group) (s.n/8) p.input.tensor.device ∧
  Ordinary p.output .fp16 s.m s.n p.input.tensor.device ∧
  p.input.tensor.strides=Strides.canonical s.k ∧ p.weight.tensor.strides=Strides.canonical s.n ∧
  p.scale.tensor.strides=Strides.canonical s.n ∧ p.zero.tensor.strides=Strides.canonical (s.n/8) ∧
  p.output.tensor.strides=Strides.canonical s.n ∧
  Host.Independent [p.output.tensor,p.input.tensor,p.weight.tensor,p.scale.tensor,p.zero.tensor]
def checkRaw (p : RawInfo) : Except Error Unit := if RawAccepted p then .ok () else .error .geometry
theorem checkRaw_iff (p : RawInfo) : checkRaw p=.ok () ↔ RawAccepted p := by simp [checkRaw]

def rawBase (p : RawInfo) : Physical.Allocation → Nat
  | .value .a => p.input.tensor.base
  | .packed .weight => p.weight.tensor.base
  | .value .scale => p.scale.tensor.base
  | .packed .zero => p.zero.tensor.base
  | .value .output => p.output.tensor.base
  | _ => 0

def rawCall (p : RawInfo) (h : RawAccepted p) (I : Native.Instructions)
    (choices : List RawChoice) (chosen : RawChoice) (heap : Heap) (caller : Nat) : TensorResult :=
  let s := rawShape p
  let data := FusedNative.call s I p.noGroups (rawTrials s h.2.1.shape choices)
    (fusedConfig chosen.config).tile chosen.instruction chosen.fusion (rawOrder s h.2.1.shape chosen.config)
    (bind (rawBase p) heap)
  ⟨p.output.tensor,writeSpan p.output.tensor.base (2*s.m*s.n) (data (.value .output)) heap,p.input.tensor.device,caller⟩

structure RawClaim (p : RawInfo) (I : Native.Instructions) (chosen : RawChoice)
    (heap : Heap) (caller : Nat) (result : TensorResult) : Prop where
  identity : result.tensor=p.output.tensor
  metadata : result.tensor.Ordinary .fp16 (rawShape p).m (rawShape p).n p.input.tensor.device
  device : result.launchDevice=p.input.tensor.device ∧ result.callerDevice=caller
  bytes : ∀ r n,r<(rawShape p).m → n<(rawShape p).n → ∀ i : Fin 2,
    result.heap (result.tensor.base+(2*(r*(rawShape p).n+n)+i.val))=
      encode16 (fusedValue (rawShape p) (fusedConfig chosen.config).tile (Native.fused I chosen.instruction)
        p.noGroups chosen.fusion (load halfCodec (bind (rawBase p) heap)) ⟨r,n⟩) i
  frame : ∀ address,¬(p.output.tensor.base≤address ∧ address<p.output.tensor.base+2*(rawShape p).m*(rawShape p).n) →
    result.heap address=heap address
  inputs : ∀ t,t∈[p.input.tensor,p.weight.tensor,p.scale.tensor,p.zero.tensor] →
    ∀ off,off<t.storageBytes → result.heap (t.base+off)=heap (t.base+off)
  grid : blocksM (rawShape p) (fusedConfig chosen.config).tile*blocksN (rawShape p) (fusedConfig chosen.config).tile<2^31
  indices : ∀ r n k,r<ceil (rawShape p).m (fusedConfig chosen.config).tile.bm*(fusedConfig chosen.config).tile.bm →
    n<(rawShape p).n → k<(rawShape p).k+(fusedConfig chosen.config).tile.bk →
    aAddress (rawShape p) r k<2^31 ∧ qAddress (rawShape p) k n<2^31 ∧ sAddress (rawShape p) k n<2^31 ∧
    zAddress (rawShape p) k n<2^31 ∧ cAddress (rawShape p) r n<2^31 ∧ cAddress (rawShape p) k n<2^31

theorem raw_native (p : RawInfo) (h : RawAccepted p) (I : Native.Instructions)
    (choices : List RawChoice) (chosen : RawChoice) (heap : Heap) (caller : Nat) :
    RawClaim p I chosen heap caller (rawCall p h I choices chosen heap caller) := by
  have layout := h.2.2.2.2.2.2.2.1.2.2.2.2.2.2.1
  have capacity : 2*(rawShape p).m*(rawShape p).n≤p.output.tensor.storageBytes := ordinary_capacity _ _ _ _ _ layout
  have independent := h.2.2.2.2.2.2.2.2.2.2.2.2.2
  refine ⟨rfl,layout,⟨rfl,rfl⟩,?_,?_,?_,?_,?_⟩
  · intro r n hr hn i
    dsimp only [rawCall]
    rw [writeSpan_byte]
    · exact FusedNative.result _ I p.noGroups _ _ _ _ (rawGrid _ h.2.1.shape chosen.config) _
        (List.Perm.refl _) _ r n hr hn i
    · have := rowMajor_bound _ _ _ _ hr hn
      have := i.isLt
      nlinarith
  · exact writeSpan_frame _ _ _ heap
  · intro t used off bound
    have apart : Host.Disjoint p.output.tensor t := (List.pairwise_cons.mp independent).1 t used
    exact writeSpan_disjoint _ _ apart _ capacity _ _ off bound
  · exact MachineDomain.fused_grid_indices _ _ (rawDomain _ _ h.2.1 h.2.2.1) _ (fusedConfig_mem chosen.config)
  · exact MachineDomain.fused_formed _ _ (rawDomain _ _ h.2.1 h.2.2.1) _ (fusedConfig_mem chosen.config)

#axiomsClean raw_trials_active
#axiomsClean raw_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
