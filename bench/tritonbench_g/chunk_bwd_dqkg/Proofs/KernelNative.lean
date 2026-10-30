import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.KernelCompiler

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelNative
open VeriTile.Triton FP.Structural Whole KernelNativeCarrier KernelTensor
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
namespace Ref
export KernelNativeReference (accumulate query key gateResult)
end Ref
def empty : State Node where
  mem:=fun _ _=>.mk .real .zero
  regs:=fun _ _ _=>none
  pids:=fun _=>0
  numPids:=fun _=>1
  undef:=fun d _ _=>defaultValue d
def initial (b : Buffers c) (read : Nat → Word) : State Node :=
  physical b (fun r i addr=>Node.atom (.input r i) (read addr)) empty
theorem input_words (isa : Instructions) (route : Routing) (b : Buffers c) (read : Nat → Word)
    (r : RegionName) (i : Nat) (hi : i<(b.tensor r).metadata.numel) :
    observe isa route ((initial b read).mem r i)=
      (VeriTile.Torch.FP32.logicalValue (fun a=>⟨read a⟩) (b.tensor r) (b.laws r) i).bits := by
  simp only [initial,physical,hi,if_true,floatCell]
  split_ifs <;> simp [observe,Node.eval,Node.evaluate,VeriTile.Torch.FP32.logicalValue,
    (b.checked r).noNeg,logical_address b r i hi]
def entry (c : Config) (s : State Node) (a : CTA c) : State Node :=
  {s with regs:=fun _ _ _=>none
          pids:=fun i=>if i=0 then a.key.val else if i=1 then a.time.val else if i=2 then a.head.val else 0
          numPids:=fun i=>if i=0 then c.NK else if i=1 then c.NT else if i=2 then c.heads else 1}
def nativeCTA (a : CTA c) : CTA (nativeConfig c) := ⟨a.key,a.time,a.head⟩
theorem all_native (c : Config) : (allCTAs c).map nativeCTA=allCTAs (nativeConfig c) := by
  simp [allCTAs,List.map_flatMap,List.map_map,Function.comp_def,nativeCTA]
  rfl
theorem native_writes (c : Config) (order : List (CTA c)) :
    gridWrites (nativeConfig c) (order.map nativeCTA)=gridWrites c order := by
  funext pair
  apply propext
  constructor
  · rintro ⟨a,ha,hw⟩
    obtain ⟨b,hb,rfl⟩ := List.mem_map.mp ha
    exact ⟨b,hb,hw⟩
  · rintro ⟨a,ha,hw⟩
    exact ⟨nativeCTA a,List.mem_map.mpr ⟨a,ha,rfl⟩,hw⟩
def blocks (c : Config) : Nat := (c.V+c.BV-1)/c.BV
def tileQ (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node)
    (a : CTA c) (r : Fin 64) (e : Fin c.BK) : Node :=
  Ref.query isa route scale c (entry c s a) (Ref.accumulate isa route c (entry c s a) (blocks c)) (r,e,())
def tileK (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node)
    (a : CTA c) (r : Fin 64) (e : Fin c.BK) : Node :=
  Ref.key isa route scale c (entry c s a) (Ref.accumulate isa route c (entry c s a) (blocks c)) (r,e,())
def tileG (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node)
    (a : CTA c) (r : Fin 64) : Node :=
  Ref.gateResult isa route scale c (entry c s a) (Ref.accumulate isa route c (entry c s a) (blocks c)) (r,())
def referenceQ (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node)
    (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K) : Word :=
  (tileQ isa route scale c s (qOwner c h t e) (rowLane t.val) (keyLane c e.val)).eval isa route
def referenceK (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node)
    (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K) : Word :=
  (tileK isa route scale c s (qOwner c h t e) (rowLane t.val) (keyLane c e.val)).eval isa route
def referenceG (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node)
    (key : Fin c.NK) (h : Fin c.heads) (t : Fin c.T) : Word :=
  (tileG isa route scale c s (gOwner c key h t) (rowLane t.val)).eval isa route
theorem tiles (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node)
    (a : CTA c) (r : Fin 64) (e : Fin c.BK) :
    Typed.resultQ (algebra isa route scale) dotModel (nativeConfig c) s (nativeCTA a) r e=tileQ isa route scale c s a r e ∧
    Typed.resultK (algebra isa route scale) dotModel (nativeConfig c) s (nativeCTA a) r e=tileK isa route scale c s a r e ∧
    Typed.resultG (algebra isa route scale) dotModel (nativeConfig c) s (nativeCTA a) r=tileG isa route scale c s a r := by
  simp only [Typed.resultQ,Typed.resultK,Typed.resultG,KernelNativeReference.recurrence,
    (KernelNativeReference.answers isa route scale c _ _).1,
    (KernelNativeReference.answers isa route scale c _ _).2.1,
    (KernelNativeReference.answers isa route scale c _ _).2.2]
  exact ⟨rfl,rfl,rfl⟩
structure Result (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (before final : State Node) : Prop where
  q : ∀ (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K),observe isa route (final.mem "dq" ((h.val*c.T+t.val)*c.K+e.val))=
    referenceQ isa route scale c before h t e
  k : ∀ (h : Fin c.heads) (t : Fin c.T) (e : Fin c.K),observe isa route (final.mem "dk" ((h.val*c.T+t.val)*c.K+e.val))=
    referenceK isa route scale c before h t e
  g : ∀ (key : Fin c.NK) (h : Fin c.heads) (t : Fin c.T),observe isa route (final.mem "dg" ((key.val*c.heads+h.val)*c.T+t.val))=
    referenceG isa route scale c before key h t
  frame : ∀ r i,¬Writes c r i → final.mem r i=before.mem r i
theorem source (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (s : State Node)
    (order : List (CTA c)) (ho : order.Perm (allCTAs c)) :
    ∃ final,Typed.runGrid (algebra isa route scale) dotModel (nativeConfig c) (order.map nativeCTA) s=some final ∧
      Result isa route scale c s final := by
  have orderNative : (order.map nativeCTA).Perm (allCTAs (nativeConfig c)) := by
    rw [←all_native]
    exact ho.map nativeCTA
  obtain ⟨final,he,hq,hk,hg,hf⟩ := Typed.whole_result (algebra isa route scale) dotModel (nativeConfig c) s
    (order.map nativeCTA) orderNative
  rw [←all_native,native_writes] at hf
  refine ⟨final,he,⟨?_,?_,?_,hf⟩⟩
  · intro h t e
    have hv := hq h t e
    change final.mem "dq" ((h.val*c.T+t.val)*c.K+e.val)=_ at hv
    rw [hv]
    change (Typed.resultQ (algebra isa route scale) dotModel (nativeConfig c) s
      (nativeCTA (qOwner c h t e)) (rowLane t.val) (keyLane c e.val)).eval isa route=_
    rw [(tiles isa route scale c s (qOwner c h t e) _ _).1]
    rfl
  · intro h t e
    have hv := hk h t e
    change final.mem "dk" ((h.val*c.T+t.val)*c.K+e.val)=_ at hv
    rw [hv]
    change (Typed.resultK (algebra isa route scale) dotModel (nativeConfig c) s
      (nativeCTA (qOwner c h t e)) (rowLane t.val) (keyLane c e.val)).eval isa route=_
    rw [(tiles isa route scale c s (qOwner c h t e) _ _).2.1]
    rfl
  · intro key h t
    have hv := hg key h t
    change final.mem "dg" ((key.val*c.heads+h.val)*c.T+t.val)=_ at hv
    rw [hv]
    change (Typed.resultG (algebra isa route scale) dotModel (nativeConfig c) s
      (nativeCTA (gOwner c key h t)) (rowLane t.val)).eval isa route=_
    rw [(tiles isa route scale c s (gOwner c key h t) _ (keyLane c 0)).2.2]
    rfl
def runQuoted (quote : ComputeKernel) (isa : Instructions) (route : Routing) (scale : Word) (c : Config) :
    List (CTA c) → State Node → Option (State Node)
  | [],s=>some s
  | a::rest,s=>do
      let t←FP.Structural.exec (FP.ProfiledRewrite.engine (Typed.withDot (algebra isa route scale) dotModel))
        quote (Typed.prepare (nativeConfig c) s (nativeCTA a))
      runQuoted quote isa route scale c rest t
theorem run_quote (isa : Instructions) (route : Routing) (scale : Word) (c : Config) (order : List (CTA c)) (s : State Node) :
    runQuoted (Whole.kernel (nativeConfig c)) isa route scale c order s=
      Typed.runGrid (algebra isa route scale) dotModel (nativeConfig c) (order.map nativeCTA) s := by
  induction order generalizing s with
  | nil=>rfl
  | cons a rest ih=>simp only [runQuoted,List.map_cons,Typed.runGrid,ih]

theorem complete {Store Byte : Type} (api : Memory.API Word Store Byte)
    (platform : KernelCompiler.Platform) (b : Buffers c) (binding : KernelCompiler.Binding platform b)
    (conversion : KernelCompiler.ScaleConversion) (argument : KernelCompiler.ScaleArgument)
    (scale : KernelCompiler.LaunchScale conversion argument c) (store : Store)
    (order : List (CTA c)) (ho : order.Perm (allCTAs c)) :
    let isa:=platform.instructions binding.object
    let route:=binding.decoded.routing
    ∃ final,runQuoted binding.decoded.source isa route scale.value.word.bits c order (initial b (api.read store))=some final ∧
      Result isa route scale.value.word.bits c (initial b (api.read store)) final ∧
      Shared.TensorImage api (placement b) (fun _ _=>observe isa route) final store ∧
      (∀ r i,Writes c r i →
        (VeriTile.Torch.FP32.logicalValue (fun addr=>⟨api.read (Shared.commit api (placement b)
          (fun _ _=>observe isa route) final store) addr⟩) (b.tensor r) (b.laws r) i).bits=observe isa route (final.mem r i)) := by
  dsimp only
  obtain ⟨final,he,hr⟩ := source (platform.instructions binding.object) binding.decoded.routing
    scale.value.word.bits c (initial b (api.read store)) order ho
  refine ⟨final,?_,hr,Shared.complete api (placement b) _ final store,?_⟩
  · rw [binding.quote,run_quote]
    exact he
  · exact Shared.native_logical_output api (placement b) _ final store

#axiomsClean input_words
#axiomsClean tiles
#axiomsClean source
#axiomsClean complete
#specNonCircular referenceQ avoiding [Typed.expectedQ,Typed.runGrid,FP.Structural.exec]
#specNonCircular referenceK avoiding [Typed.expectedK,Typed.runGrid,FP.Structural.exec]
#specNonCircular referenceG avoiding [Typed.expectedG,Typed.runGrid,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelNative
