import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.Materialized

namespace VeriTile.Bench.Repairs.BmmChunkBwd.DirectFP
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open NativeGraph
noncomputable section
open Classical
set_option maxHeartbeats 6000000

abbrev Byte := Fin 256
abbrev Heap := Nat → Byte

def load32 (memory : Heap) (base index : Nat) : Word :=
  BitVec.ofNat 32 (((List.range 4).map fun b => (memory (base+4*index+b)).val*256^b).sum)
def byte (w : Word) (b : Nat) : Byte := ⟨w.toNat/256^b%256,Nat.mod_lt _ (by decide)⟩

/-- Direct physical original bytes, including the caller's existing output.
All four roles have separate admitted spans; no preparation copy occurs. -/
def initial (p : Metadata) (a : Direct.Allocation p) (heap : Heap) : State Expr where
  mem := fun r off => .mk .fp32 (.leaf (load32 heap (Direct.byteBase p a r) off))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun dtype _ _ => defaultValue dtype

def one (rest : Algebra Expr) (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (choice : Fin 9) (s : State Expr) : State Expr :=
  NativeExecution.grid (algebra rest) (dot choice) (call p) (tuning choice) (Direct.placement p)
    (Direct.nativeAllocation p h a) (HostFP.grid (call p) (tuning choice)) s

def trials (rest : Algebra Expr) (p : Metadata) (h : InputContract p) (a : Direct.Allocation p) :
    List (Fin 9) → State Expr → State Expr
  | [],s => s
  | c::cs,s => trials rest p h a cs (one rest p h a c s)

def execute (rest : Algebra Expr) (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (cs : List (Fin 9)) (chosen : Fin 9) (s : State Expr) : State Expr :=
  if (p.tensor .a).numel=0 then s else one rest p h a chosen (trials rest p h a cs s)

theorem initial_good (p : Metadata) (a : Direct.Allocation p) (heap : Heap) : Good (initial p a heap) :=
  fun _ _ => ⟨rfl,True.intro⟩
theorem one_good (rest : Algebra Expr) (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (choice : Fin 9) (s : State Expr) (hs : Good s) : Good (one rest p h a choice s) :=
  grid_good rest (call p) choice _ _ _ s hs

theorem trials_good (rest : Algebra Expr) (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (cs : List (Fin 9)) (s : State Expr) (hs : Good s) : Good (trials rest p h a cs s) := by
  induction cs generalizing s with
  | nil => exact hs
  | cons c cs ih => exact ih _ (one_good rest p h a c s hs)

theorem execute_good (rest : Algebra Expr) (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (cs : List (Fin 9)) (chosen : Fin 9) (s : State Expr) (hs : Good s) : Good (execute rest p h a cs chosen s) := by
  unfold execute
  split_ifs
  · exact hs
  · exact one_good rest p h a chosen _ (trials_good rest p h a cs s hs)

/-- Full actual typed-source launch correspondence. All native controls,
addresses, masks and source stores are established from admission. -/
def SourceLaunch (rest : Algebra Expr) (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (choice : Fin 9) (s : State Expr) : Prop :=
  ∃ source,HostFP.run (FPDefault.backend (algebra rest) (dot choice)) (call p) (tuning choice) Bool.false
      (HostFP.grid (call p) (tuning choice)) (FPFlat.lift (call p) (Direct.placement p) s)=some source ∧
    (one rest p h a choice s).mem=(FPFlat.publish (Direct.placement p) (call p).Out s source).mem

theorem source_one (rest : Algebra Expr) (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (positive : 0<(p.tensor .a).numel) (choice : Fin 9) (s : State Expr) (hs : Good s) :
    SourceLaunch rest p h a choice s :=
  NativeExecution.source_grid (algebra rest) (dot choice) (call p) choice (Direct.precision p)
    (Direct.nativeMetadata p h positive choice) (Direct.placement p) (Direct.nativeAllocation p h a)
    (Direct.flat p a) (Direct.layout p h a positive (tuning choice)) rfl (Direct.residualTag p)
    (Direct.freshStorage p) s (fun off _ => (hs _ _).1)

/-- Independent blocked arithmetic readback for every actual active writer.
The entire native launch and entire typed source produce this same cell. -/
theorem one_readback (rest : Algebra Expr) (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (positive : 0<(p.tensor .a).numel) (choice : Fin 9) (s : State Expr) (hs : Good s)
    (q : HostFP.Pid) (hq : q∈HostFP.grid (call p) (tuning choice))
    (i : TileIndex [(tuning choice).BM,(tuning choice).BN])
    (hi : FPReference.active (HostFP.launch (call p) (tuning choice) q s) (tuning choice).BM (tuning choice).BN
      (HostContract.parameters (call p)) i) :
    (one rest p h a choice s).mem (call p).Out (FPFlat.outOffset (call p) (tuning choice) q s i)=
      .mk .fp32 (FPReference.expected (algebra rest) (dot choice) (tuning choice).BM (tuning choice).BN
        (tuning choice).BCS (HostContract.parameters (call p))
        (HostFP.launch (call p) (tuning choice) q (FPFlat.lift (call p) (Direct.placement p) s)) i) := by
  obtain ⟨source,run,mem⟩ := source_one rest p h a positive choice s hs
  obtain ⟨source',run',values,_⟩ := FPGridOutput.complete (algebra rest) (dot choice) (call p) (tuning choice)
    (Direct.precision p) (Direct.flat p a) (Direct.layout p h a positive (tuning choice)) (FPFlat.lift (call p) (Direct.placement p) s)
  have eq : source'=source := Option.some.inj (run'.symm.trans run)
  subst source'
  have bound := FPFlat.output_bound (call p) (tuning choice) (Direct.flat p a)
    (Direct.layout p h a positive (tuning choice)) q hq s i hi
  have pub := FPFlat.publish_read (Direct.placement p) (call p).Out s source
    (FPFlat.outOffset (call p) (tuning choice) q s i) bound
  rw [mem]
  simp only [Direct.placement,FPFlat.address,Nat.zero_add,id_eq] at pub
  exact pub.trans (values q hq i hi)

def Certified (rest : Algebra Expr) (p : Metadata) (h : InputContract p) (a : Direct.Allocation p) :
    List (Fin 9) → State Expr → Prop
  | [],_ => True
  | c::cs,s => SourceLaunch rest p h a c s ∧ Certified rest p h a cs (one rest p h a c s)

theorem certified (rest : Algebra Expr) (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (positive : 0<(p.tensor .a).numel) (cs : List (Fin 9)) (s : State Expr) (hs : Good s) :
    Certified rest p h a cs s := by
  induction cs generalizing s with
  | nil => trivial
  | cons c cs ih => exact ⟨source_one rest p h a positive c s hs,ih _ (one_good rest p h a c s hs)⟩

theorem one_frame (rest : Algebra Expr) (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (positive : 0<(p.tensor .a).numel) (choice : Fin 9) (s : State Expr) (hs : Good s)
    (r : RegionName) (off : Nat) (outside : ¬(r=(call p).Out ∧ off<(p.tensor .a).numel)) :
    (one rest p h a choice s).mem r off=s.mem r off := by
  obtain ⟨source,_,eq⟩ := source_one rest p h a positive choice s hs
  rw [congrFun (congrFun eq r) off]
  apply FPFlat.publish_frame
  simpa [FPFlat.scratchCell,Direct.placement,Direct.extent,call] using outside

theorem trials_frame (rest : Algebra Expr) (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (positive : 0<(p.tensor .a).numel) (cs : List (Fin 9)) (s : State Expr) (hs : Good s)
    (r : RegionName) (off : Nat) (outside : ¬(r=(call p).Out ∧ off<(p.tensor .a).numel)) :
    (trials rest p h a cs s).mem r off=s.mem r off := by
  induction cs generalizing s with
  | nil => rfl
  | cons c cs ih =>
    exact (ih _ (one_good rest p h a c s hs)).trans (one_frame rest p h a positive c s hs r off outside)

theorem execute_frame (rest : Algebra Expr) (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (cs : List (Fin 9)) (chosen : Fin 9) (s : State Expr) (hs : Good s)
    (r : RegionName) (off : Nat) (outside : ¬(r=(call p).Out ∧ off<(p.tensor .a).numel)) :
    (execute rest p h a cs chosen s).mem r off=s.mem r off := by
  unfold execute
  split_ifs with empty
  · rfl
  · have pos : 0<(p.tensor .a).numel := by omega
    exact (one_frame rest p h a pos chosen _ (trials_good rest p h a cs s hs) r off outside).trans
      (trials_frame rest p h a pos cs s hs r off outside)

def finalState (rest : Algebra Expr) (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (cs : List (Fin 9)) (chosen : Fin 9) (heap : Heap) : State Expr :=
  execute rest p h a cs chosen (initial p a heap)
def graph (p : Metadata) (s : State Expr) (off : Nat) : Expr := (s.mem (call p).Out off).read .fp32

def outputHeap (p : Metadata) (a : Direct.Allocation p) (actual : Arithmetic)
    (s : State Expr) (heap : Heap) (address : Nat) : Byte :=
  if a.pointer≤address ∧ address<a.pointer+4*(p.tensor .a).numel then
    byte (interpret actual (graph p s ((address-a.pointer)/4))) ((address-a.pointer)%4)
  else heap address

structure Result where
  object : Nat
  shape : List Nat
  device : VeriTile.Torch.Device
  dtype : VeriTile.Torch.DType
  pointer : Nat
  heap : Heap

def result (p : Metadata) (a : Direct.Allocation p) (actual : Arithmetic) (s : State Expr) (heap : Heap) : Result :=
  ⟨a.object,(p.tensor .a).shape,(p.tensor .a).device,.floating .fp32,a.pointer,outputHeap p a actual s heap⟩

def runWrapper (rest : Algebra Expr) (actual : Arithmetic) (p : Metadata) (h : InputContract p)
    (a : Direct.Allocation p) (cs : List (Fin 9)) (chosen : Fin 9) (heap : Heap) : Except InputError Result := do
  checkInputs p
  pure (result p a actual (finalState rest p h a cs chosen heap) heap)

theorem output_byte (p : Metadata) (a : Direct.Allocation p) (actual : Arithmetic)
    (s : State Expr) (heap : Heap) (off : Nat) (ho : off<(p.tensor .a).numel) (b : Fin 4) :
    outputHeap p a actual s heap (a.pointer+4*off+b.val)=byte (interpret actual (graph p s off)) b.val := by
  have bound := b.isLt
  have inside : a.pointer≤a.pointer+4*off+b.val ∧ a.pointer+4*off+b.val<a.pointer+4*(p.tensor .a).numel := by omega
  rw [outputHeap,if_pos inside]
  have diff : a.pointer+4*off+b.val-a.pointer=4*off+b.val := by omega
  rw [diff]
  have div : (4*off+b.val)/4=off := by omega
  have mod : (4*off+b.val)%4=b.val := by omega
  rw [div,mod]

theorem output_frame (p : Metadata) (a : Direct.Allocation p) (actual : Arithmetic)
    (s : State Expr) (heap : Heap) (address : Nat)
    (outside : ¬(a.pointer≤address ∧ address<a.pointer+4*(p.tensor .a).numel)) :
    outputHeap p a actual s heap address=heap address := by simp [outputHeap,outside]

#axiomsClean source_one
#axiomsClean one_readback
#axiomsClean certified
#axiomsClean execute_good
#axiomsClean execute_frame
#axiomsClean output_byte
#axiomsClean output_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.DirectFP
