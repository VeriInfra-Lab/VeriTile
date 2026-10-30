import bench.repairs.tritonbench_g.attention_kernel.Proofs.FamilyNativeStaging
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole002.NativeRouting
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionKernel.Whole001
open NativePlan
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

/-- The entire public launch-configuration domain, including both storage
precisions. This finite data does not contain numerical values or graph nodes. -/
structure Configuration where
  bm : Width
  bn : Width
  dim : Width
  eightWarps : Bool
  stage : Fin 4
  storage : TypedSource.Storage

def Configuration.numWarps (c : Configuration) : Nat := if c.eightWarps then 8 else 4

def Configuration.Matches (c : Configuration) (p : Source.Parameters) (abi : Whole.InputABI p) : Prop :=
  c.bm.value=p.bm ∧ c.bn.value=p.bn ∧ c.dim.value=p.dim ∧
  c.numWarps=abi.numWarps ∧ c.stage.val+1=abi.numStages ∧ c.storage=p.storage

theorem width_exists (n : Nat) (h : n∈[16,32,64,128]) : ∃ w : Width,w.value=n := by
  simp only [List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with h|h|h|h
  · exact ⟨.n16,h.symm⟩
  · exact ⟨.n32,h.symm⟩
  · exact ⟨.n64,h.symm⟩
  · exact ⟨.n128,h.symm⟩

theorem configuration_exists (p : Source.Parameters) (abi : Whole.InputABI p) :
    ∃ c : Configuration,c.Matches p abi := by
  obtain ⟨bm,hm⟩:=width_exists p.bm abi.blocks.1
  obtain ⟨bn,hn⟩:=width_exists p.bn abi.blocks.2.1
  obtain ⟨dim,hd⟩:=width_exists p.dim abi.blocks.2.2
  have hs:=abi.launchConfig.2
  let stage : Fin 4:=⟨abi.numStages-1,by omega⟩
  have hw:=abi.launchConfig.1
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hw
  rcases hw with hw|hw
  · exact ⟨⟨bm,bn,dim,Bool.false,stage,p.storage⟩,hm,hn,hd,hw.symm,by dsimp [stage];omega,rfl⟩
  · exact ⟨⟨bm,bn,dim,Bool.true,stage,p.storage⟩,hm,hn,hd,hw.symm,by dsimp [stage];omega,rfl⟩

/-- Integer-only decoded compiler routing metadata. A producer is identified
by a logical coordinate/owner, never by a floating-point value or an output.
Shared addresses and ldmatrix addresses are checked against these logical
routes at the ordinary compiler boundary. Stages and JIT alignment may alter
physical addresses while preserving these routes. -/
structure Routes where
  a : Fin 32→Fin 4→Fin 2→Nat×Nat
  b : Fin 32→Fin 2→Fin 2→Nat×Nat
  c : Fin 32→Fin 4→Nat×Nat
  localSource : Fin 8→Fin 4→Fin 32→Nat
  shuffle : Fin 32→Fin 3→Nat
  shared : Fin 8→Nat
  broadcast : Fin 32→Nat
  qkChunk : Fin 8→Fin 16→Nat
  pvChunk : Fin 8→Fin 16→Nat

def mask (i : Fin 3) : Nat := [4,2,1][i.val]

/-- An ordinary local routing contract. Every field is an integer/coordinate
identity. No execution, graph evaluation, floating-point equality, dot result,
reduction result, or final-output equality occurs in this contract. The source
staging itself is the fixed, explicit `FamilyNative.algebra`/`dotModel`. -/
def LocalRouting (width : Width) (columns : WarpColumns) (r : Routes) : Prop :=
  (∀ lane reg half,r.a lane reg half=aCoordinate lane reg half) ∧
  (∀ lane reg half,r.b lane reg half=bCoordinate lane reg half) ∧
  (∀ lane reg,r.c lane reg=cCoordinate lane reg) ∧
  (∀ warp lane j,r.localSource warp lane j=localIndex width columns warp.val lane.val j.val) ∧
  (∀ lane i,r.shuffle lane i=Nat.xor lane.val (mask i)) ∧
  (∀ warp,r.shared warp=warp.val) ∧
  (∀ owner,r.broadcast owner=0) ∧
  (∀ block k,r.qkChunk block k=block.val*16+k.val) ∧
  (∀ block k,r.pvChunk block k=block.val*16+k.val)

def canonicalRoutes (width : Width) (columns : WarpColumns) : Routes :=
  ⟨aCoordinate,bCoordinate,cCoordinate,
    (fun warp lane j=>localIndex width columns warp.val lane.val j.val),
    (fun lane i=>Nat.xor lane.val (mask i)),(fun warp=>warp.val),(fun _=>0),
    (fun block k=>block.val*16+k.val),(fun block k=>block.val*16+k.val)⟩

theorem canonical_routing (width : Width) (columns : WarpColumns) :
    LocalRouting width columns (canonicalRoutes width columns) := by
  simp [LocalRouting,canonicalRoutes]

/-- A selected source-specific family member with its local routing contract.
The caller supplies decoded compiler metadata; this structure is not a claim
that Lean implements or verifies the compiler, assembler, or GPU runtime. -/
structure CompilerPlan (p : Source.Parameters) (abi : Whole.InputABI p) where
  config : Configuration
  reflects : config.Matches p abi
  columns : WarpColumns
  columns_bound : columns.value≤abi.numWarps
  routes : Routes
  routing : LocalRouting config.bn columns routes

def CompilerPlan.selection (c : CompilerPlan p abi) : Selection p abi :=
  ⟨c.config.bn,c.reflects.2.1,c.columns,c.columns_bound⟩

def CompilerPlan.layout (c : CompilerPlan p abi) (scaleWord : BitVec 32) : FamilyNative.Layout :=
  FamilyNative.layout c.selection scaleWord

/-- Every wrapper configuration has a family member. This is a domain
coverage theorem, separate from checking a particular compiler artifact. -/
theorem plan_exists (p : Source.Parameters) (abi : Whole.InputABI p) :
    Nonempty (CompilerPlan p abi) := by
  obtain ⟨cfg,hcfg⟩:=configuration_exists p abi
  have hw : 1≤abi.numWarps := by
    have h:=abi.launchConfig.1
    simp only [List.mem_cons,List.not_mem_nil,or_false] at h
    omega
  exact ⟨⟨cfg,hcfg,.w1,hw,canonicalRoutes cfg.bn .w1,canonical_routing cfg.bn .w1⟩⟩

/-- No selected plan silently drops or duplicates a key position. -/
theorem selected_reduction_coverage (c : CompilerPlan p abi) (owner : Fin 32) :
    (tree c.config.bn c.columns owner).leaves.Perm (List.range p.bn) := by
  rw [←c.reflects.2.1]
  exact reduction_coverage c.config.bn c.columns owner

/-- MMA packing is bijective: two accumulator routes cannot collide. -/
theorem c_injective (l₁ l₂ : Fin 32) (r₁ r₂ : Fin 4)
    (h : cCoordinate l₁ r₁=cCoordinate l₂ r₂) : l₁=l₂ ∧ r₁=r₂ := by
  have hl₁:=l₁.isLt;have hl₂:=l₂.isLt;have hr₁:=r₁.isLt;have hr₂:=r₂.isLt
  have h1:=congrArg Prod.fst h;have h2:=congrArg Prod.snd h
  simp only [cCoordinate] at h1 h2
  constructor <;> apply Fin.ext <;> omega

/-- QK and PV use exactly all K coordinates, in increasing 16-wide chunks. -/
theorem chunk_coverage (w : Width) (k : Fin w.value) :
    k.val/16<w.value/16 ∧ k.val/16*16+k.val%16=k.val := by
  have hk:=k.isLt
  cases w <;> dsimp [Width.value] at * <;> omega

#axiomsClean configuration_exists
#axiomsClean canonical_routing
#axiomsClean plan_exists
#axiomsClean selected_reduction_coverage
#axiomsClean c_injective
#axiomsClean chunk_coverage
#specNonCircular LocalRouting avoiding [Native.Graph.eval,FP.Structural.exec,Whole.answer]
#specNonCircular CompilerPlan avoiding [Native.Graph.eval,FP.Structural.exec,Whole.answer]
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole002.NativeRouting
