import bench.tritonbench_g.uniform_sampling.Proofs.KernelStores

namespace VeriTile.Bench.Repairs.UniformSampling.KernelExecution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.UniformSampling
noncomputable section
set_option maxHeartbeats 3500000
set_option maxRecDepth 20000

def fieldOf (f : Fin 4) : Random.Direct.Field :=
  match f.val with | 0 => .c0 | 1 => .c1 | 2 => .c2 | _ => .c3

def entry {α : Type} (pid : Nat) (s : State α) : State α :=
  { s with pids := fun ax => if ax=0 then pid else 0 }

def words {α : Type} (seed offset : Int) (B : Nat) (s : State α)
    (f : Random.Direct.Field) : TileIndex [B] → Random.Word :=
  fun i => ScalarSpec.word seed offset (KernelPrefix.indices B s i) f

def affineState {α : Type} (M : Algebra α) (fused : α → α → α → α)
    (seed offset : Int) (lower upper : Float32Bits) (B : Nat) (s : State α) : State α :=
  KernelAffine.result M fused (words seed offset B s) lower upper
    (KernelPrefix.result M seed offset lower upper B s)

def offsetState {α : Type} (M : Algebra α) (fused : α → α → α → α)
    (seed offset : Int) (lower upper : Float32Bits) (B : Nat) (s : State α) : State α :=
  KernelOffsets.result (s.pids 0) B (affineState M fused seed offset lower upper B s)

def values {α : Type} (M : Algebra α) (fused : α → α → α → α)
    (seed offset : Int) (lower upper : Float32Bits) (B : Nat) (s : State α) (f : Fin 4) :
    Values α .real [B] := KernelAffine.values M fused (words seed offset B s (fieldOf f)) lower upper

def result {α : Type} (M : Algebra α) (fused : α → α → α → α) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : Float32Bits) (B : Nat) (s : State α) : State α :=
  KernelStores.result y N (s.pids 0) B
    (fun f i => M.cast (some .fp32) .real .real (values M fused seed offset lower upper B s f i))
    (offsetState M fused seed offset lower upper B s)

theorem affine_exec {α : Type} [Inhabited α] (M : Algebra α) (fused : α → α → α → α)
    (hf : M.fma .fp32 = some fused) (seed offset : Int) (lower upper : Float32Bits)
    (B : Nat) (s : State α) :
    run M (KernelAffine.body B) (KernelPrefix.result M seed offset lower upper B s) =
      some (affineState M fused seed offset lower upper B s) := by
  apply KernelAffine.exec M fused hf (words seed offset B s) lower upper
  · exact TypedTuple.output_register seed offset (KernelPrefix.indices B s) _
  all_goals simp [KernelPrefix.result,TypedTuple.outputState,TypedTuple.fieldState,
    TupleExecution.fieldName,TypedTuple.capturedState,KernelPrefix.beforeTuple,
    KernelPrefix.preparedState,KernelPrefix.wrappedState,KernelPrefix.indexState,
    KernelPrefix.widened,TypedBindings.counterState,TypedBindings.lowState,
    TypedBindings.boundState,KernelPrefix.scalarState]

theorem offset_exec {α : Type} [Inhabited α] (M : Algebra α) (fused : α → α → α → α)
    (seed offset : Int) (lower upper : Float32Bits) (B : Nat) (s : State α) :
    run M (KernelOffsets.body B) (affineState M fused seed offset lower upper B s) =
      some (offsetState M fused seed offset lower upper B s) := by
  apply KernelOffsets.exec M (s.pids 0) B
  all_goals simp [affineState,KernelAffine.result,KernelPrefix.result,
    TypedTuple.outputState,TypedTuple.fieldState,TupleExecution.fieldName,
    TypedTuple.capturedState,KernelPrefix.beforeTuple,KernelPrefix.preparedState,
    KernelPrefix.wrappedState,KernelPrefix.indexState,KernelPrefix.widened,
    TypedBindings.counterState,TypedBindings.lowState,TypedBindings.boundState,KernelPrefix.scalarState]

theorem value_register {α : Type} (M : Algebra α) (fused : α → α → α → α)
    (seed offset : Int) (lower upper : Float32Bits) (B : Nat) (s : State α) (f : Fin 4) :
    (offsetState M fused seed offset lower upper B s).regs .real [B] (KernelStores.fieldName f) =
      some (values M fused seed offset lower upper B s f) := by
  have hv := KernelAffine.result_register M fused (words seed offset B s) lower upper
    (KernelPrefix.result M seed offset lower upper B s) (fieldOf f)
  fin_cases f <;> simpa [offsetState,KernelOffsets.result,KernelStores.fieldName,
    affineState,values,fieldOf,TupleExecution.fieldName] using hv

theorem offset_register {α : Type} (M : Algebra α) (fused : α → α → α → α)
    (seed offset : Int) (lower upper : Float32Bits) (B : Nat) (s : State α) (f : Fin 4) :
    (offsetState M fused seed offset lower upper B s).regs .int [B] (KernelStores.offsetName f) =
      some (KernelOffsets.offsets (s.pids 0) B f.val) := by
  fin_cases f <;> simp [offsetState,KernelOffsets.result,KernelStores.offsetName]

theorem source_split (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : Float32Bits) (B : Nat) (hB : Random.ArangeSizeDomain B) :
    (KernelSource.source swapped y N seed offset lower upper B hB).surfaceBody =
      (((KernelSource.source swapped y N seed offset lower upper B hB).surfaceBody.take 15 ++
        KernelAffine.body B) ++ KernelOffsets.body B) ++ KernelStores.body y N B := by
  cases swapped <;> rfl

theorem executes {α : Type} [Inhabited α] (M : Algebra α) (fused : α → α → α → α)
    (hf : M.fma .fp32 = some fused) (swapped : Bool) (y : RegionName) (N : Nat)
    (seed offset : Int) (lower upper : Float32Bits) (B : Nat) (hB : Random.ArangeSizeDomain B)
    (s : State α) (hseed : Random.IntegerType.i64.inRange seed = Bool.true) :
    FP.Structural.exec M (KernelSource.source swapped y N seed offset lower upper B hB) s =
      some (result M fused y N seed offset lower upper B s) := by
  change run M (KernelSource.source swapped y N seed offset lower upper B hB).surfaceBody s = _
  rw [source_split,run_append,run_append,run_append,
    KernelPrefix.prefix_exec M swapped y N seed offset lower upper B hB s hseed,
    Option.bind_some,affine_exec M fused hf,Option.bind_some,offset_exec,Option.bind_some]
  exact KernelStores.exec M y N (s.pids 0) B (values M fused seed offset lower upper B s) _
    (value_register M fused seed offset lower upper B s) (offset_register M fused seed offset lower upper B s)

theorem readback {α : Type} (M : Algebra α) (fused : α → α → α → α)
    (y : RegionName) (N : Nat) (seed offset : Int) (lower upper : Float32Bits)
    (B : Nat) (hB : 0<B) (s : State α) (f : Fin 4) (i : Fin B)
    (active : KernelOffsets.output (s.pids 0) B f.val i.val < N) :
    (result M fused y N seed offset lower upper B s).mem y (KernelOffsets.output (s.pids 0) B f.val i.val) =
      .mk .real (M.cast (some .fp32) .real .real (values M fused seed offset lower upper B s f (i,PUnit.unit))) :=
  KernelStores.readback y N (s.pids 0) B hB _ _ f i active

theorem frame {α : Type} (M : Algebra α) (fused : α → α → α → α)
    (y : RegionName) (N : Nat) (seed offset : Int) (lower upper : Float32Bits)
    (B : Nat) (s : State α) (r : RegionName) (a : Nat)
    (outside : KernelStores.Outside y N (s.pids 0) B r a) :
    (result M fused y N seed offset lower upper B s).mem r a = s.mem r a :=
  KernelStores.frame y N (s.pids 0) B _ _ r a outside

#axiomsClean executes
#axiomsClean readback
#axiomsClean frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.KernelExecution
