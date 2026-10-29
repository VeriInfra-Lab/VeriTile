import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.HostTypes

namespace VeriTile.Bench.Repairs.FastCeLoss.Host
open Source API HostStages
open CrossEntropy1.Host (Step)
noncomputable section
set_option maxHeartbeats 4000000

def fail {α σ ρ : Type} (message : String) : Action α σ ρ := fun world => .error message world

def call (stream : Nat) (operation : Operation) (inputs outputs : List Descriptor)
    (scalars : List PythonScalar := []) (integers : List Int := []) : Call :=
  ⟨operation,inputs,outputs,scalars,integers,stream⟩

def baseBuffers (logits labels losses chunks lse : Descriptor) : Buffers :=
  ⟨logits.tensor,labels.tensor,losses.tensor,chunks.tensor,lse.tensor,losses.tensor,losses.tensor,0⟩

def returnedBuffers (result : ForwardReturn) (upstream : Descriptor) : Buffers :=
  ⟨result.context.logits.tensor,result.context.labels.tensor,result.customLoss.tensor,
    result.chunks.tensor,result.context.lse.tensor,result.scalar.tensor,upstream.tensor,
    (upstream.strides.headD 0).toNat⟩

def labelValues {α σ : Type} (labels : Descriptor) (world : World α σ) : Nat → Int :=
  fun row => world.integer (addr labels.tensor row)

def denominatorValue {α σ : Type} (a : Arithmetic α) (scalars : ScalarInterpreter α)
    (args : Arguments) (count : Option Descriptor) (world : World α σ) : α :=
  match args.items with
  | some item => scalars.apply item
  | none => match count with
    | some tensor => a.natural ((world.integer (addr tensor.tensor 0)).toNat)
    | none => a.natural 0

def finishForward {α σ : Type} (a : Arithmetic α) (runtime : Runtime σ) (stream : Nat)
    (s : Shape) (logits labels losses chunks : Descriptor) (label : Nat → Int) : Action α σ Descriptor :=
  if s.chunks=1 then pure chunks else do
    let global ← allocate runtime [s.rows] .fp32 logits.device
    let hb := baseBuffers logits labels losses chunks global
    perform runtime (call stream .logsumexp [chunks] [global]) (lseStage a s hb)
    perform runtime (call stream .addInplace [losses,global] [losses]) (addStage a s hb)
    perform runtime (call stream .maskIgnored [losses,labels] [losses]) (maskStage a s hb label)
    pure global

def countItems {α σ : Type} (runtime : Runtime σ) (stream : Nat) (args : Arguments)
    (logits : Descriptor) (label : Nat → Int) : Action α σ (Option Descriptor) :=
  match args.items with
  | some _ => pure none
  | none => do
    let countTensor ← allocate runtime [] .int64 logits.device
    perform runtime (call stream .countNonignored [args.labels] [countTensor]) id
      (writeInteger countTensor (Source.count args.shape label))
    pure (some countTensor)

/-- Actual public wrapper and custom forward. Every runtime call threads its
native state. Numeric memory is updated only by the fixed scalar/grid meaning. -/
def forward {α σ : Type} (a : Arithmetic α) (scalars : ScalarInterpreter α)
    (runtime : Runtime σ) (stream : Nat) (args : Arguments) : Action α σ ForwardReturn := do
  if args.labels.tensor.shape≠[args.batch,args.sequence] then
    fail "labels.shape assertion"
  else
    let s := args.shape
    let logits ← view runtime args.logits [s.rows,s.vocab]
    let labels ← view runtime args.labels [s.rows]
    let losses ← allocate runtime [s.rows] .fp32 logits.device
    let chunks ← allocate runtime (if s.chunks=1 then [s.rows] else [s.rows,s.chunks]) .fp32 logits.device
    let initial ← read
    let label := labelValues labels initial
    let b := baseBuffers logits labels losses chunks chunks
    let grid := if s.chunks=1 then [s.rows] else [s.rows,s.chunks]
    let kernel := if s.chunks=1 then Kernel.forward else Kernel.chunked
    let warps := if s.chunks=1 then selectedWarps s.block else 32
    if s.chunks=1 ∧ 65536<s.block then
      fail "calculate_settings maximum blocksize"
    else
      perform runtime (call stream (.launch kernel grid s.block warps args.options)
        [logits,labels] [losses,chunks] [args.capArgument,args.scaleArgument]
        [logits.strides.headD 0,(s.vocab : Int),(s.chunks : Int)])
        (forwardGrid a args.options s b label)
      let lse ← finishForward a runtime stream s logits labels losses chunks label
      perform runtime (call stream .saveContext [logits,lse,labels] [])
      let count ← countItems runtime stream args logits label
      let sum ← allocate runtime [] .fp32 logits.device
      perform runtime (call stream .sum [losses] [sum])
        (fun memory => writeRows sum.tensor 1 (fun _ =>
          a.hostSum (fun row : Fin s.rows => a.load (memory (addr losses.tensor row.val)))) memory)
      let scalar ← allocate runtime [] .fp32 logits.device
      let beforeDivision ← read
      let denominator := denominatorValue a scalars args count beforeDivision
      perform runtime (call stream .divide ([sum]++count.toList) [scalar] args.items.toList)
        (fun memory => writeRows scalar.tensor 1 (fun _ => a.hostDiv (a.load (memory (addr sum.tensor 0))) denominator) memory)
      perform runtime (call stream .returnValue [scalar] [])
      pure ⟨scalar,losses,⟨logits,lse,labels,args.options⟩,chunks,sum,count,args⟩

/-- Ordinary autograd for the outer sum/division, followed by the actual custom
backward. The expanded upstream may have stride zero. The exact saved rank-2
object is returned by the custom function; view pullback restores public rank 3. -/
def backward {α σ : Type} (a : Arithmetic α) (scalars : ScalarInterpreter α)
    (runtime : Runtime σ) (stream : Nat) (saved : ForwardReturn) (scalarUpstream : Descriptor) :
    Action α σ WholeReturn := do
  let s := saved.args.shape
  let initial ← read
  let denominator := denominatorValue a scalars saved.args saved.count initial
  let divided ← allocate runtime [] .fp32 saved.scalar.device
  perform runtime (call stream .divideBackward ([scalarUpstream]++saved.count.toList)
    [divided] saved.args.items.toList)
    (fun memory => writeRows divided.tensor 1
      (fun _ => a.upstreamDiv (a.load (memory (addr scalarUpstream.tensor 0))) denominator) memory)
  let expanded ← expand runtime divided s.rows
  perform runtime (call stream .expandSumBackward [divided] [expanded])
  let beforeKernel ← read
  let b := returnedBuffers saved expanded
  let labels := labelValues saved.context.labels beforeKernel
  perform runtime (call stream (.launch .backward [s.rows,s.blocks] 4096 8 saved.context.options)
    [saved.context.logits,expanded,saved.context.lse,saved.context.labels] [saved.context.logits]
    [saved.args.capArgument,saved.args.scaleArgument]
    [saved.context.logits.strides.headD 0,expanded.strides.headD 0,(s.vocab : Int)])
    (backwardGrid a saved.context.options s b labels)
  let gradient ← view runtime saved.context.logits [saved.args.batch,saved.args.sequence,s.vocab]
  let tuple := [some saved.context.logits.tensor,none,none,none]
  perform runtime (call stream .returnValue [saved.context.logits,gradient] [])
  pure ⟨saved,tuple,gradient,expanded⟩

/-- One whole forward/autograd/backward execution. No saved logits, LSE or
precomputed denominator are accepted from outside this run. -/
def whole {α σ : Type} (a : Arithmetic α) (scalars : ScalarInterpreter α)
    (runtime : Runtime σ) (stream : Nat) (args : Arguments) (upstream : Descriptor) : Action α σ WholeReturn := do
  let saved ← forward a scalars runtime stream args
  backward a scalars runtime stream saved upstream

#specNonCircular whole avoiding [Complete.lseGraph,Complete.lossGraph,Complete.scalarGraph,API.gradientGraph]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Host
