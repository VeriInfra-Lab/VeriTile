import bench.tritonbench_g.chunk_bwd_dqkg.Target

/-! Independent native-word observation of all three logical results. The
formula evaluates the source-ordered numerical recurrence on logical inputs;
it does not inspect a source execution or a memory state. Arbitrary symbolic
input terms remain admitted. -/
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.NativeTarget
open VeriTile Triton FP.Structural Whole KernelNativeCarrier
noncomputable section

def words (isa : Instructions) (route : Routing) (scale : Word) (c : Config)
    (vals : ChannelValues Node (IO.nativeInputs c)) :
    ChannelValues Word (IO.nativeOutputs c) :=
  fun o j => mapValue (Node.eval isa route) (IO.nativeOutputs c)[o].dtype
    (IO.nativeResult c (Target.native isa route scale c (IO.nativeValues c vals)) o j)

#specNonCircular words avoiding [State,run,FP.Structural.exec,KernelNative.runQuoted,
  Typed.runGrid,runSchedule,IO.nativeIO,IO.nativeJob]
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.NativeTarget
