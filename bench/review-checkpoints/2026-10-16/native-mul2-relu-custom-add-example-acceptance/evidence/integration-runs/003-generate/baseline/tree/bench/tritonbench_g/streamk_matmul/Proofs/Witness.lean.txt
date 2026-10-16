import bench.tritonbench_g.streamk_matmul.SourceCorrect
namespace VeriTile.Bench.TritonBenchG.StreamK
open VeriTile.Triton
set_option maxRecDepth 32000
set_option maxHeartbeats 20000000

def witnessBases : Region→Address
 | .a=>(0,0) | .b=>(1,0) | .c=>(2,0) | .locks=>(3,0)
def splitHost : HostInput := ⟨1,1,3,1,1,1,3,true⟩
def splitConfig : Config := hostConfig splitHost witnessBases

def splitMemory : Memory Word where
 data:=fun a=>if a.1=0 then
    if a.2=0 then ⟨0xcb800000⟩ else if a.2=1 then ⟨0x3f800000⟩ else ⟨0x4b800000⟩
   else if a.1=1 then ⟨0x3f800000⟩ else ⟨0x42b60000⟩
 locks:=fun _=>0

def schedule01 : List Nat := List.replicate 55 2++List.replicate 55 0++List.replicate 55 1
def schedule10 : List Nat := List.replicate 55 2++List.replicate 55 1++List.replicate 55 0

def splitResult01 := firstRun splitConfig software schedule01 splitMemory
def splitResult10 := firstRun splitConfig software schedule10 splitMemory
theorem source_atomic_order_changes_output :
 allDone 3 splitResult01.2.control=true ∧ allDone 3 splitResult10.2.control=true ∧
 splitResult01.2.mem.data (2,0)=⟨0x3f800000⟩ ∧ splitResult10.2.mem.data (2,0)=⟨0⟩ ∧
 splitResult01.2.mem.data (9,73)=splitMemory.data (9,73) := by decide +kernel

def blockedRun := firstRun splitConfig software (List.replicate 55 0) splitMemory

theorem failed_polls_do_not_finish_or_publish :
 allDone 3 blockedRun.2.control=false ∧
 blockedRun.2.mem.data (2,0)=splitMemory.data (2,0) ∧
 blockedRun.2.mem.locks (3,0)=0 ∧
 Event.cas (3,0) 1 1∈blockedRun.1 ∧
 (hostRun splitConfig software (List.replicate 55 0) [] splitMemory).isNone=true := by
 decide +kernel

def multiHost : HostInput := ⟨5,1,3,1,1,1,2,true⟩
def multiConfig := hostConfig multiHost witnessBases
def multiMemory : Memory Word where
 data:=fun a=>if a.1=0 || a.1=1 then ⟨0x3f800000⟩ else ⟨0x42b60000⟩
 locks:=fun _=>0
def multiFirstSchedule : List Nat := List.replicate 150 1++List.replicate 150 0
def multiSecondSchedule : List Nat := List.replicate 100 0++List.replicate 100 1
def multiFirst := firstRun multiConfig software multiFirstSchedule multiMemory
def multiSecond := secondRun multiConfig software multiSecondSchedule multiFirst.2

theorem complete_nonzero_two_wave_execution :
 (hostPlan multiHost).streamTiles=3 ∧ (hostPlan multiHost).blockingTiles=2 ∧
 allDone 2 multiFirst.2.control=true ∧ allDone 2 multiSecond.2.control=true ∧
 (hostRun multiConfig software multiFirstSchedule multiSecondSchedule multiMemory).isSome=true ∧
 (List.range 5).all (fun i=>(multiSecond.2.mem.data (2,Int.ofNat i)).bits==0x40400000)=true ∧
 multiSecond.2.mem.data (9,73)=multiMemory.data (9,73) := by decide +kernel

def halfHost : HostInput := ⟨1,1,1,1,1,1,0,true⟩
def halfConfig := hostConfig halfHost witnessBases
def halfMemory : Memory Word where
 data:=fun a=>if a.1=0 then ⟨0x3f800001⟩ else ⟨0x3f800000⟩
 locks:=fun _=>0
def halfResult := run halfConfig software halfConfig.fullGrid (List.replicate 60 0)
 (initial originalSource.fullWave software halfMemory)
theorem complete_full_wave_loses_fp32_bits :
 allDone 1 halfResult.2.control=true ∧
 halfResult.2.mem.data (2,0)=⟨0x3f800000⟩ ∧
 halfResult.2.mem.data (2,0)≠halfMemory.data (0,0) := by decide +kernel

/-- BLOCK=32 is a native-compilable shape in the archived original kernels.
The host accepts M=N=K=1; the source has no padding mask and reads A[1]. -/
def tailHost : HostInput := ⟨1,1,1,32,32,32,0,true⟩
def tailConfig := hostConfig tailHost witnessBases
def tailEvents := controlTrace originalSource.fullWave tailConfig tailConfig.fullGrid
 (List.replicate 24 0) (fun _=>0)
theorem accepted_host_reads_outside_A_allocation :
 Event.load 0 "ft.a" 1 (0,1)∈tailEvents ∧ tailHost.m*tailHost.k=1 := by decide +kernel

#axiomsClean source_atomic_order_changes_output
#axiomsClean failed_polls_do_not_finish_or_publish
#axiomsClean complete_nonzero_two_wave_execution
#axiomsClean complete_full_wave_loses_fp32_bits
#axiomsClean accepted_host_reads_outside_A_allocation
end VeriTile.Bench.TritonBenchG.StreamK
