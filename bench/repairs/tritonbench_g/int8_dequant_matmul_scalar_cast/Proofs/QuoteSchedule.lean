import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.QuoteValue
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.QuoteDescriptors

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
open VeriTile.Triton FP.Structural
open IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def physicalLane (p : PhysicalCommit s c) : TileIndex [c.BM,c.BN] :=
  p.lane

/-- A schedule consumes the actual quotation's store descriptors. The full
quotation is evaluated to obtain its final value, pointer and mask registers.
`fullResult.memory` proves these same descriptors are the source epilogue's
store/atomic fold. This permits arbitrary interleavings of distinct programs'
active store lanes, including different FP16 atomic orders at different cells. -/
def sourceCommit (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (b : Buffers) (divisor : F)
    (p : PhysicalCommit s c) (m : Memory F H) : Option (Memory F H) := do
  let final←execute (floating q) q s.M s.N (source s c bias)
    (initialState q b m divisor p.pid.val p.split.val)
  let i:=physicalLane p
  let pointer:=register final .ptr [c.BM,c.BN] "C" i
  let enabled:=register final .bool [c.BM,c.BN] "mask" i
  if pointer.1="C" ∧ enabled then
    let address:=(b.out,b.outOffset+pointer.2)
    let output:=asH q.ops (register final .fp16 [c.BM,c.BN] "out" i)
    return write16 m address (if c.S=1 then output else q.ops.hadd (read16 q.ops m address) output)
  else return m

def sourceGrid (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (b : Buffers) (divisor : F) :
    List (PhysicalCommit s c)→Memory F H→Option (Memory F H)
  | [],m=>some m
  | p::ps,m=>do sourceGrid q s c bias b divisor ps (←sourceCommit q s c bias b divisor p m)

theorem physical_active (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (b : Buffers) (m : Memory F H) (divisor : F)
    (p : PhysicalCommit s c) :
    let base:=(preludeExecution (floating q) q s c bias
      (initialState q b m divisor p.pid.val p.split.val)).val
    active s c base (physicalLane p) ∧
      offset s c base (physicalLane p)=p.logical.1.1.val*s.N+p.logical.1.2.val := by
  let initial:=initialState q b m divisor p.pid.val p.split.val
  have valid : initial.pids 0<Geometry.count (geometry s c) := by simpa [initial,initialState] using p.pid.isLt
  have rm:=prelude_rm (floating q) q s c bias initial valid
  have rn:=prelude_rn (floating q) q s c bias initial valid
  dsimp only
  unfold active offset
  rw [rm,rn]
  simp only [initial,initialState,State.setReg_pids,↓reduceIte,physicalLane]
  have hr : row s c p.pid.val p.lane.1<s.M := p.active.1
  have hc : col s c p.pid.val p.lane.2.1<s.N := p.active.2
  constructor
  · exact ⟨Int.ofNat_lt.mpr hr,Int.ofNat_lt.mpr hc⟩
  · simp only [physicalLane,Prod.fst,Prod.snd,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
    rfl

theorem sourceCommit_eq (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (b : Buffers) (divisor : F)
    (selected : Selection s c) (plans : q.plans=selectedPlan s c selected)
    (p : PhysicalCommit s c) (m : Memory F H) :
    sourceCommit q s c bias b divisor p m=some (commit q.ops s c b
      (partialValue q.ops s c b selected divisor bias m p.logical.1.1 p.logical.1.2 p.split) p.logical m) := by
  let initial:=initialState q b m divisor p.pid.val p.split.val
  have hd : initial.regs .fp32 [] "divfactor"=some (fun _=>.inl divisor) := by simp [initial,initialState]
  obtain ⟨final,hf,result⟩:=fullResult q s c bias initial divisor hd
  dsimp only [initial] at result
  have enabled:=physical_active q s c bias b m divisor p
  have hv:=source_value q s c bias b m divisor selected plans p.pid.val p.split.val p.pid.isLt p.split.isLt
    (physicalLane p) p.active.1 p.active.2
  unfold sourceCommit
  rw [hf]
  simp only [bind,pure,Option.bind_some,result.address,result.mask,result.out,Prod.fst,Prod.snd,
    enabled.1,decide_true,enabled.2,hv,asH,ite_true,and_self,true_and]
  unfold commit outputAddress
  simp only [Nat.add_assoc]
  rfl

theorem sourceGrid_eq (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (b : Buffers) (divisor : F)
    (selected : Selection s c) (plans : q.plans=selectedPlan s c selected)
    (events : List (PhysicalCommit s c)) (m : Memory F H) :
    sourceGrid q s c bias b divisor events m=
      some (gridRun q.ops s c b selected divisor bias (events.map PhysicalCommit.logical) m) := by
  induction events generalizing m with
  | nil=>rfl
  | cons p ps ih=>
    rw [sourceGrid,sourceCommit_eq q s c bias b divisor selected plans p m]
    simpa only [bind,pure,Option.bind_some,List.map_cons,gridRun] using ih _

#axiomsClean physical_active
#axiomsClean sourceCommit_eq
#axiomsClean sourceGrid_eq
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
