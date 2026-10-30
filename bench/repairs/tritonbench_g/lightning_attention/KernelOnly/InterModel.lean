import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.IntraModel
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
variable {α : Type} [Inhabited α]
def subK (M : Algebra α) (p : Parameters) (head row : Nat) (m : Memory α) : Matrix α p.CB p.D :=
  fun i => read M m "key" (head*p.N*p.D+(row+i.1.val)*p.D+i.2.1.val) (row+i.1.val<p.N)
def subVT (M : Algebra α) (p : Parameters) (head row : Nat) (m : Memory α) : Matrix α p.E p.CB :=
  fun i => read M m "value" (head*p.N*p.E+(row+i.2.1.val)*p.E+i.1.val) (row+i.2.1.val<p.N)
def subQT (M : Algebra α) (p : Parameters) (head row : Nat) (m : Memory α) : Matrix α p.D p.CB :=
  fun i => read M m "query" (head*p.N*p.D+(row+i.2.1.val)*p.D+i.1.val) (row+i.2.1.val<p.N)
def subG (M : Algebra α) (p : Parameters) (head row : Nat) (m : Memory α) : Matrix α p.CB p.E :=
  fun i => read M m "incoming" (head*p.N*p.E+(row+i.1.val)*p.E+i.2.1.val) (row+i.1.val<p.N)

/-- Uncast gradient loads read the current memory. The intra producer and
prior inter stores therefore remain connected to each later update. -/
def gradientRows (M : Algebra α) (r : RegionName) (N width head row rows : Nat) (m : Memory α) : Matrix α rows width :=
  fun i => (if row+i.1.val<N then (m r (head*N*width+(row+i.1.val)*width+i.2.1.val)).read .real else zero M)
def gradientTrans (M : Algebra α) (r : RegionName) (N width head row rows : Nat) (m : Memory α) : Matrix α width rows :=
  fun i => (if row+i.2.1.val<N then (m r (head*N*width+(row+i.2.1.val)*width+i.1.val)).read .real else zero M)

def queryUpdate (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head row : Nat)
    (carry : Matrix α p.E p.D) (m : Memory α) : Memory α :=
  let contribution := dot D (subG M p head row m) carry
  let previous := gradientRows M "queryGradient" p.N p.D head row p.CB m
  writeRows M "queryGradient" p.N p.D head row p.CB (fun i => add M (contribution i) (previous i)) m
def keyValueUpdate (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head row : Nat)
    (carry : Matrix α p.D p.E) (m : Memory α) : Memory α :=
  let keyContribution := dot D carry (subVT M p head row m)
  let valueContribution := dot D (subK M p head row m) carry
  let previousK := gradientTrans M "keyGradient" p.N p.D head row p.CB m
  let previousV := gradientRows M "valueGradient" p.N p.E head row p.CB m
  writeRows M "valueGradient" p.N p.E head row p.CB (fun i => add M (valueContribution i) (previousV i))
    (writeTrans M "keyGradient" p.N p.D head row p.CB (fun i => add M (keyContribution i) (previousK i)) m)

def queryBlockMemory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat)
    (carry : Matrix α p.E p.D) (m : Memory α) : Memory α :=
  if block=0 then m else foldRange (fun sub mem => queryUpdate M D p head
    ((block*p.NC+sub)*p.CB) carry mem) 0 p.NC 1 m
def keyValueBlockMemory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head reverseBlock : Nat)
    (carry : Matrix α p.D p.E) (m : Memory α) : Memory α :=
  if reverseBlock=0 then m else foldRange (fun sub mem => keyValueUpdate M D p head
    (p.NT*p.BT-(reverseBlock*p.NC+sub+1)*p.CB) carry mem) 0 p.NC 1 m

def queryBlockDelta (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat)
    (input : Memory α) : Matrix α p.E p.D :=
  foldRange (fun sub acc =>
    let row := (block*p.NC+sub)*p.CB
    let update := dot D (subVT M p head row input) (subK M p head row input)
    fun i => add M (acc i) (update i)) 0 p.NC 1 (fun _ => zero M)
def keyValueBlockDelta (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head reverseBlock : Nat)
    (input : Memory α) : Matrix α p.D p.E :=
  foldRange (fun sub acc =>
    let row := p.NT*p.BT-(reverseBlock*p.NC+sub+1)*p.CB
    let update := dot D (subQT M p head row input) (subG M p head row input)
    fun i => add M (acc i) (update i)) 0 p.NC 1 (fun _ => zero M)
def queryCarry (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head : Nat)
    (input : Memory α) : Nat → Matrix α p.E p.D
  | 0 => fun _ => zero M
  | n+1 => fun i => add M (queryCarry M D p head input n i) (queryBlockDelta M D p head n input i)
def keyValueCarry (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head : Nat)
    (input : Memory α) : Nat → Matrix α p.D p.E
  | 0 => fun _ => zero M
  | n+1 => fun i => add M (keyValueCarry M D p head input n i) (keyValueBlockDelta M D p head n input i)

def queryScanMemory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head : Nat)
    (input current : Memory α) : Memory α :=
  foldRange (fun block mem => queryBlockMemory M D p head block (queryCarry M D p head input block) mem) 0 p.NT 1 current
def keyValueScanMemory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head : Nat)
    (input current : Memory α) : Memory α :=
  foldRange (fun block mem => keyValueBlockMemory M D p head block (keyValueCarry M D p head input block) mem) 0 p.NT 1 current
def interMemory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head : Nat)
    (input current : Memory α) : Memory α :=
  keyValueScanMemory M D p head input (queryScanMemory M D p head input current)

end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
