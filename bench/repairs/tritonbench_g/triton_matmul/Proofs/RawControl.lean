import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawDecomposition

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
open VeriTile.Triton FP.Structural FPValue
noncomputable section

/-- The signed control expressions actually evaluated by the pasted source.
Their relation to nonnegative geometry is proved separately. -/
def signedCeil (n block : Nat) : Int :=
  (n : Int).tdiv block + if (n : Int).tmod block = 0 then 0 else 1

structure Control where
  (numM numN inGroup group first size pidM pidN startM startN : Int)

def control {α : Type} (p : Parameters) (s : State α) : Control := Id.run do
  let pid := (s.pids 0 : Int)
  let nm := signedCeil p.M (BM p)
  let nn := signedCeil p.N (BN p)
  let width := (p.GROUP_SIZE_M : Int) * nn
  let group := pid.tdiv width
  let first := group * p.GROUP_SIZE_M
  let size := min (nm - first) (p.GROUP_SIZE_M : Int)
  let pm := first + pid.tmod size
  let pn := (pid.tmod width).tdiv size
  return ⟨nm,nn,width,group,first,size,pm,pn,pm * BM p,pn * BN p⟩

def maskM {α : Type} (p : Parameters) (s : State α) (i : Fin (BM p)) : Bool :=
  decide ((i.val : Int) < (p.M : Int) - (control p s).startM)
def maskN {α : Type} (p : Parameters) (s : State α) (i : Fin (BN p)) : Bool :=
  decide ((i.val : Int) < (p.N : Int) - (control p s).startN)
def offsetM {α : Type} (p : Parameters) (s : State α) (i : Fin (BM p)) : Int :=
  (if maskM p s i then (control p s).startM else 0) +
    (if maskM p s i then (i.val : Int) else 0)
def offsetN {α : Type} (p : Parameters) (s : State α) (i : Fin (BN p)) : Int :=
  (if maskN p s i then (control p s).startN else 0) +
    (if maskN p s i then (i.val : Int) else 0)

structure Context {α : Type} (p : Parameters) (initial current : State α) : Prop where
  memory : current.mem = initial.mem
  dimM : current.regs .int [] "M" = some (fun _ => (p.M : Int))
  dimN : current.regs .int [] "N" = some (fun _ => (p.N : Int))
  dimK : current.regs .int [] "K" = some (fun _ => (p.K : Int))
  strideAM : current.regs .int [] "stride_am" = some (fun _ => (p.stride_am : Int))
  strideAK : current.regs .int [] "stride_ak" = some (fun _ => (p.stride_ak : Int))
  strideBK : current.regs .int [] "stride_bk" = some (fun _ => (p.stride_bk : Int))
  strideBN : current.regs .int [] "stride_bn" = some (fun _ => (p.stride_bn : Int))
  strideCM : current.regs .int [] "stride_cm" = some (fun _ => (p.stride_cm : Int))
  strideCN : current.regs .int [] "stride_cn" = some (fun _ => (p.stride_cn : Int))
  blocks : current.regs .int [] "blocks_k" = some (fun _ => signedCeil p.K (BK p))
  offk : current.regs .int [BK p] "offs_k" = some (fun i => (i.1.val : Int))
  am : current.regs .int [BM p] "offs_am" = some (fun i => offsetM p initial i.1)
  bn : current.regs .int [BN p] "offs_bn" = some (fun i => offsetN p initial i.1)
  mm : current.regs .bool [BM p] "mask_m" = some (fun i => maskM p initial i.1)
  nn : current.regs .bool [BN p] "mask_n" = some (fun i => maskN p initial i.1)

end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
