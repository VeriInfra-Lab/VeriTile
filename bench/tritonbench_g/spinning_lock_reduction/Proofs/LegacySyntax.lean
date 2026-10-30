import bench.tritonbench_g.spinning_lock_reduction.SourceReader
import VeriTile.Triton.DSL.BoolTerms

/-! Legacy quotation retained for the frozen program-preservation fixtures.
Canonical source programs use the shared ControlSource frontend. -/
namespace VeriTile.Bench.Repairs.SpinningLock.SourceSyntax
open Lean VeriTile.Triton.DSL SourceReader

syntax:40 tritonExpr:41 " or " tritonExpr:41 : tritonExpr
syntax "while " tritonExpr " { " tritonStmt* " }" : tritonStmt
syntax "pass" : tritonStmt
syntax "spinning_triton " "{" tritonStmt* "}" : term

private def quotedName (n : TSyntax `ident) : TSyntax `term :=
  ⟨Syntax.mkStrLit n.getId.toString⟩

private def readParam (n : TSyntax `ident) : MacroM (TSyntax `term) :=
  match n.getId.toString with
  | "num_sms" => `(Param.numSms)
  | "k" => `(Param.groupK)
  | "M" => `(Param.m)
  | "N" => `(Param.n)
  | "stride_cm" => `(Param.strideM)
  | "stride_cn" => `(Param.strideN)
  | "BM" => `(Param.bm)
  | "BN" => `(Param.bn)
  | "PHASE" => `(Param.phase)
  | _ => Macro.throwErrorAt n "unsupported source parameter"

private partial def readI (e : TSyntax `tritonExpr) : MacroM (TSyntax `term) :=
  match e with
  | `(tritonExpr| ($x:tritonExpr)) => readI x
  | `(tritonExpr| $n:num) => `(IntExpr.literal $n)
  | `(tritonExpr| tl.program_id(0)) => `(IntExpr.pid)
  | `(tritonExpr| tl.cast($x:tritonExpr, tl.int64)) => do
      let x ← readI x; `(IntExpr.signed $x)
  | `(tritonExpr| tl.cast($x:tritonExpr, tl.uint64)) => do
      let x ← readI x; `(IntExpr.unsigned $x)
  | `(tritonExpr| tl.arange(0, $size:ident)) =>
      match size.getId.toString with
      | "BM" => `(IntExpr.row)
      | "BN" => `(IntExpr.col)
      | _ => Macro.throwErrorAt size "only the BM row and BN column ranges are supported"
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => do
      let a ← readI a; let b ← readI b; `(IntExpr.add $a $b)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => do
      let a ← readI a; let b ← readI b; `(IntExpr.mul $a $b)
  | `(tritonExpr| $a:tritonExpr // $b:tritonExpr) => do
      let a ← readI a; let b ← readI b; `(IntExpr.div $a $b)
  | `(tritonExpr| $a:tritonExpr % $b:tritonExpr) => do
      let a ← readI a; let b ← readI b; `(IntExpr.rem $a $b)
  | `(tritonExpr| $x:ident[:, None]) => do
      unless x.getId.toString ∈ ["rm1", "rm"] do
        Macro.throwErrorAt x "row projection requires a row-valued source register"
      do let n := quotedName x; `(IntExpr.variable $n)
  | `(tritonExpr| $x:ident[None, :]) => do
      unless x.getId.toString ∈ ["rn1", "rn"] do
        Macro.throwErrorAt x "column projection requires a column-valued source register"
      do let n := quotedName x; `(IntExpr.variable $n)
  | `(tritonExpr| $x:ident) => do
      if x.getId.toString ∈ ["num_sms", "k", "M", "N", "stride_cm", "stride_cn", "BM", "BN", "PHASE"] then
        let p ← readParam x; `(IntExpr.param $p)
      else do let n := quotedName x; `(IntExpr.variable $n)
  | _ => Macro.throwErrorAt e "unsupported spinning integer expression"

private partial def readB (e : TSyntax `tritonExpr) : MacroM (TSyntax `term) :=
  match e with
  | `(tritonExpr| ($x:tritonExpr)) => readB x
  | `(tritonExpr| ($x:ident < $y:tritonExpr)[:, None]) => do
      unless x.getId.toString == "rm" do Macro.throwErrorAt x "expected row mask projection"
      let x ← `(tritonExpr| $x:ident)
      let a ← readI x; let b ← readI y
      `(BoolExpr.lt $a $b)
  | `(tritonExpr| ($x:ident < $y:tritonExpr)[None, :]) => do
      unless x.getId.toString == "rn" do Macro.throwErrorAt x "expected column mask projection"
      let x ← `(tritonExpr| $x:ident)
      let a ← readI x; let b ← readI y
      `(BoolExpr.lt $a $b)
  | `(tritonExpr| $a:tritonExpr == $b:tritonExpr) => do
      let a ← readI a; let b ← readI b; `(BoolExpr.eq $a $b)
  | `(tritonExpr| $a:tritonExpr != $b:tritonExpr) => do
      let a ← readI a; let b ← readI b; `(BoolExpr.ne $a $b)
  | `(tritonExpr| $a:tritonExpr < $b:tritonExpr) => do
      let a ← readI a; let b ← readI b; `(BoolExpr.lt $a $b)
  | `(tritonExpr| $a:tritonExpr and $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr & $b:tritonExpr) => do
      let a ← readB a; let b ← readB b; `(BoolExpr.and $a $b)
  | `(tritonExpr| $a:tritonExpr or $b:tritonExpr) => do
      let a ← readB a; let b ← readB b; `(BoolExpr.or $a $b)
  | `(tritonExpr| not $a:tritonExpr) => do
      let a ← readB a; `(BoolExpr.not $a)
  | `(tritonExpr| $x:ident) => do let n := quotedName x; `(BoolExpr.variable $n)
  | _ => Macro.throwErrorAt e "unsupported spinning Boolean expression"

private partial def readP (e : TSyntax `tritonExpr) : MacroM (TSyntax `term) :=
  match e with
  | `(tritonExpr| ($x:tritonExpr)) => readP x
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => do
      let a ← readP a; let b ← readI b; `(Pointer.offset $a $b)
  | `(tritonExpr| $x:ident) =>
      match x.getId.toString with
      | "P" => `(Pointer.base Region.partials)
      | "C" => `(Pointer.base Region.output)
      | "locks" => `(Pointer.base Region.locks)
      | _ => do let n := quotedName x; `(Pointer.variable $n)
  | _ => Macro.throwErrorAt e "unsupported spinning pointer expression"

private def readCondition (e : TSyntax `tritonExpr) : MacroM (TSyntax `term) :=
  match e with
  | `(tritonExpr| tl.atomic_cas($p:tritonExpr, $c:tritonExpr, $r:tritonExpr) != $v:tritonExpr) => do
      let p ← readP p; let c ← readI c; let r ← readI r; let v ← readI v
      `(Condition.casNe $p $c $r $v)
  | _ => do let b ← readB e; `(Condition.pure $b)

private partial def readStmt (s : TSyntax `tritonStmt) : MacroM (TSyntax `term) :=
  match s with
  | `(tritonStmt| pass) => `(Statement.pass)
  | `(tritonStmt| return) => `(Statement.returnNow)
  | `(tritonStmt| tl.debug_barrier()) => `(Statement.blockBarrier)
  | `(tritonStmt| tl.static_assert($b:tritonExpr)) => do
      let b ← readB b; `(Statement.staticAssert $b)
  | `(tritonStmt| $x:ident = tl.zeros([$rows:ident, $cols:ident], dtype=tl.float32)) => do
      let r ← readParam rows; let c ← readParam cols
      let n := quotedName x; `(Statement.zeroFP32 $n $r $c)
  | `(tritonStmt| $x:ident = tl.load($p:tritonExpr)) => do
      let p ← readP p; let n := quotedName x; `(Statement.loadFP32 $n $p)
  | `(tritonStmt| $x:ident = $e:tritonExpr) => do
      if x.getId.toString ∈ ["P_", "C_"] then
        let p ← readP e; let n := quotedName x; `(Statement.pointer $n $p)
      else if x.getId.toString ∈ ["is_reducer", "mask"] then
        let b ← readB e; let n := quotedName x; `(Statement.boolean $n $b)
      else
        let i ← readI e; let n := quotedName x; `(Statement.integer $n $i)
  | `(tritonStmt| $x:ident += $e:tritonExpr) => do
      if x.getId.toString == "acc" then
        match e with
        | `(tritonExpr| $v:ident) => do let n := quotedName x; let v := quotedName v; `(Statement.addFP32 $n $v)
        | _ => Macro.throwErrorAt e "FP32 update requires a loaded register"
      else
        let i ← readI e
        let n := quotedName x; `(Statement.integer $n (IntExpr.add (IntExpr.variable $n) $i))
  | `(tritonStmt| tl.store($p:tritonExpr, $x:ident)) => do
      let p ← readP p; let n := quotedName x; `(Statement.storeFP32 $p $n none)
  | `(tritonStmt| tl.store($p:tritonExpr, $x:ident, mask=$b:tritonExpr)) => do
      let p ← readP p; let b ← readB b
      let n := quotedName x; `(Statement.storeFP32 $p $n (some $b))
  | `(tritonStmt| tl.atomic_xchg($p:tritonExpr, $v:tritonExpr)) => do
      let p ← readP p; let v ← readI v; `(Statement.xchg $p $v)
  | `(tritonStmt| for $idx:ident in range($a:num, $b:num) { $body:tritonStmt* }) => do
      let body ← body.mapM readStmt
      let idx := quotedName idx; `(Statement.forRange $idx $a $b [$body,*])
  | `(tritonStmt| while $test:tritonExpr { $body:tritonStmt* }) => do
      let test ← readCondition test; let body ← body.mapM readStmt
      `(Statement.whileLoop $test [$body,*])
  | `(tritonStmt| if $test:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) => do
      let test ← readB test; let yes ← yes.mapM readStmt; let no ← no.mapM readStmt
      `(Statement.branch $test [$yes,*] [$no,*])
  | `(tritonStmt| if $test:tritonExpr { $yes:tritonStmt* }) => do
      let test ← readB test; let yes ← yes.mapM readStmt
      `(Statement.branch $test [$yes,*] [])
  | _ => Macro.throwErrorAt s "unsupported spinning source statement"

macro_rules
  | `(spinning_triton { $body:tritonStmt* }) => do
      let body ← body.mapM readStmt
      `(([$body,*] : List Statement))

end VeriTile.Bench.Repairs.SpinningLock.SourceSyntax
