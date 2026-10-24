from pathlib import Path
W=Path(__file__).resolve().parent;T=W.parent/'add-value-finite-acceptance-revision1-wip/after/bench/tritonbench_g/add_value';A=W/'after/bench/tritonbench_g/sin_kernel'
for name in ['NativeArithmetic','NativeMemory','NativeProgram','NativeGrid']:
 s=(T/'Proofs'/(name+'.lean')).read_text().replace('bench.tritonbench_g.add_value','bench.tritonbench_g.sin_kernel').replace('AddValueNative','SinKernelNative').replace('Word (.storage d)','Word d').replace('wideDType d','match d with | .fp32 => .fp32 | .fp64 => .fp64')
 s=s.replace('result b d value argument','result b d value').replace('(source d).eval b value argument','(source d).eval b value').replace('source_scalar b d value argument','source_scalar b d value')
 if name=='NativeMemory':
  s=s.replace('  In particular bool-pointer stores pack the raw signed i8 cell; they do not\n  normalize it to0/1. There is no floating arithmetic in this primitive.','  There is no floating arithmetic or rounding conversion in this primitive.')
 if name=='NativeProgram':s=s.replace('including all raw boolean bytes','including every finite floating bit pattern')
 if name=='NativeGrid':
  # Every complete program schedule is represented, not just ascending pids.
  s=s.replace('def launchPlan','structure Schedule (n : Nat) where\n  pids : List Nat\n  complete : pids.Perm (List.range (grid n))\n\ndef ascendingSchedule (n : Nat) : Schedule n := ⟨List.range (grid n),List.Perm.refl _⟩\n\ndef launchPlan',1)
  start=s.index('def launchPlan')
  head=s[:start];tail=s[start:]
  tail=tail.replace('(p : Layout) (n : Nat)', '(p : Layout) (n : Nat) (schedule : Schedule n)')
  tail=tail.replace('(List.range (grid n))','schedule.pids')
  tail=tail.replace('launchPlan d ops b p n memory','launchPlan d ops b p n schedule memory').replace('executeLaunch d ops b p n memory','executeLaunch d ops b p n schedule memory')
  tail=tail.replace('launch_correspondence d ops b p n memory','launch_correspondence d ops b p n schedule memory').replace('launch_write_bounds d ops b p n memory','launch_write_bounds d ops b p n schedule memory')
  tail=tail.replace('(p : Layout) (memory : M) : executeLaunch d ops b p (2^31) memory', '(p : Layout) (schedule : Schedule (2^31)) (memory : M) : executeLaunch d ops b p (2^31) schedule memory')
  s=head+tail
 (A/'Proofs'/(name+'.lean')).write_text(s)
print(A)
