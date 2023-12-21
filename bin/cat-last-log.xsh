#!/usr/bin/env xonsh

# Cat the last SLURM log

$XONSH_SHOW_TRACEBACK = True
$RAISE_SUBPROC_ERROR = True

logs=$(ls slurm_logs/)
last_log=sorted(logs.split())[-1]
print(f"Last log is {last_log}")
cat slurm_logs/@(last_log)
