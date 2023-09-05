#!/usr/bin/env xonsh

logs=$(ls slurm_logs/)
last_log=sorted(logs.split())[-1]
print(f"Last log is {last_log}")
cat slurm_logs/@(last_log)
