#!/usr/bin/env xonsh

$XONSH_SHOW_TRACEBACK = True

for job_id in $(squeue -u "$USER" -o "%A" -h).split("\n"):
  job_id = job_id.strip()
  if not job_id:
    continue
  print(job_id)
  get-name @(job_id)
