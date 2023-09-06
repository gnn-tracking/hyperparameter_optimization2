#!/usr/bin/env xonsh

$XONSH_SHOW_TRACEBACK = True
$RAISE_SUBPROC_ERROR = True

search_str=$ARG1
results=$(find .  -path  ./lightning_logs/*@(search_str)*/hparams.yaml).strip()

print(results)

if len(results.split()) >= 2:
    raise ValueError("More than one result found")

cat @(results.strip()) | less
