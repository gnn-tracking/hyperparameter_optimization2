#!/usr/bin/env xonsh

$XONSH_SHOW_TRACEBACK = True
$RAISE_SUBPROC_ERROR = True

local_bin = p"~/.local/bin".expanduser()
assert local_bin.is_dir()

scripts = p`bin/.*\.xsh`
print(f"Installing {len(scripts)} scripts")

for script in scripts:
    link = local_bin / script.name
    ln -s @(script) @(link)
