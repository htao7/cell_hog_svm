@echo off
setlocal EnableDelayedExpansion
set i=0
cd training
for %%a in (*.tif) do (
    ren "%%a" "!i!.new"
    set /a i+=1
)
ren *.new *.tif