#!/bin/bash
# Resize /dev/shm — RunPod containers run as root, so remount may work.
# vLLM uses /dev/shm for inter-process communication; Docker default is 64MB.
mount -o remount,size=2G /dev/shm 2>/dev/null && echo "[entrypoint] /dev/shm resized to 2G" || echo "[entrypoint] /dev/shm remount failed (non-fatal)"
exec python3 -u /handler.py
