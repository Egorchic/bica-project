
import base64
import json
import shutil
import subprocess
from pathlib import Path


def run_powershell(script: str) -> dict:
    encoded = base64.b64encode(
        script.encode("utf-16-le")
    ).decode("ascii")

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            encoded
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return json.loads(result.stdout)


def get_system_info():
    script = r"""
    $ErrorActionPreference = "Stop"
    [Console]::OutputEncoding =
        New-Object System.Text.UTF8Encoding($false)

    $os = Get-CimInstance Win32_OperatingSystem
    $system = Get-CimInstance Win32_ComputerSystem

    $cpu = @(
        Get-CimInstance Win32_Processor |
        Select-Object Name,
                      NumberOfCores,
                      NumberOfLogicalProcessors,
                      MaxClockSpeed,
                      CurrentClockSpeed
    )

    $ram = @(
        Get-CimInstance Win32_PhysicalMemory |
        Select-Object Capacity,
                      Speed,
                      ConfiguredClockSpeed,
                      Manufacturer,
                      PartNumber
    )

    $gpu = @(
        Get-CimInstance Win32_VideoController |
        Select-Object Name,
                      AdapterRAM,
                      DriverVersion,
                      VideoProcessor
    )

    $disks = @(
        Get-CimInstance Win32_LogicalDisk |
        Where-Object DriveType -eq 3 |
        Select-Object DeviceID, Size, FreeSpace
    )

    @{
        OS = @{
            Name = $os.Caption
            Version = $os.Version
            Architecture = $os.OSArchitecture
        }
        CPU = $cpu
        RAM = @{
            TotalBytes = $system.TotalPhysicalMemory
            AvailableKB = $os.FreePhysicalMemory
            Modules = $ram
        }
        GPU = $gpu
        Disks = $disks
    } | ConvertTo-Json -Depth 6 -Compress
    """

    return run_powershell(script)


def get_nvidia_info():
    executable = shutil.which("nvidia-smi")

    if not executable:
        candidate = Path(
            r"C:\Program Files\NVIDIA Corporation"
            r"\NVSMI\nvidia-smi.exe"
        )
        if candidate.exists():
            executable = str(candidate)

    if not executable:
        return "nvidia-smi не найден"

    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu="
                "name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=15
        )

        if result.returncode != 0:
            return result.stderr.strip()

        return result.stdout.strip()

    except Exception as e:
        return str(e)


def main():
    print("Сбор характеристик системы...\n")

    info = get_system_info()

    print("=== ХАРАКТЕРИСТИКИ КОМПЬЮТЕРА ===")
    print(json.dumps(
        info,
        indent=2,
        ensure_ascii=False
    ))

    print("\n=== NVIDIA GPU ===")
    print("Поля: модель, VRAM (МБ), свободно (МБ), драйвер")
    print(get_nvidia_info())


if __name__ == "__main__":
    main()
