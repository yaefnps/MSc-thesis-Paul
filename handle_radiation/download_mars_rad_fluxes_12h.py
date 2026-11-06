import time
from calendar import monthrange
from datetime import datetime, timedelta
from pathlib import Path
import sys

import cdsapi

# Monat auswählen

YEAR = int(sys.argv[1])
MONTH = int(sys.argv[2])

OUTPUT_DIR = (
    Path(f'/jetfs/scratch/psplechtna/thesisdata/radfluxes/hybrid/{YEAR}/{MONTH:02d}'))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEVELS = "/".join(str(level) for level in range(1, 138))

# Gewünschte Zielzeiten: erster Tag 00 UTC bis letzter Tag 23 UTC
START = datetime(YEAR, MONTH, 1, 0)
END = datetime(YEAR, MONTH, monthrange(YEAR, MONTH)[1], 23)

c = cdsapi.Client()

# Für 00-06 UTC am Monatsersten: 18 UTC Lauf des Vortags
init_time = START - timedelta(hours=6)

while init_time + timedelta(hours=1) <= END:
    
    # Schritte 1-12 verwenden, beschränkt auf den gewünschten Monat.
    steps = [
        step
        for step in range(1, 13)
        if START <= init_time + timedelta(hours=step) <= END
    ]
    
    first_time = init_time + timedelta(hours=steps[0])
    last_time = init_time + timedelta(hours=steps[-1])
    
    filename = (
        f"TMTTRAD_{first_time:%Y%m%d%H}"
        f"_{last_time:%Y%m%d%H}.nc"
    )
    target = OUTPUT_DIR / filename
    temporary_target = OUTPUT_DIR / f"{filename}.part"
    
    if target.exists():
        print(f"*** Bereits vorhandene: {target.name}", flush=True)
        
    else:
        print(
            f"\nLauf: {init_time:%Y-%m-%d %H} UTC\n"
            f"Zielzeiten: {first_time:%Y-%m-%d %H} bis "
            f"{last_time:%Y-%m-%d %H} UTC\n"
            f"Schritte: {'/'.join(map(str, steps))}",
            flush=True,
        )
        
        start_time = time.monotonic()
        
        c.retrieve(
            "reanalysis-era5-complete",
            {
                "class":     "ea",
                "expver":    "1",
                "stream":    "oper",
                "type":      "fc",
                "levtype":   "ml",
                "levellist": LEVELS,
                "date":      init_time.strftime("%Y-%m-%d"),
                "time":      init_time.strftime("%H:00:00"),
                "step":      "/".join(map(str, steps)),
                "param":     "235001/235002",
                "area":      "90/-179.5/-90/180",
                "grid":      "0.50/0.50",
                "format":    "netcdf"
            },
            str(temporary_target),
        )
        
        # Endgültiger Name wird erst nach erfolgreichem Download vergeben
        temporary_target.replace(target)
        
        elapsed = time.monotonic() - start_time
        print(f"*** {target.name} saved successfully, "
              f"time: {elapsed:.3f}s / {elapsed / 60:.3f} mins ***",
              flush=True,
        )
        
    # Wechsel zwischen 06- und 18-UTC Lauf
    init_time += timedelta(hours=12)
        
print(f"\nMonat {YEAR}-{MONTH:02d} vollständig heruntergeladen.")
