import subprocess
from pathlib import Path

script_path = Path.cwd() / "get_sky_brightness_at_nearest_astro_twilight.py"
sites = {(43.05148, -78), (-30.2446, -70.7494), (31.9583, -111.5967)}

if __name__ == "__main__":
    for i, (lat, lon) in enumerate(sites):
        print(f"> processing site {i+1}")
        subprocess.run(["python3", script_path, "--lat", str(lat), "--lon", str(lon)])
