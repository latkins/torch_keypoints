import io

import requests
from tqdm.auto import tqdm


def stream(source, dest: io.BufferedReader):
    r = requests.get(source, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    chunk_size = 10 * (1024 * 1024)
    with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
        for chunk in r.iter_content(chunk_size):
            if chunk:
                dest.write(chunk)
                pbar.update(chunk_size)
