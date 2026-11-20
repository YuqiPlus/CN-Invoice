"""Collect media from explicit public GitHub repositories; never execute their code."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZipFile

import httpx


MEDIA = {'.png', '.jpg', '.jpeg', '.webp', '.pdf', '.tif', '.tiff', '.bmp'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repositories', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--archive-limit-mib', type=int, default=40)
    args = parser.parse_args()
    if '.private' not in args.output.resolve().parts:
        parser.error('Private media output required')
    if not 1 <= args.workers <= 12:
        parser.error('workers must be between 1 and 12')
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / 'repositories'
    cache.mkdir(exist_ok=True)
    documents = args.output / 'documents'
    documents.mkdir(exist_ok=True)
    repos = json.loads(args.repositories.read_text())

    def collect(repo):
        name = repo['full_name']
        record = cache / (hashlib.sha256(name.encode()).hexdigest() + '.json')
        if record.exists():
            return json.loads(record.read_text())
        result = {'repo': name, 'files': []}
        try:
            if repo.get('size', 0) > 120000:
                result['skipped'] = 'repository_too_large_for_archive_collection'
            else:
                branch = repo['default_branch']
                url = f'https://codeload.github.com/{name}/zip/refs/heads/{branch}'
                data = bytearray()
                with httpx.Client(timeout=35, follow_redirects=True) as client:
                    with client.stream('GET', url) as response:
                        response.raise_for_status()
                        for chunk in response.iter_bytes():
                            data.extend(chunk)
                            if len(data) > args.archive_limit_mib * 1024**2:
                                raise ValueError('Archive exceeds configured size limit')
                with ZipFile(io.BytesIO(data)) as archive:
                    written = 0
                    for entry in archive.infolist():
                        suffix = Path(entry.filename).suffix.lower()
                        if suffix not in MEDIA or not 10000 <= entry.file_size <= 25 * 1024**2:
                            continue
                        if len(result['files']) >= 2000 or written + entry.file_size > 100 * 1024**2:
                            result['truncated'] = True
                            break
                        payload = archive.read(entry)
                        digest = hashlib.sha256(payload).hexdigest()
                        target = documents / (digest + suffix)
                        target.write_bytes(payload)
                        relative = entry.filename.split('/', 1)[-1]
                        result['files'].append({
                            'repo': name, 'path': relative, 'branch': branch,
                            'url': f'https://raw.githubusercontent.com/{name}/{branch}/{relative}',
                            'local_file': str(target.resolve()), 'sha256': digest,
                            'size': len(payload), 'license': repo.get('license'),
                        })
                        written += len(payload)
        except Exception as exc:
            result['error'] = type(exc).__name__
        record.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(collect, repo) for repo in repos]):
            results.append(future.result())
            if len(results) % 25 == 0:
                print(json.dumps({'repositories': len(results), 'of': len(repos),
                                  'media_files': sum(len(r['files']) for r in results)}), flush=True)
    (args.output / 'discovered.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps({'repositories': len(results), 'media_files': sum(len(r['files']) for r in results),
                      'errors': sum('error' in r for r in results)}), flush=True)


if __name__ == '__main__':
    main()
