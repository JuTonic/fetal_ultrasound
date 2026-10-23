#!/usr/bin/env bash
# Keep the PC's Docker/containerd/project data on D: without relocating running WSL.
set -euo pipefail
if ((EUID != 0)); then echo "Run with sudo" >&2; exit 1; fi
image=/mnt/d/NeuroML/storage.ext4
storage=/srv/neuroml
mkdir -p /mnt/d/NeuroML "$storage"
if [[ ! -f "$image" ]]; then
    truncate -s 64G "$image"
    mkfs.ext4 -F -m 0 "$image"
fi
if ! mountpoint -q "$storage"; then mount -o loop "$image" "$storage"; fi
mkdir -p "$storage/docker" "$storage/containerd" "$storage/project"
chown semyon:semyon "$storage/project"
# No training is running during setup; preserve the original state/configuration.
systemctl stop docker.socket docker.service containerd.service
rsync -aHAX /var/lib/docker/ "$storage/docker/"
rsync -aHAX /var/lib/containerd/ "$storage/containerd/"
python3 - <<'PY'
import json,re
from pathlib import Path
path=Path('/etc/docker/daemon.json')
if path.exists() and not path.with_suffix('.neuroml-backup.json').exists():
    path.with_suffix('.neuroml-backup.json').write_bytes(path.read_bytes())
config=json.loads(path.read_text()) if path.exists() else {}
config['data-root']='/srv/neuroml/docker'
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(config,indent=2)+'\n')
path=Path('/etc/containerd/config.toml')
content=path.read_text() if path.exists() else 'version = 3\n'
if path.exists() and not path.with_suffix('.neuroml-backup.toml').exists():
    path.with_suffix('.neuroml-backup.toml').write_bytes(path.read_bytes())
root='root = "/srv/neuroml/containerd"'
if re.search(r'^root\s*=',content,re.M):
    content=re.sub(r'^root\s*=.*$',root,content,flags=re.M)
else:
    content=root+'\n'+content
path.write_text(content)
fstab=Path('/etc/fstab')
content=fstab.read_text()
entry='/mnt/d/NeuroML/storage.ext4 /srv/neuroml ext4 loop,defaults 0 0'
if entry not in content:
    fstab.write_text(content.rstrip()+'\n'+entry+'\n')
for service in ('docker','containerd'):
    directory=Path(f'/etc/systemd/system/{service}.service.d')
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'neuroml-storage.conf').write_text('[Unit]\nRequiresMountsFor=/srv/neuroml\nConditionPathIsMountPoint=/srv/neuroml\n')
PY
systemctl daemon-reload
systemctl start containerd.service docker.service
findmnt -T "$storage"
docker info --format 'Docker storage: {{.DockerRootDir}}'
