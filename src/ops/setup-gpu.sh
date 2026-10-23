#!/usr/bin/env bash
# Run with sudo on each Ubuntu worker; do not install Linux display drivers in WSL.
set -euo pipefail
if ((EUID != 0)); then
    echo "Run: sudo bash $0" >&2
    exit 1
fi
. /etc/os-release
# Use signed official sources for this operation without changing the user's mirrors.
repo_dir=$(mktemp -d)
chmod 755 "$repo_dir"
trap 'rm -rf "$repo_dir"' EXIT
cat > "$repo_dir/sources.list" <<EOF
deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://archive.ubuntu.com/ubuntu $VERSION_CODENAME main universe
deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://archive.ubuntu.com/ubuntu $VERSION_CODENAME-updates main universe
deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://security.ubuntu.com/ubuntu $VERSION_CODENAME-security main universe
EOF
apt_options=(-o "Dir::Etc::sourcelist=$repo_dir/sources.list" -o 'Dir::Etc::sourceparts=-')
apt-get "${apt_options[@]}" update
apt-get "${apt_options[@]}" install -y --no-install-recommends ca-certificates curl gnupg
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
    gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
cat /etc/apt/sources.list.d/nvidia-container-toolkit.list >> "$repo_dir/sources.list"
apt-get "${apt_options[@]}" update
apt-get "${apt_options[@]}" install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
printf 'NVIDIA Docker support configured.\n'
