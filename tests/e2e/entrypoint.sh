#!/bin/sh
# Install the test's public key for sguser, then run sshd in the foreground.
set -eu

if [ -z "${SSH_PUBKEY:-}" ]; then
    echo "SSH_PUBKEY environment variable is required" >&2
    exit 1
fi

mkdir -p /home/sguser/.ssh
printf '%s\n' "$SSH_PUBKEY" > /home/sguser/.ssh/authorized_keys
chown -R sguser:sguser /home/sguser/.ssh
chmod 700 /home/sguser/.ssh
chmod 600 /home/sguser/.ssh/authorized_keys

exec /usr/sbin/sshd -D -e
