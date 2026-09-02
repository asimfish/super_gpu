FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server python3 procps util-linux gawk \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd \
    && ssh-keygen -A \
    && useradd -m -s /bin/bash sguser

COPY fake-nvidia-smi /usr/local/bin/nvidia-smi
COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /usr/local/bin/nvidia-smi /entrypoint.sh

EXPOSE 22
ENTRYPOINT ["/entrypoint.sh"]
