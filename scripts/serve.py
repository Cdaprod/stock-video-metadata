#!/usr/bin/env python3
import argparse
import time
import docker
from pathlib import Path
from datetime import datetime

client = docker.from_env()

# where your persistent data lives
MODEL_PATH    = Path("B:/Models").resolve()
WEAVIATE_DATA = MODEL_PATH / "weaviate-data"
MINIO_DATA    = MODEL_PATH / "minio-data"

SERVICES = {
    "llama-cpp": dict(
        image="ghcr.io/ggerganov/llama.cpp:server",
        ports={"8000/tcp": 8000},
        env={"MODEL": "/models/llama-model.gguf"},
        volumes={ str(MODEL_PATH): {"bind": "/models", "mode": "ro"} },
        cmd=None,
        start_fn=lambda: start_llama_cpp()
    ),
    "weaviate": dict(
        image="semitechnologies/weaviate:latest",
        ports={"8080/tcp": 8080},
        env={
            "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED":"true",
            "PERSISTENCE_DATA_PATH":"/var/lib/weaviate"
        },
        volumes={ str(WEAVIATE_DATA): {"bind":"/var/lib/weaviate","mode":"rw"} },
        cmd=None,
        start_fn=lambda: start_weaviate()
    ),
    "minio": dict(
        image="minio/minio:latest",
        ports={"9000/tcp":9000, "9001/tcp":9001},
        env={
            "MINIO_ROOT_USER":"minioadmin",
            "MINIO_ROOT_PASSWORD":"minioadmin"
        },
        cmd="server /data --console-address ':9001'",
        volumes={ str(MINIO_DATA): {"bind":"/data","mode":"rw"} },
        start_fn=lambda: start_minio()
    ),
}

def remove_if_exists(name:str):
    try:
        c = client.containers.get(name)
        c.remove(force=True)
    except docker.errors.NotFound:
        pass

def start_llama_cpp():
    name="llama-cpp"
    remove_if_exists(name)
    client.containers.run(
        SERVICES[name]["image"],
        name=name,
        command=SERVICES[name]["cmd"],
        volumes=SERVICES[name]["volumes"],
        environment=SERVICES[name]["env"],
        ports=SERVICES[name]["ports"],
        restart_policy={"Name":"unless-stopped"},
        detach=True
    )

def start_weaviate():
    name="weaviate"
    remove_if_exists(name)
    client.containers.run(
        SERVICES[name]["image"],
        name=name,
        command=SERVICES[name]["cmd"],
        volumes=SERVICES[name]["volumes"],
        environment=SERVICES[name]["env"],
        ports=SERVICES[name]["ports"],
        restart_policy={"Name":"unless-stopped"},
        detach=True
    )

def start_minio():
    name="minio"
    remove_if_exists(name)
    client.containers.run(
        SERVICES[name]["image"],
        name=name,
        command=SERVICES[name]["cmd"],
        volumes=SERVICES[name]["volumes"],
        environment=SERVICES[name]["env"],
        ports=SERVICES[name]["ports"],
        restart_policy={"Name":"unless-stopped"},
        detach=True
    )

def start_service(name:str):
    if name=="all":
        for svc in SERVICES:
            print(f"▶ Starting {svc}…")
            SERVICES[svc]["start_fn"]()
        return
    if name not in SERVICES:
        raise ValueError(f"Unknown service '{name}'")
    print(f"▶ Starting {name}…")
    SERVICES[name]["start_fn"]()

def stop_service(name:str):
    if name=="all":
        for svc in SERVICES:
            print(f"⏹ Stopping {svc}…")
            remove_if_exists(svc)
        return
    try:
        c = client.containers.get(name)
        print(f"⏹ Stopping {name}…")
        c.remove(force=True)
    except docker.errors.NotFound:
        print(f"❗ {name} is not running")

def restart_service(name:str):
    stop_service(name)
    # give Docker a moment
    time.sleep(0.5)
    start_service(name)

def status_report():
    out = {}
    for name in SERVICES:
        try:
            c = client.containers.get(name)
            out[name] = {
                "id":      c.id[:12],
                "status":  c.status,
                "image":   c.image.tags[0] if c.image.tags else c.image.short_id,
                "created": datetime.fromtimestamp(c.attrs["Created"]).isoformat(),
                "ports":   { str(p["PublicPort"]):p["PrivatePort"] for p in c.ports.values() for p in ([p] if isinstance(p,dict) else p) }
            }
        except docker.errors.NotFound:
            out[name] = None
    return out

def get_logs(name:str, tail:int=100) -> str:
    """Return last `tail` lines of logs for service `name`."""
    try:
        c = client.containers.get(name)
        return c.logs(tail=tail).decode("utf-8", errors="ignore")
    except docker.errors.NotFound:
        return f"❌ Service '{name}' not found"

def stream_logs(name:str):
    """Generator that yields new log lines (follow=True)."""
    c = client.containers.get(name)
    for raw in c.logs(stream=True, follow=True, tail=0):
        yield raw.decode("utf-8", errors="ignore")

def print_status():
    report = status_report()
    for svc, info in report.items():
        if info:
            print(f"• {svc}: {info['status']} (id={info['id']}) ports={info['ports']}")
        else:
            print(f"• {svc}: not running")

def main():
    p = argparse.ArgumentParser(prog="serve.py",
        description="Start/stop/status/logs for your services")
    sp = p.add_subparsers(dest="cmd", required=True)

    sp_status = sp.add_parser("status", help="Show status of all services")
    sp_start  = sp.add_parser("start", help="Start one or ALL services")
    sp_start.add_argument("service", choices=list(SERVICES.keys())+["all"])
    sp_stop   = sp.add_parser("stop",  help="Stop one or ALL services")
    sp_stop.add_argument("service", choices=list(SERVICES.keys())+["all"])
    sp_restart= sp.add_parser("restart",help="Restart one or ALL services")
    sp_restart.add_argument("service", choices=list(SERVICES.keys())+["all"])
    sp_logs   = sp.add_parser("logs",   help="Fetch logs")
    sp_logs.add_argument("service", choices=list(SERVICES.keys()))
    sp_logs.add_argument("--tail",    "-t", type=int, default=100, help="Last N lines")
    sp_logs.add_argument("--follow",  "-f", action="store_true", help="Follow")

    args = p.parse_args()

    if args.cmd=="status":
        print_status()
    elif args.cmd=="start":
        start_service(args.service)
    elif args.cmd=="stop":
        stop_service(args.service)
    elif args.cmd=="restart":
        restart_service(args.service)
    elif args.cmd=="logs":
        if args.follow:
            try:
                for line in stream_logs(args.service):
                    print(line, end="")
            except KeyboardInterrupt:
                pass
        else:
            print(get_logs(args.service, tail=args.tail))

if __name__=="__main__":
    main()