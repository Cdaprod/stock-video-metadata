# scripts/serve_control.py
import docker
from pathlib import Path
from datetime import datetime

class ServeManager:
    MODEL_PATH    = Path("B:/Models").resolve()
    WEAVIATE_DATA = MODEL_PATH / "weaviate-data"
    MINIO_DATA    = MODEL_PATH / "minio-data"

    SERVICES = {
        "llama-cpp": dict(
            image="ghcr.io/ggerganov/llama.cpp:server",
            ports={"8000/tcp": 8000},
            env={"MODEL": "/models/llama-model.gguf"},
            volumes={ str(MODEL_PATH): {"bind": "/models", "mode": "ro"} },
            cmd=None
        ),
        "weaviate": dict(
            image="semitechnologies/weaviate:latest",
            ports={"8080/tcp": 8080},
            env={
                "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED":"true",
                "PERSISTENCE_DATA_PATH":"/var/lib/weaviate"
            },
            volumes={ str(WEAVIATE_DATA): {"bind":"/var/lib/weaviate","mode":"rw"} },
            cmd=None
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
        ),
    }

    def __init__(self):
        self.client = docker.from_env()

    def _remove_if_exists(self, name):
        try:
            c = self.client.containers.get(name)
            c.remove(force=True)
        except docker.errors.NotFound:
            pass

    def start(self, name):
        if name == "all":
            return {s: self.start(s) for s in self.SERVICES}
        if name not in self.SERVICES:
            return f"Unknown service: {name}"
        self._remove_if_exists(name)
        cfg = self.SERVICES[name]
        container = self.client.containers.run(
            cfg["image"],
            name=name,
            command=cfg.get("cmd"),
            volumes=cfg.get("volumes", {}),
            environment=cfg.get("env", {}),
            ports=cfg.get("ports", {}),
            restart_policy={"Name":"unless-stopped"},
            detach=True,
            auto_remove=False
        )
        return f"Started {name} ({container.short_id})"

    def stop(self, name):
        if name == "all":
            return {s: self.stop(s) for s in self.SERVICES}
        try:
            c = self.client.containers.get(name)
            c.remove(force=True)
            return f"Stopped {name}"
        except docker.errors.NotFound:
            return f"{name} not running"

    def restart(self, name):
        self.stop(name)
        return self.start(name)

    def status(self):
        out = {}
        for name in self.SERVICES:
            try:
                c = self.client.containers.get(name)
                out[name] = {
                    "id": c.short_id,
                    "status": c.status,
                    "image": c.image.tags[0] if c.image.tags else c.image.short_id,
                    "created": datetime.fromtimestamp(c.attrs["Created"]).isoformat(),
                    "ports": c.attrs.get("NetworkSettings", {}).get("Ports", {}),
                }
            except docker.errors.NotFound:
                out[name] = None
        return out

    def logs(self, name, tail=50):
        try:
            c = self.client.containers.get(name)
            return c.logs(tail=tail).decode("utf-8", errors="ignore")
        except docker.errors.NotFound:
            return f"{name} not running"

    def stream_logs(self, name, n_lines=10):
        from itertools import islice
        try:
            c = self.client.containers.get(name)
            for line in islice(c.logs(stream=True, follow=True), n_lines):
                print(line.decode("utf-8", errors="ignore"), end="")
        except docker.errors.NotFound:
            print(f"{name} not running")

# USAGE IN A NOTEBOOK CELL:
# from scripts.serve_control import ServeManager
# svc = ServeManager()
# svc.start("all")            # start all
# svc.status()                # check statuses
# print(svc.logs("minio", 30))
# svc.stop("llama-cpp")       # stop one
# svc.restart("weaviate")
# svc.stream_logs("minio", 50) # print 50 new log lines as they appear