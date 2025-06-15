import docker
from pathlib import Path

client = docker.from_env()

# Paths
MODEL_PATH = Path("B:/Models").resolve()
WEAVIATE_DATA = MODEL_PATH / "weaviate-data"
MINIO_DATA = MODEL_PATH / "minio-data"

def start_llama_cpp():
    container_name = "llama-cpp"
    try:
        client.containers.get(container_name).remove(force=True)
    except docker.errors.NotFound:
        pass

    client.containers.run(
        image="ghcr.io/ggerganov/llama.cpp:server",
        name=container_name,
        volumes={
            str(MODEL_PATH): {'bind': '/models', 'mode': 'ro'},
        },
        environment={
            "MODEL": "/models/llama-model.gguf",
        },
        ports={'8000/tcp': 8000},
        restart_policy={"Name": "unless-stopped"},
        detach=True,
        auto_remove=False,
    )

def start_weaviate():
    container_name = "weaviate"
    try:
        client.containers.get(container_name).remove(force=True)
    except docker.errors.NotFound:
        pass

    client.containers.run(
        image="semitechnologies/weaviate:latest",
        name=container_name,
        ports={'8080/tcp': 8080},
        environment={
            "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
            "PERSISTENCE_DATA_PATH": "/var/lib/weaviate",
        },
        volumes={
            str(WEAVIATE_DATA): {'bind': '/var/lib/weaviate', 'mode': 'rw'},
        },
        restart_policy={"Name": "unless-stopped"},
        detach=True,
        auto_remove=False,
    )

def start_minio():
    container_name = "minio"
    try:
        client.containers.get(container_name).remove(force=True)
    except docker.errors.NotFound:
        pass

    client.containers.run(
        image="minio/minio:latest",
        name=container_name,
        command="server /data --console-address ':9001'",
        ports={'9000/tcp': 9000, '9001/tcp': 9001},
        environment={
            "MINIO_ROOT_USER": "minioadmin",
            "MINIO_ROOT_PASSWORD": "minioadmin",
        },
        volumes={
            str(MINIO_DATA): {'bind': '/data', 'mode': 'rw'},
        },
        restart_policy={"Name": "unless-stopped"},
        detach=True,
        auto_remove=False,
    )

def start_all():
    start_llama_cpp()
    start_weaviate()
    start_minio()
    print("✅ All containers started.")

if __name__ == "__main__":
    start_all()