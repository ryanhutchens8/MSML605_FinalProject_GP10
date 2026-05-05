import logging
import os
import time
from datetime import datetime, timezone

import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException


DYNAMIC_API_URL = os.getenv("DYNAMIC_API_URL", "http://dynamic-service:8000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus-service:9090")
# tried 1500 but it triggered too often early in the simulation
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD_MW", "2000"))
COOLDOWN_HOURS = int(os.getenv("COOLDOWN_HOURS", "336"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SEC", "60"))
NAMESPACE = os.getenv("NAMESPACE", "load-forecast")
JOB_IMAGE = os.getenv("JOB_IMAGE", "load-forecast-api:v18")
RETRAINED_DIR = os.getenv("RETRAINED_MODELS_DIR", "/app/retrained-models")
JOB_TIMEOUT_SEC = int(os.getenv("JOB_TIMEOUT_SEC", "600"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [drift-monitor] %(levelname)s %(message)s",
)
log = logging.getLogger("drift-monitor")


last_retrain_time = None
retrain_count = 0


def query_prometheus(metric: str):
    url = f"{PROMETHEUS_URL}/api/v1/query"
    try:
        resp = requests.get(url, params={"query": metric}, timeout=5)
        resp.raise_for_status()
        results = resp.json().get("data", {}).get("result", [])
        if results:
            return float(results[0]["value"][1])
        return None
    except Exception:
        log.warning("Prometheus query failed")
        return None


def get_latest_api_datetime():
    try:
        resp = requests.get(f"{DYNAMIC_API_URL}/", timeout=5)
        resp.raise_for_status()
        return resp.json().get("latest_datetime")
    except Exception:
        log.warning("Could not read latest API timestamp")
        return None


def load_k8s_client():
    try:
        # works when running inside the cluster
        config.load_incluster_config()
        log.info("Loaded in-cluster k8s config")
    except config.ConfigException:
        # fallback for running locally
        config.load_kube_config()
        log.info("Loaded local kube config")
    return client.BatchV1Api()


def create_retrain_job(batch_v1, end_time: str) -> str:
    job_name = f"retrain-job-{int(time.time())}"

    job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name=job_name,
            namespace=NAMESPACE,
        ),
        spec=client.V1JobSpec(
            ttl_seconds_after_finished=300,
            backoff_limit=1,
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={"app": "retrain-job"},
                ),
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[
                        client.V1Container(
                            name="retrain",
                            image=JOB_IMAGE,
                            image_pull_policy="IfNotPresent",
                            command=[
                                "python", "-m", "app.retrain",
                                "--end-time", end_time,
                            ],
                            env=[
                                client.V1EnvVar(
                                    name="RETRAINED_MODELS_DIR",
                                    value=RETRAINED_DIR,
                                ),
                            ],
                            volume_mounts=[
                                client.V1VolumeMount(
                                    name="model-storage",
                                    mount_path=RETRAINED_DIR,
                                ),
                            ],
                        )
                    ],
                    volumes=[
                        client.V1Volume(
                            name="model-storage",
                            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                                claim_name="model-storage",
                            ),
                        )
                    ],
                ),
            ),
        ),
    )

    batch_v1.create_namespaced_job(namespace=NAMESPACE, body=job)
    log.info(f"Created retrain job: {job_name}")
    return job_name


def wait_for_job(batch_v1, job_name: str) -> bool:
    log.info(f"Waiting for job {job_name} (timeout={JOB_TIMEOUT_SEC}s)...")
    start = time.time()

    while time.time() - start < JOB_TIMEOUT_SEC:
        try:
            job = batch_v1.read_namespaced_job(name=job_name, namespace=NAMESPACE)
            if job.status.succeeded and job.status.succeeded > 0:
                log.info(f"Job {job_name} completed successfully")
                return True
            if job.status.failed and job.status.failed > 0:
                log.error(f"Job {job_name} failed")
                return False
        except ApiException:
            log.warning("Error reading job status")

        time.sleep(10)

    log.error(f"Job {job_name} timed out after {JOB_TIMEOUT_SEC}s")
    return False


def trigger_reload():
    try:
        resp = requests.post(f"{DYNAMIC_API_URL}/reload", timeout=10)
        resp.raise_for_status()
        log.info("Model reload triggered")
        return True
    except Exception:
        log.error("Failed to trigger /reload")
        return False


def mark_retrain_start():
    try:
        resp = requests.post(f"{DYNAMIC_API_URL}/retrain-start", timeout=5)
        resp.raise_for_status()
    except Exception:
        log.warning("Could not mark retrain start")


def cooldown_ok(now: datetime) -> bool:
    if last_retrain_time is None:
        return True
    elapsed_hours = (now - last_retrain_time).total_seconds() / 3600
    return elapsed_hours >= COOLDOWN_HOURS


def run():
    global last_retrain_time, retrain_count

    log.info("Drift monitor starting")
    log.info(f"Prometheus: {PROMETHEUS_URL}")
    log.info(f"Dynamic API: {DYNAMIC_API_URL}")
    log.info(f"Threshold: {DRIFT_THRESHOLD} MW | Cooldown: {COOLDOWN_HOURS}h | Poll: {POLL_INTERVAL}s")

    batch_v1 = load_k8s_client()

    while True:
        mae = query_prometheus('rolling_mae_mw{mode="dynamic"}')
        drift_active = query_prometheus('drift_active{mode="dynamic"}')

        log.info(
            f"rolling_mae={mae} | "
            f"drift_active={drift_active} | "
            f"retrains={retrain_count}"
        )

        now = datetime.now(timezone.utc)

        # only retrain if drift is detected and cooldown has passed
        if mae is not None and mae > DRIFT_THRESHOLD and cooldown_ok(now):
            log.info(f"Drift found | MAE={mae:.2f} MW | starting retrain")

            last_retrain_time = now
            retrain_count += 1
            end_time = get_latest_api_datetime() or now.strftime("%Y-%m-%d %H:%M:%S")
            try:
                job_name = create_retrain_job(batch_v1, end_time)
            except ApiException:
                log.error("Failed to create retrain job")
                time.sleep(POLL_INTERVAL)
                continue

            mark_retrain_start()
            success = wait_for_job(batch_v1, job_name)

            if success:
                trigger_reload()
            else:
                log.error("Retrain job failed, keeping current model")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
