#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import requests
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

_ = load_dotenv(override=True)

# ========= Config =========
BASE_URL = os.getenv("RUNPOD_BASE_URL", "https://rest.runpod.io/v1")
API_KEY = os.getenv("RUNPOD_API_KEY")

if not API_KEY:
    raise SystemExit("Erro: defina a variável de ambiente RUNPOD_API_KEY.")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ========= Utils =========
def _safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return {"raw_text": resp.text}

def _extract_list(data: Any) -> List[Dict[str, Any]]:
    """
    A API do Runpod pode retornar:
    - { "data": [...] }  ou
    - { "pods": [...] }  ou
    - [ ... ]
    Esta função normaliza para uma lista.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "pods", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []

def _extract_obj(data: Any) -> Dict[str, Any]:
    """
    Para casos de criação/consulta que retornam:
    - { "data": { ... } }  ou  { ... }
    """
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], dict):
            return data["data"]
        return data
    return {}

def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


# ========= API Calls =========
def list_pods() -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/pods"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = _safe_json(resp)
    return _extract_list(data)

def get_pod(pod_id: str) -> Dict[str, Any]:
    url = f"{BASE_URL}/pods/{pod_id}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return _extract_obj(_safe_json(resp))

def create_pod(payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{BASE_URL}/pods"
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=120)
    resp.raise_for_status()
    return _extract_obj(_safe_json(resp))

def start_pod(pod_id: str) -> Dict[str, Any]:
    url = f"{BASE_URL}/pods/{pod_id}/start"
    resp = requests.post(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return _extract_obj(_safe_json(resp))

def stop_pod(pod_id: str) -> Dict[str, Any]:
    url = f"{BASE_URL}/pods/{pod_id}/stop"
    resp = requests.post(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return _extract_obj(_safe_json(resp))

def delete_pod(pod_id: str) -> Dict[str, Any]:
    url = f"{BASE_URL}/pods/{pod_id}"
    resp = requests.delete(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return _extract_obj(_safe_json(resp))


# ========= Payload Exemplo (edite conforme seu caso) =========
DEFAULT_CREATE_PAYLOAD = {
    "allowedCudaVersions": ["12.8"],
    "cloudType": "SECURE",
    "computeType": "GPU",
    "containerDiskInGb": 50,
    "cpuFlavorIds": ["cpu3c"],
    "cpuFlavorPriority": "availability",
    "dataCenterIds": [
        "EU-RO-1", "CA-MTL-1", "EU-SE-1", "US-IL-1", "EUR-IS-1", "EU-CZ-1",
        "US-TX-3", "EUR-IS-2", "US-KS-2", "US-GA-2", "US-WA-1", "US-TX-1",
        "CA-MTL-3", "EU-NL-1", "US-TX-4", "US-CA-2", "US-NC-1", "OC-AU-1",
        "US-DE-1", "EUR-IS-3", "CA-MTL-2", "AP-JP-1", "EUR-NO-1", "EU-FR-1",
        "US-KS-3", "US-GA-1"
    ],
    "dataCenterPriority": "availability",
    "dockerEntrypoint": [],
    "dockerStartCmd": [],
    "env": {"APP_ENV": "prod"},
    "globalNetworking": False,
    "gpuCount": 1,
    "gpuTypeIds": ["NVIDIA RTX A4500", "NVIDIA RTX A5000"],
    "gpuTypePriority": "availability",
    # Verifique a tag da imagem que você precisa:
    "imageName": "s0qbusjvhk",# "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    "interruptible": False,
    "locked": False,
    "minDiskBandwidthMBps": 123,
    "minDownloadMbps": 123,
    "minRAMPerGPU": 8,
    "minUploadMbps": 123,
    "minVCPUPerGPU": 2,
    "name": "ubchat-backend",
    # Se você tem um volume de rede, mantenha; caso contrário, remova.
    # "networkVolumeId": "i63gi55ihk",
    "ports": ["8888/http", "8000/http", "22/tcp"],
    "supportPublicIp": True,
    "vcpuCount": 2,
    "volumeInGb": 20,
    # "volumeMountPath": "/workspace"
}


# ========= CLI =========
def main():
    parser = argparse.ArgumentParser(
        description="Cliente simples para Runpod REST API (pods)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="Listar pods")

    p_info = sub.add_parser("info", help="Detalhes de um pod")
    p_info.add_argument("pod_id")

    p_create = sub.add_parser("create", help="Criar pod")
    p_create.add_argument("--payload", help="Arquivo JSON com payload; se omitido, usa DEFAULT_CREATE_PAYLOAD")

    p_stop = sub.add_parser("stop", help="Parar um pod")
    p_stop.add_argument("pod_id")

    p_start = sub.add_parser("start", help="Iniciar ou Retomar um pod")
    p_start.add_argument("pod_id")

    p_delete = sub.add_parser("delete", help="Deletar um pod")
    p_delete.add_argument("pod_id")

    args = parser.parse_args()

    try:
        if args.cmd == "list":
            pods = list_pods()
            if not pods:
                print("Nenhum pod encontrado.")
                return
            # Imprime id e nome para facilitar
            minimal = [{"id": p.get("id"), "name": p.get("name"), "status": p.get("status")} for p in pods]
            _print_json(minimal)

        elif args.cmd == "info":
            pod = get_pod(args.pod_id)
            _print_json(pod)

        elif args.cmd == "create":
            payload = DEFAULT_CREATE_PAYLOAD
            if args.payload:
                with open(args.payload, "r", encoding="utf-8") as f:
                    payload = json.load(f)

            pod = create_pod(payload)
            _print_json(pod)
            pod_id = pod.get("id") or pod.get("podId") or pod.get("pod_id")
            if pod_id:
                print(f"\nPod criado: {pod_id}")
            else:
                print("\nAviso: não foi possível identificar 'id' no retorno. Veja o JSON acima.")

        elif args.cmd == "start":
            out = start_pod(args.pod_id)
            _print_json(out)

        elif args.cmd == "stop":
            out = stop_pod(args.pod_id)
            _print_json(out)

        elif args.cmd == "delete":
            out = delete_pod(args.pod_id)
            _print_json(out)

    except requests.HTTPError as http_err:
        resp = http_err.response
        print(f"\nHTTP {resp.status_code} em {resp.request.method} {resp.request.url}")
        _print_json(_safe_json(resp))
        raise SystemExit(1)
    except requests.RequestException as req_err:
        print(f"\nErro de requisição: {req_err}")
        raise SystemExit(1)


if __name__ == "__main__":

    main()




    """
    # 1) Listar pods
    python runpod_client.py list

    # 2) Criar pod usando payload padrão do script
    python runpod_client.py create

    # 3) Criar pod com payload customizado em arquivo
    python runpod_client.py create --payload ./meu_payload.json

    # 4) Ver detalhes de um pod
    python runpod_client.py info <POD_ID>

    # 5) Parar um pod
    python runpod_client.py stop <POD_ID>

    # 6) Deletar um pod
    python runpod_client.py delete <POD_ID>

    """