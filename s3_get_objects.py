# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

# Carrega variáveis de ambiente (.env)
_ = load_dotenv(override=True)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  # ajuste se necessário

session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)
s3 = session.client("s3")

def list_buckets():
    """Lista os buckets da conta."""
    resp = s3.list_buckets()
    buckets = resp.get("Buckets", [])
    if buckets:
        print("S3 Buckets:")
        for b in buckets:
            print(f"  Name: {b['Name']}, Creation Date: {b['CreationDate']}")
    else:
        print("No S3 buckets found in this account.")

def list_bucket_objects(bucket_name: str, prefix: str = ""):
    """Lista objetos de um bucket (opcionalmente filtrando por prefixo)."""
    paginator = s3.get_paginator("list_objects_v2")
    found_any = False
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # ignora "pastas" (chaves terminadas em '/')
            if key.endswith("/"):
                continue
            found_any = True
            size = obj["Size"]
            print(f"{key}  ({size} bytes)")
    if not found_any:
        print("Nenhum objeto encontrado para esse bucket/prefixo.")

def download_bucket(
    bucket_name: str,
    local_dir: str,
    prefix: str = "",
    keep_structure: bool = True,
):
    """
    Baixa todos os objetos de um bucket (ou prefixo) para um diretório local.

    - keep_structure=True mantém a mesma hierarquia de pastas do S3.
    """
    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    total = 0

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue  # ignora chaves de "pasta"
            if keep_structure:
                local_path = os.path.join(local_dir, key)  # mantém hierarquia
            else:
                local_path = os.path.join(local_dir, os.path.basename(key))

            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                s3.download_file(bucket_name, key, local_path)
                total += 1
                print(f"Baixado: s3://{bucket_name}/{key} -> {local_path}")
            except ClientError as e:
                print(f"Falha ao baixar {key}: {e}")

    if total == 0:
        print("Nada para baixar (confira bucket/prefixo e permissões).")
    else:
        print(f"Concluído. {total} arquivo(s) baixado(s).")

if __name__ == "__main__":
    # Exemplos de uso:
    list_buckets()
    # list_bucket_objects("rag-teste-bnu")                       # lista tudo
    # list_bucket_objects("rag-teste-bnu", prefix="docs/")      # lista por prefixo

    # Baixar todo o bucket mantendo a estrutura no diretório ./downloads
    # download_bucket("rag-teste-bnu", local_dir="downloads", prefix="", keep_structure=True)

    # Para baixar apenas um subdiretório/prefixo do bucket:
    # download_bucket("rag-teste-bnu", local_dir="downloads_docs", prefix="docs/", keep_structure=True)
