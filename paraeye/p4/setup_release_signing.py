"""
ParaEye 发布签名一键配置（本地执行，不入库）。

做两件事：
1. 生成 PKCS12 签名 keystore（RSA 2048 / 25 年），连同口令写入本地 gitignore 目录
   paraeye/release/，供你自行备份；
2. 把 keystore（base64）与三条口令用 libsodium sealed box 加密后写入 GitHub
   Actions Secrets，供 tag 触发的 release 构建签名使用。

用法：
  python setup_release_signing.py <repo> <PAT>
"""

import base64
import json
import secrets
import ssl
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import pkcs12
from cryptography.x509.oid import NameOID
from nacl import encoding, public

BASE = Path(__file__).resolve().parents[1] / "release"
KEYSTORE_PATH = BASE / "paraeye-release.p12"
INFO_PATH = BASE / "keystore-info.txt"
ALIAS = "paraeye"
API = "https://api.github.com"


def api(path: str, token: str, payload: dict | None = None, method: str = "GET") -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        API + path, data=data, method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "paraeye-release-setup",
        },
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
            body = r.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SystemExit(f"GitHub API {method} {path} -> HTTP {e.code}\n{detail}") from None


def make_keystore() -> tuple[bytes, str, str, str]:
    if KEYSTORE_PATH.exists():
        raise SystemExit(f"{KEYSTORE_PATH} 已存在，删掉再跑（避免轮换掉已在用的签名）")
    store_pw = secrets.token_urlsafe(18)
    key_pw = secrets.token_urlsafe(18)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "ParaEye"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ParaEye"),
        x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
    ])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject).issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=365 * 25))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    p12 = pkcs12.serialize_key_and_certificates(
        name=b"paraeye", key=key, cert=cert, cas=None,
        encryption_algorithm=serialization.BestAvailableEncryption(store_pw.encode()),
    )
    KEYSTORE_PATH.write_bytes(p12)
    INFO_PATH.write_text(
        "ParaEye 发布签名（务必备份！丢失后无法给已安装的 ParaEye 发布更新）\n"
        f"keystore : {KEYSTORE_PATH}\n"
        f"alias    : {ALIAS}\n"
        f"store pwd: {store_pw}\n"
        f"key pwd  : {key_pw}\n"
        "GitHub Secrets: PARAKEYSTORE_B64 / PARAKEYSTORE_PASSWORD / PARAKEY_ALIAS / PARAKEY_PASSWORD\n",
        encoding="utf-8",
    )
    return p12, store_pw, ALIAS, key_pw


def put_secrets(repo: str, token: str, key_id: str, key_b64: str, values: dict[str, str]) -> None:
    pubkey = public.PublicKey(key_b64.encode(), encoding.Base64Encoder)
    for name, value in values.items():
        sealed = public.SealedBox(pubkey).encrypt(value.encode(), encoding.Base64Encoder).decode()
        api(f"/repos/{repo}/actions/secrets/{name}", token,
            {"encrypted_value": sealed, "key_id": key_id}, method="PUT")
        print(f"  secret 已写入: {name} ({len(value)} 字符)")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    repo, token = sys.argv[1], sys.argv[2]

    p12, store_pw, alias, key_pw = make_keystore()
    print(f"keystore 已生成: {KEYSTORE_PATH} ({len(p12)} 字节)")
    print(f"凭证摘要: {INFO_PATH}")

    pk = api(f"/repos/{repo}/actions/secrets/public-key", token)
    put_secrets(repo, token, pk["key_id"], pk["key"], {
        "PARAKEYSTORE_B64": base64.b64encode(p12).decode(),
        "PARAKEYSTORE_PASSWORD": store_pw,
        "PARAKEY_ALIAS": alias,
        "PARAKEY_PASSWORD": key_pw,
    })
    print("完成：GitHub Actions Secrets 就绪，可打 tag v* 出签名包。")


if __name__ == "__main__":
    main()
