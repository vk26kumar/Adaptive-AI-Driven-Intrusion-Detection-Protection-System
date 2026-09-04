"""
federated/gen_certs.py
Generate a self-signed CA plus a server certificate for TLS between the
federated server and the client nodes.  Run once:

    python -m federated.gen_certs

Produces federated/certs/{ca.crt, server.pem, server.key}.
Clients trust ca.crt; the server presents server.pem / server.key.
"""
import datetime
import ipaddress
import os

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from federated.config import CERT_DIR


def _key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def generate(cert_dir: str = CERT_DIR, days: int = 365) -> None:
    os.makedirs(cert_dir, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc)

    ca_key = _key()
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "A-IDAPS-FL Root CA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name).issuer_name(ca_name)
        .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now).not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    srv_key = _key()
    srv_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    srv_cert = (
        x509.CertificateBuilder()
        .subject_name(srv_name).issuer_name(ca_name)
        .public_key(srv_key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now).not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        ]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    with open(os.path.join(cert_dir, "ca.crt"), "wb") as f:
        f.write(ca_cert.public_bytes(serialization.Encoding.PEM))
    with open(os.path.join(cert_dir, "server.pem"), "wb") as f:
        f.write(srv_cert.public_bytes(serialization.Encoding.PEM))
    with open(os.path.join(cert_dir, "server.key"), "wb") as f:
        f.write(srv_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
    print(f"TLS certificates written to {cert_dir}")


def load_server_certificates(cert_dir: str = CERT_DIR):
    """Return (ca, cert, key) bytes in the order flwr.server.start_server expects."""
    with open(os.path.join(cert_dir, "ca.crt"), "rb") as f:
        ca = f.read()
    with open(os.path.join(cert_dir, "server.pem"), "rb") as f:
        crt = f.read()
    with open(os.path.join(cert_dir, "server.key"), "rb") as f:
        key = f.read()
    return ca, crt, key


def load_root_certificate(cert_dir: str = CERT_DIR) -> bytes:
    with open(os.path.join(cert_dir, "ca.crt"), "rb") as f:
        return f.read()


if __name__ == "__main__":
    generate()
