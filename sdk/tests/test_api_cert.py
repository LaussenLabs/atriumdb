# AtriumDB is a timeseries database software designed to best handle the unique features and
# challenges that arise from clinical waveform data.
#     Copyright (C) 2023  The Hospital for Sick Children
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
import datetime
import ipaddress
import logging
import pickle
import socket
import ssl
import threading
import os
import time
from pathlib import Path
from typing import Iterator, Optional, Union
from unittest import mock

import pytest
import requests
import uvicorn
from click.testing import CliRunner, Result
from dotenv import dotenv_values
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID

from atriumdb.atrium_sdk import AtriumSDK
from atriumdb.cli.atriumdb_cli import cli
from tests.mock_api.app import app
from tests.mock_api.sdk_dependency import get_sdk_instance
from tests.test_mit_bih import write_mit_bih_to_dataset, assert_mit_bih_to_dataset
from tests.testing_framework import free_port

MAX_RECORDS = 2
SEED = 42


def _write_private_ca_and_server_cert(cert_dir: Path) -> tuple[Path, Path, Path]:
    """
    Make a throwaway private CA and a server cert for 127.0.0.1/localhost signed by it, like the real API server's
    setup. The CA and leaf carry key usage and key identifier extensions because Python 3.13+ verifies strictly.
    Returns (ca_cert_path, server_cert_path, server_key_path).
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key, server_key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AtriumDB Test Private CA")])

    ca_cert = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name).public_key(ca_key.public_key())
               .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=5))
               .not_valid_after(now + datetime.timedelta(days=1))
               .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
               .add_extension(x509.KeyUsage(digital_signature=False, content_commitment=False, key_encipherment=False,
                                            data_encipherment=False, key_agreement=False, key_cert_sign=True,
                                            crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
               .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
               .sign(ca_key, hashes.SHA256()))

    server_cert = (x509.CertificateBuilder()
                   .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")]))
                   .issuer_name(ca_name).public_key(server_key.public_key())
                   .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=5))
                   .not_valid_after(now + datetime.timedelta(days=1))
                   .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                                                               x509.DNSName("localhost")]), critical=False)
                   .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                   .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False,
                                                data_encipherment=False, key_agreement=True, key_cert_sign=False,
                                                crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
                   .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
                   .add_extension(x509.SubjectKeyIdentifier.from_public_key(server_key.public_key()), critical=False)
                   .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
                   .sign(ca_key, hashes.SHA256()))

    ca_cert_path, server_cert_path, server_key_path = cert_dir / "ca.pem", cert_dir / "server.pem", cert_dir / "server.key"
    ca_cert_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    server_cert_path.write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))
    server_key_path.write_bytes(server_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                         serialization.NoEncryption()))
    return ca_cert_path, server_cert_path, server_key_path


def _start_mock_api(port: int, server_cert_path: Optional[Path] = None, server_key_path: Optional[Path] = None):
    """Run the mock API in a daemon thread (https when a cert is given) and wait until it accepts connections."""
    server_kwargs = {} if server_cert_path is None else {"ssl_certfile": str(server_cert_path), "ssl_keyfile": str(server_key_path)}
    threading.Thread(target=lambda: uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", **server_kwargs),
                     daemon=True).start()

    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Mock API server did not start listening on 127.0.0.1:{port} within 30 seconds.")


@pytest.fixture(scope="module")
def mock_api_servers(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, str, Path]]:
    """
    One local sqlite dataset with MIT-BIH data, served by the mock API over both https (private CA cert) and plain
    http. Returns (https_url, http_url, ca_cert_path).
    """
    sdk = AtriumSDK.create_dataset(dataset_location=tmp_path_factory.mktemp("dataset"), database_type="sqlite")
    write_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS, seed=SEED, use_period=True)
    app.dependency_overrides[get_sdk_instance] = lambda: sdk

    ca_cert_path, server_cert_path, server_key_path = _write_private_ca_and_server_cert(tmp_path_factory.mktemp("certs"))
    https_port, http_port = free_port(), free_port()
    _start_mock_api(https_port, server_cert_path, server_key_path)
    _start_mock_api(http_port)

    yield f"https://127.0.0.1:{https_port}", f"http://127.0.0.1:{http_port}", ca_cert_path
    app.dependency_overrides.pop(get_sdk_instance, None)


@pytest.fixture(autouse=True)
def no_cert_env_var(monkeypatch: pytest.MonkeyPatch):
    # Keep a developer's own ATRIUMDB_API_CERT_FILE from leaking into these tests. The sdk reloads ./.env with
    # override=True whenever cert_file is None, which would put it straight back, so that load is turned off too.
    monkeypatch.delenv("ATRIUMDB_API_CERT_FILE", raising=False)
    monkeypatch.setattr("atriumdb.atrium_sdk.load_dotenv", lambda *args, **kwargs: False)


def _api_sdk(api_url: str, cert_file: Optional[str] = None) -> AtriumSDK:
    api_sdk = AtriumSDK(metadata_connection_type="api", api_url=api_url, validate_token=False, cert_file=cert_file)
    # validate_token=False never sets an expiry, so give one far in the future for the token check in _request
    api_sdk.token_expiry = time.time() + 1_000_000
    return api_sdk


def test_https_with_private_ca_cert(mock_api_servers: tuple[str, str, Path]):
    https_url, _, ca_cert_path = mock_api_servers

    # exercises both the requests (metadata) and websocket (block) paths over TLS
    api_sdk = _api_sdk(https_url, cert_file=str(ca_cert_path))
    assert_mit_bih_to_dataset(api_sdk, max_records=MAX_RECORDS, seed=SEED, use_period=True)
    api_sdk.close()


def test_https_sdk_with_cert_pickles(mock_api_servers: tuple[str, str, Path]):
    https_url, _, ca_cert_path = mock_api_servers

    # an sdk with a cert must still pickle (e.g. for spawned DataLoader workers) and the copy must still connect over TLS
    api_sdk_copy = pickle.loads(pickle.dumps(_api_sdk(https_url, cert_file=str(ca_cert_path))))
    assert_mit_bih_to_dataset(api_sdk_copy, max_records=MAX_RECORDS, seed=SEED, use_period=True)
    api_sdk_copy.close()


def test_https_cert_file_from_env_var(mock_api_servers: tuple[str, str, Path], monkeypatch: pytest.MonkeyPatch):
    https_url, _, ca_cert_path = mock_api_servers
    monkeypatch.setenv("ATRIUMDB_API_CERT_FILE", str(ca_cert_path))

    api_sdk = _api_sdk(https_url)
    assert api_sdk.cert_file == str(ca_cert_path)
    assert len(api_sdk.get_all_devices()) == MAX_RECORDS
    api_sdk.close()


def test_https_without_cert_fails(mock_api_servers: tuple[str, str, Path]):
    https_url, _, _ = mock_api_servers

    # the default public CA bundle can't verify a private CA, which is the failure this feature fixes
    api_sdk = _api_sdk(https_url)
    with pytest.raises(requests.exceptions.SSLError):
        api_sdk.get_all_devices()


def test_http_with_cert_warns_and_ignores(mock_api_servers: tuple[str, str, Path], caplog: pytest.LogCaptureFixture):
    _, http_url, ca_cert_path = mock_api_servers

    with caplog.at_level(logging.WARNING, logger="atriumdb.atrium_sdk"):
        api_sdk = _api_sdk(http_url, cert_file=str(ca_cert_path))
    assert any("certificate is ignored" in record.getMessage() for record in caplog.records)
    assert api_sdk.cert_file is None

    # still works over plain http/ws, including the websocket block path
    assert_mit_bih_to_dataset(api_sdk, max_records=MAX_RECORDS, seed=SEED, use_period=True)
    api_sdk.close()


def test_missing_cert_file_raises(tmp_path: Path):
    missing_cert_path = tmp_path / "does_not_exist.pem"
    with pytest.raises(ValueError, match="does_not_exist.pem"):
        AtriumSDK(metadata_connection_type="api", api_url="https://127.0.0.1:1", validate_token=False,
                  cert_file=str(missing_cert_path))


def test_invalid_pem_raises(tmp_path: Path):
    not_a_cert_path = tmp_path / "not_a_cert.pem"
    not_a_cert_path.write_text("this is not a certificate")
    with pytest.raises(ValueError, match="not_a_cert.pem' could not be loaded"):
        AtriumSDK(metadata_connection_type="api", api_url="https://127.0.0.1:1", validate_token=False,
                  cert_file=str(not_a_cert_path))


def test_empty_cert_env_var_means_no_cert(monkeypatch: pytest.MonkeyPatch):
    # `ATRIUMDB_API_CERT_FILE=` left empty in .env is treated as not set, not as a file named ''
    monkeypatch.setenv("ATRIUMDB_API_CERT_FILE", "")
    api_sdk = AtriumSDK(metadata_connection_type="api", api_url="https://127.0.0.1:1", validate_token=False)
    assert api_sdk.cert_file is None and api_sdk.requests_verify is True


def test_websocket_cert_error_raised_in_caller(mock_api_servers: tuple[str, str, Path]):
    https_url, _, _ = mock_api_servers

    # the TLS failure happens in the connect thread, it must reach the caller as a clear error, not a later AttributeError
    api_sdk = _api_sdk(https_url)
    with pytest.raises(ConnectionError, match="with no cert_file") as error_info:
        api_sdk._websocket_connect()
    assert isinstance(error_info.value.__cause__, ssl.SSLCertVerificationError)
    assert api_sdk.websock_conn is None


PRIVATE_ENDPOINT, PUBLIC_ENDPOINT = "https://private.example", "https://public.example"


@pytest.fixture
def cli_in_tmp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the CLI in an empty folder (so ./.env starts empty) with os.environ restored afterwards. Returns a cert path."""
    monkeypatch.chdir(tmp_path)
    # login's load_dotenv writes into os.environ, so give it a throwaway copy that pytest swaps back at teardown
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for env_var in ("ATRIUMDB_API_CERT_FILE", "ATRIUMDB_ENDPOINT_URL"):
        os.environ.pop(env_var, None)
    cert_path = tmp_path / "ca.pem"
    cert_path.write_text("not read, requests is mocked")
    return cert_path


AUTH_CONFIG = {"auth0_tenant": "tenant.example", "auth0_client_id": "client", "auth0_audience": "audience",
               "algorithms": ["RS256"]}


def _run_login(cli_args: list[str], auth_status: int = 200) -> tuple[Result, dict[str, Optional[str]], Union[str, bool, None]]:
    """
    Run `atriumdb <cli_args>` with the AtriumDB /auth/cli/code request mocked to return auth_status, and the Auth0
    device code request mocked to fail, so login stops right after the AtriumDB server step.
    Returns (click result, the saved .env values, the verify value the /auth/cli/code request was sent with, or None if
    login stopped before sending it).
    """
    auth_config_response = mock.Mock(status_code=auth_status, json=lambda: AUTH_CONFIG)
    with mock.patch("atriumdb.cli.atriumdb_cli.requests.get", return_value=auth_config_response) as mock_get, \
            mock.patch("atriumdb.cli.atriumdb_cli.requests.post", return_value=mock.Mock(status_code=500)):
        result = CliRunner().invoke(cli, cli_args)
    return result, dotenv_values("./.env"), mock_get.call_args.kwargs["verify"] if mock_get.called else None


def test_login_saves_cert_given_on_command_line(cli_in_tmp_dir: Path):
    cert_path = cli_in_tmp_dir

    _, saved_env, verify = _run_login(["--cert-file", str(cert_path), "login", "--endpoint-url", PRIVATE_ENDPOINT])
    assert saved_env["ATRIUMDB_API_CERT_FILE"] == str(cert_path.resolve())
    assert verify == str(cert_path)


def test_login_same_endpoint_keeps_cert(cli_in_tmp_dir: Path):
    cert_path = cli_in_tmp_dir
    _run_login(["--cert-file", str(cert_path), "login", "--endpoint-url", PRIVATE_ENDPOINT])

    # logging in again, or refreshing the token, without --cert-file must keep using the saved cert
    for cli_args in (["login", "--endpoint-url", PRIVATE_ENDPOINT], ["refresh-token"]):
        _, saved_env, verify = _run_login(cli_args)
        assert saved_env["ATRIUMDB_API_CERT_FILE"] == str(cert_path.resolve()), cli_args
        assert verify == str(cert_path.resolve()), cli_args


def test_login_different_endpoint_removes_cert(cli_in_tmp_dir: Path):
    cert_path = cli_in_tmp_dir
    _run_login(["--cert-file", str(cert_path), "login", "--endpoint-url", PRIVATE_ENDPOINT])

    # a different server without --cert-file drops the old server's cert so its public CA still verifies
    result, saved_env, verify = _run_login(["login", "--endpoint-url", PUBLIC_ENDPOINT])
    assert "ATRIUMDB_API_CERT_FILE" not in saved_env
    assert "ATRIUMDB_API_CERT_FILE" not in os.environ
    assert verify is True
    assert "Removed the saved cert file" in result.output


def test_login_missing_cert_path_saves_nothing(cli_in_tmp_dir: Path):
    # a typo in --cert-file is a usage error before anything is written, not a traceback and a broken .env
    result, saved_env, verify = _run_login(["--cert-file", "typo.pem", "login", "--endpoint-url", PRIVATE_ENDPOINT])
    assert result.exit_code == 2 and "typo.pem' does not exist" in result.output
    assert saved_env == {} and verify is None


def test_login_failed_request_does_not_save_cert(cli_in_tmp_dir: Path):
    cert_path = cli_in_tmp_dir
    _, saved_env, verify = _run_login(["--cert-file", str(cert_path), "login", "--endpoint-url", PRIVATE_ENDPOINT],
                                      auth_status=404)
    assert verify == str(cert_path)
    assert "ATRIUMDB_API_CERT_FILE" not in saved_env


def test_login_http_endpoint_warns_and_skips_cert(cli_in_tmp_dir: Path):
    cert_path = cli_in_tmp_dir
    result, saved_env, verify = _run_login(["--cert-file", str(cert_path), "login", "--endpoint-url", "http://private.example"])
    assert "does not use https://" in result.output
    assert "ATRIUMDB_API_CERT_FILE" not in saved_env
    assert verify is True
