"""Signed client for your own Kalshi account: balance, orders, fills, positions, place and cancel.

Credentials are read from the environment and are never stored in this repository:

    export KALSHI_KEY_ID=<the API key id shown when the key was created>
    export KALSHI_KEY_FILE=<path to the private-key file Kalshi let you download>

Every request carries the key id, a millisecond timestamp, and the base64 signature of
`timestamp + METHOD + path`, where the path includes /trade-api/v2 and leaves out the query string.
An Ed25519 key signs the message directly; an RSA key uses PSS with SHA-256.

Kalshi runs several matching engines ("exchange shards") and holds cash separately on each: 0 is the default, 2 carries
crypto and commodities. The app moves cash between them by itself; an API order needs the cash on the market's shard first.
"""
import base64
import os
import re
import textwrap
import time

import requests

from .api import KALSHI

PREFIX = "/trade-api/v2"


def load_key(path):
    """Read a PEM private key, rebuilding the line breaks if it was saved on one line."""
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    text = open(os.path.expanduser(path)).read()
    m = re.search(r"-----BEGIN ([A-Z ]+)-----(.*?)-----END \1-----", text, re.S)
    if not m:
        raise SystemExit(f"{path} does not hold a PEM private key")
    body = "".join(m.group(2).split())
    pem = f"-----BEGIN {m.group(1)}-----\n" + "\n".join(textwrap.wrap(body, 64)) + f"\n-----END {m.group(1)}-----\n"
    return load_pem_private_key(pem.encode(), password=None)


def message(ts_ms, method, path):
    """The string that gets signed."""
    return f"{ts_ms}{method.upper()}{PREFIX}{path.split('?')[0]}"


def sign(key, text):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    if isinstance(key, Ed25519PrivateKey):
        raw = key.sign(text.encode())
    else:
        raw = key.sign(text.encode(), padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                       hashes.SHA256())
    return base64.b64encode(raw).decode()


def centicents(dollars):
    return int(round(float(dollars) * 10_000))


class Account:
    def __init__(self):
        try:
            self.key_id, key_file = os.environ["KALSHI_KEY_ID"], os.environ["KALSHI_KEY_FILE"]
        except KeyError as missing:
            raise SystemExit(f"set {missing.args[0]} first (see the top of kfv/account.py)")
        self.key = load_key(key_file)
        self.http = requests.Session()

    def call(self, method, path, params=None, body=None):
        ts = str(int(time.time() * 1000))
        headers = {"KALSHI-ACCESS-KEY": self.key_id, "KALSHI-ACCESS-TIMESTAMP": ts,
                   "KALSHI-ACCESS-SIGNATURE": sign(self.key, message(ts, method, path)), "Accept": "application/json"}
        r = self.http.request(method, KALSHI + path, params=params, json=body, headers=headers, timeout=20)
        if r.status_code >= 300:
            raise RuntimeError(f"{method} {path} -> {r.status_code} {r.text[:400]}")
        return r.json() if r.text else {}

    def balance(self, shard=None):
        """Available cash in dollars: on one exchange shard, or across all of them when `shard` is None."""
        try:
            d = self.call("GET", "/portfolio/balance", None if shard is None else {"exchange_index": shard})
        except RuntimeError:
            if shard is None:
                raise
            return 0.0                     # no cash has ever been moved to that shard
        return d["balance"] / 100.0

    def move_cash(self, dollars, source_shard, destination_shard):
        """Move cash between exchange shards of the same account. The API counts in hundredths of a cent."""
        return self.call("POST", "/portfolio/intra_exchange_instance_transfer",
                         body=dict(source="event_contract", destination="event_contract", amount=centicents(dollars),
                                   source_exchange_shard=source_shard, destination_exchange_shard=destination_shard))

    def orders(self, status="resting", event=None):
        params = {"status": status, "limit": 200}
        if event:
            params["event_ticker"] = event
        return self.call("GET", "/portfolio/orders", params).get("orders", [])

    def fills(self, since_ts):
        return self.call("GET", "/portfolio/fills", {"min_ts": int(since_ts), "limit": 500}).get("fills", [])

    def positions(self):
        return self.call("GET", "/portfolio/positions", {"limit": 200})

    def place(self, payload):
        return self.call("POST", "/portfolio/events/orders", body=payload)

    def cancel(self, order_id, ticker):
        return self.call("DELETE", f"/portfolio/events/orders/{order_id}", {"market_ticker": ticker})
