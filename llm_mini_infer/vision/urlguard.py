"""The address block for image URLs (`images.allow_urls`): which hosts the server fetches from.

An image URL may name any host, so a client could make the server fetch from its own network: loopback, private,
link-local and cloud-metadata addresses. These are refused, by an explicit table (`BLOCKED`) and, as a second net, by
`ipaddress`'s `is_global` (not enough on its own: on Python 3.12 it calls multicast, Azure's 168.63.129.16 and NAT64
addresses global). IPv4-mapped (`::ffff:a.b.c.d`), IPv4-translated (`::ffff:0:a.b.c.d`), NAT64 (`64:ff9b::/96`) and
6to4 (`2002::/16`) addresses are checked as the IPv4 address they carry.

The check runs where the connection is made: `safe_resolver` wraps aiohttp's resolver and refuses a name's whole answer
when any address in it is blocked, and aiohttp then connects only to the addresses it returned, so a second DNS answer
cannot slip in between the check and the connection. A URL naming an address skips the resolver, so `check_url` runs on
the URL itself before every request, the first and each redirect's. Two settings widen or narrow it:
`images.allowed_networks` (CIDRs fetched from despite the table: an in-cluster image store) and
`images.allowed_domains` (exact hostnames, on the URL and every redirect: vLLM's and SGLang's `--allowed-media-domains`;
empty: any host). A refusal never names the address a name resolved to.

What it cannot catch (docs/operations.md, "Access"): a service reachable on a public address, a route through a
proxy that rewrites destinations, and what a public target's status and timing tell.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Iterable
from typing import Any

IP = ipaddress.IPv4Address | ipaddress.IPv6Address
Net = ipaddress.IPv4Network | ipaddress.IPv6Network

BLOCKED: tuple[Net, ...] = tuple(ipaddress.ip_network(n) for n in (
    # IPv4
    "0.0.0.0/8",  # "this network"
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",  # private
    "100.64.0.0/10",  # carrier-grade NAT (Alibaba Cloud's metadata, 100.100.100.200)
    "127.0.0.0/8",  # loopback
    "169.254.0.0/16",  # link-local: the clouds' metadata, 169.254.169.254
    "192.0.0.0/24",  # IETF protocol assignments
    "192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24",  # documentation (TEST-NET-1, -2, -3)
    "192.88.99.0/24",  # 6to4 relay anycast
    "198.18.0.0/15",  # benchmarking
    "224.0.0.0/4",  # multicast
    "240.0.0.0/4",  # reserved, the broadcast address included
    "168.63.129.16/32",  # Azure's wire server
    # IPv6
    "::/128", "::1/128",  # unspecified, loopback
    "::/96",  # IPv4-compatible (deprecated)
    "64:ff9b:1::/48",  # local-use NAT64
    "100::/64",  # discard
    "2001::/23",  # IETF protocol assignments (Teredo among them)
    "2001:db8::/32",  # documentation
    "fc00::/7",  # unique local (AWS's metadata, fd00:ec2::254)
    "fe80::/10",  # link-local
    "fec0::/10",  # site-local (deprecated)
    "ff00::/8",  # multicast
))
NAT64 = ipaddress.ip_network("64:ff9b::/96")
SIIT = ipaddress.ip_network("::ffff:0:0:0/96")  # IPv4-translated (::ffff:0:a.b.c.d)
REFUSED = "the URL's host is not one this server fetches from (images.allowed_networks, images.allowed_domains)"
_LABEL = re.compile(r"(?!-)[a-z0-9-]{1,63}(?<!-)")


class Blocked(Exception):
    """A URL this server does not fetch from: `host` and `reason` for the server's log (never the resolved address
    for the client)."""

    def __init__(self, host: str, reason: str):
        super().__init__(f"{host}: {reason}")
        self.host, self.reason = host, reason


class BlockedAddress(OSError):
    """A name whose DNS answer holds a blocked address, raised in the resolver: aiohttp wraps it in its
    ClientConnectorDNSError (`os_error`)."""

    def __init__(self, host: str):
        super().__init__(f"{host}: an address in its DNS answer is blocked")
        self.host = host


def unwrap(ip: IP) -> IP:
    """The IPv4 address an IPv4-mapped, IPv4-translated (SIIT), NAT64 or 6to4 address carries; any other address
    as it is."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return ip.ipv4_mapped
        if ip in NAT64 or ip in SIIT:
            return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if ip.sixtofour is not None:
            return ip.sixtofour
    return ip


def parse_networks(values: Iterable[str]) -> tuple[Net, ...]:
    """`images.allowed_networks`: CIDRs (host bits allowed, as `ip_network(strict=False)`); ValueError names the bad
    one."""
    out = []
    for v in values:
        try:
            out.append(ipaddress.ip_network(str(v).strip(), strict=False))
        except ValueError:
            raise ValueError(f"images.allowed_networks: {v!r} is not a network (CIDR, such as 10.0.0.0/8)") from None
    return tuple(out)


def normalize_host(host: str) -> str:
    """A host as the check compares it: an IP literal in its normal form, a name lowercased, without its trailing dot,
    in its IDNA (ASCII) form, as yarl writes a URL's host."""
    h = host.strip().rstrip(".").lower()
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    try:
        return str(ipaddress.ip_address(h))
    except ValueError:
        pass
    from yarl import URL

    return (URL.build(scheme="http", host=h).raw_host or "").lower()


def parse_domains(values: Iterable[str]) -> frozenset[str]:
    """`images.allowed_domains`: exact hostnames (no scheme, port, path or wildcard), normalized (`normalize_host`)."""
    out = set()
    for v in values:
        bad = ValueError(f"images.allowed_domains: {v!r} is not a hostname (an exact name, such as images.example.com)")
        if not isinstance(v, str) or not v.strip() or any(c in v for c in "/:@*?# \t"):
            if isinstance(v, str) and v.strip().startswith("[") and v.strip().endswith("]"):  # an IPv6 literal
                try:
                    out.add(normalize_host(v))
                    continue
                except ValueError:
                    pass
            raise bad
        try:
            h = normalize_host(v)
        except (ValueError, UnicodeError):
            raise bad from None
        try:
            ipaddress.ip_address(h)
        except ValueError:
            if not all(_LABEL.fullmatch(label) for label in h.split(".")):
                raise bad from None
        out.add(h)
    return frozenset(out)


class Policy:
    """Where the server fetches image URLs from: every public address, plus `allowed_networks`, and when
    `allowed_domains` is set, only those names."""

    def __init__(self, allowed_networks: Iterable[str] = (), allowed_domains: Iterable[str] = ()):
        self.networks = parse_networks(allowed_networks)
        self.domains = parse_domains(allowed_domains)

    def blocked(self, ip: IP) -> bool:
        """Whether the server refuses to connect to `ip`."""
        inner = unwrap(ip)
        if any(a in n for n in self.networks for a in {ip, inner} if a.version == n.version):
            return False
        if any(inner in n for n in BLOCKED if n.version == inner.version):
            return True
        return not inner.is_global


def check_url(url: Any, policy: Policy) -> None:
    """Raise `Blocked` unless the server fetches `url` (a yarl URL, as the request is made with it): http or https, a
    host, an address that is not blocked, and a name in `allowed_domains` when that is set. A name's addresses are
    checked by the resolver, when the connection is made (`safe_resolver`)."""
    host = url.host or ""
    if url.scheme not in ("http", "https"):
        raise Blocked(host, f"scheme {url.scheme!r}")
    if not host:
        raise Blocked(host, "no host")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and policy.blocked(ip):
        raise Blocked(host, "a blocked address")
    if policy.domains and normalize_host(url.raw_host or host) not in policy.domains:
        raise Blocked(host, "not in images.allowed_domains")


_resolver_class: type | None = None


def safe_resolver(policy: Policy, inner: Any = None) -> Any:
    """An aiohttp resolver that answers with `inner`'s addresses (default: aiohttp's own resolver) and refuses the
    whole answer, with `BlockedAddress`, when any of them is blocked (or not an address)."""
    global _resolver_class
    if _resolver_class is None:
        from aiohttp.abc import AbstractResolver
        from aiohttp.resolver import DefaultResolver

        class SafeResolver(AbstractResolver):
            def __init__(self, policy: Policy, inner: Any = None):
                self.policy, self.inner = policy, inner if inner is not None else DefaultResolver()

            async def resolve(self, host: str, port: int = 0,
                              family: socket.AddressFamily = socket.AF_INET) -> list[Any]:
                answer = await self.inner.resolve(host, port, family)
                for a in answer:
                    try:
                        ip = ipaddress.ip_address(a["host"])
                    except ValueError:
                        raise BlockedAddress(host) from None
                    if self.policy.blocked(ip):
                        raise BlockedAddress(host)
                return answer

            async def close(self) -> None:
                await self.inner.close()

        _resolver_class = SafeResolver
    return _resolver_class(policy, inner)
