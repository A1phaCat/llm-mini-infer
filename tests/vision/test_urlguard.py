"""The address block for image URLs (vision.urlguard), without a network: which addresses the policy refuses, how
`images.allowed_networks` exempts them, which URLs `check_url` refuses, and how the two settings are parsed."""

from __future__ import annotations

import ipaddress

import pytest
from yarl import URL

from llm_mini_infer.vision.urlguard import Blocked, Policy, check_url, parse_domains, parse_networks

BLOCKED = [
    "127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "169.254.169.254", "100.100.100.200", "0.0.0.0",
    "255.255.255.255", "224.0.0.1", "192.0.0.192", "198.18.0.1", "192.0.2.1", "198.51.100.1", "203.0.113.1",
    "192.88.99.1", "168.63.129.16", "::1", "::", "::127.0.0.1", "::ffff:127.0.0.1", "::ffff:7f00:1",
    "::ffff:0:127.0.0.1", "::ffff:0:a00:1", "64:ff9b::7f00:1", "64:ff9b:1::1", "2002:7f00:1::", "fc00::1",
    "fd00:ec2::254", "fe80::1", "fec0::1", "ff02::1", "2001::1", "2001:db8::1", "100::1",
]
ALLOWED = ["8.8.8.8", "1.1.1.1", "2606:4700::1111", "64:ff9b::808:808", "::ffff:0:808:808", "2002:808:808::"]


@pytest.mark.parametrize("addr", BLOCKED)
def test_the_policy_refuses_the_addresses_of_the_servers_own_network(addr) -> None:
    """Loopback, private, link-local, metadata, reserved and documentation ranges, and the IPv6 forms that carry them
    (IPv4-mapped, IPv4-translated, NAT64, 6to4), are blocked."""
    assert Policy().blocked(ipaddress.ip_address(addr))


@pytest.mark.parametrize("addr", ALLOWED)
def test_the_policy_allows_public_addresses(addr) -> None:
    """Public IPv4 and IPv6 addresses, and the IPv4-translated, NAT64 and 6to4 forms of public IPv4 ones, are not
    blocked."""
    assert not Policy().blocked(ipaddress.ip_address(addr))


def test_allowed_networks_exempt_only_their_own_addresses() -> None:
    """A network in allowed_networks lifts the block for its addresses and their IPv4-mapped forms; other private
    ranges stay blocked."""
    p = Policy(["127.0.0.0/8"])
    assert not p.blocked(ipaddress.ip_address("127.0.0.1"))
    assert not p.blocked(ipaddress.ip_address("::ffff:127.0.0.1"))
    assert p.blocked(ipaddress.ip_address("10.0.0.1"))
    assert p.blocked(ipaddress.ip_address("::1"))


@pytest.mark.parametrize("url", ["ftp://example.com/x.png", "file:///etc/passwd", "http:///x"])
def test_check_url_refuses_other_schemes_and_hostless_urls(url) -> None:
    """Only http and https URLs with a host pass."""
    with pytest.raises(Blocked):
        check_url(URL(url), Policy())


def test_check_url_checks_an_ip_literal() -> None:
    """A URL naming an address is checked on the URL itself, IPv6 literals and mapped forms too."""
    for url in ("http://127.0.0.1/x", "http://169.254.169.254/latest", "http://[::1]:8000/x",
                "http://[::ffff:7f00:1]/x"):
        with pytest.raises(Blocked):
            check_url(URL(url), Policy())
    check_url(URL("http://8.8.8.8/x"), Policy())
    check_url(URL("http://127.0.0.1/x"), Policy(["127.0.0.0/8"]))


def test_check_url_leaves_a_name_to_the_resolver() -> None:
    """Without allowed_domains a name passes the URL check; its addresses are checked when it is resolved."""
    check_url(URL("https://images.example.com/x.png"), Policy())


def test_allowed_domains_compare_normalised_names() -> None:
    """An entry is lowercased, without its trailing dot, in punycode; the URL's host is compared the same way, and a
    host not listed is refused."""
    p = Policy(allowed_domains=["Images.Example.COM."])
    check_url(URL("http://images.example.com/x"), p)
    check_url(URL("http://IMAGES.example.com./x"), p)
    with pytest.raises(Blocked):
        check_url(URL("http://other.example.com/x"), p)
    with pytest.raises(Blocked):
        check_url(URL("http://images.example.com.evil.test/x"), p)
    u = Policy(allowed_domains=["bücher.example"])
    check_url(URL("http://xn--bcher-kva.example/x"), u)
    check_url(URL("http://bücher.example/x"), u)


@pytest.mark.parametrize("entry", ["not-a-cidr", "http://x", "a/b", "10.0.0.0/33", ""])
def test_parse_networks_names_the_bad_entry(entry) -> None:
    """A value that is not a CIDR is a ValueError that names it and the setting."""
    with pytest.raises(ValueError, match=r"images\.allowed_networks"):
        parse_networks(["10.0.0.0/8", entry])


def test_parse_networks_accepts_cidrs_with_host_bits() -> None:
    """IPv4 and IPv6 CIDRs, a bare address and host bits are accepted."""
    nets = parse_networks(["10.0.0.0/8", "10.1.2.3/8", "fd00::/8", "192.168.1.1"])
    assert [str(n) for n in nets] == ["10.0.0.0/8", "10.0.0.0/8", "fd00::/8", "192.168.1.1/32"]


@pytest.mark.parametrize("entry", ["http://x", "a/b", "*.example.com", "", "host:80", "a b"])
def test_parse_domains_names_the_bad_entry(entry) -> None:
    """A value that is not an exact hostname is a ValueError that names it and the setting."""
    with pytest.raises(ValueError, match=r"images\.allowed_domains"):
        parse_domains(["ok.example.com", entry])


def test_parse_domains_accepts_names_and_bracketed_ipv6_literals() -> None:
    """Names are normalised; an IPv6 literal in brackets is accepted and compared in its normal form."""
    assert parse_domains(["A.Example.", "[::1]", "[FD00:0:0::1]"]) == {"a.example", "::1", "fd00::1"}
    p = Policy(["::1/128"], ["[::1]"])
    check_url(URL("http://[::1]/x"), p)
