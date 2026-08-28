""" HTTP utils """
import socket
import random

def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except (OSError, OverflowError):
            return False

def find_available_port(base_port: int, consecutive: int = 1):
    """Find the starting port of `consecutive` consecutive available ports.

    The scan starts at a random offset above ``base_port`` so that concurrently
    starting processes do not all probe the same port first.
    """
    port = base_port + random.randint(0, 100)
    max_port = 65536 - consecutive
    while port <= max_port:
        for i in range(consecutive):
            if not is_port_available(port + i):
                port += i + 1  # skip past the occupied port
                break
        else:
            return port
    raise RuntimeError(
        f"No {consecutive} consecutive available ports found above {base_port}"
    )