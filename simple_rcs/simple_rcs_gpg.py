import logging
import os
import subprocess
import sys
import tempfile


logger = logging.getLogger(__name__)
# Basic logging config if not already configured by application
if not logging.root.handlers:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def _get_gpg_command() -> str:
    """Determines the GPG command to use based on SRCS_GPGSIGN_PATH or system PATH."""
    return os.environ.get("SRCS_GPGSIGN_PATH", "gpg")

def get_gpg_uid(key_id: str) -> tuple[str, str]:  # noqa: C901
    """
    Retrieves the User ID (Name <email>) and trust level string associated with a GPG key ID.
    Returns a tuple (uid_string, trust_level_string) or (key_id, "unknown") if lookup fails.
    Trust level is 'u' (ultimate), 'f' (fully), 'm' (marginal), 'q' (undefined), 'n' (never), 'e' (expired).
    """
    gpg_cmd = _get_gpg_command()
    try:
        # Check if key_id is valid to avoid "No public key" error output being misinterpreted or raising exception
        res = subprocess.run(
            [gpg_cmd, "--list-keys", "--with-colons", key_id],
            capture_output=True, text=True, check=True,
        )

        uid_string = key_id
        trust_level = "unknown"

        for line in res.stdout.splitlines():
            parts = line.split(':')

            # pub:u:4096:1:KEYID:...
            # Index 1 is trust (validity)
            if line.startswith("pub:"):
                if len(parts) > 1 and parts[1]:
                    owner_trust_char = parts[1]
                    if owner_trust_char == 'u':
                        trust_level = "ultimate"
                    elif owner_trust_char == 'f':
                        trust_level = "fully"
                    elif owner_trust_char == 'm':
                        trust_level = "marginal"
                    elif owner_trust_char == 'q':
                        trust_level = "undefined"
                    elif owner_trust_char == 'n':
                        trust_level = "never"
                    elif owner_trust_char == 'e':
                        trust_level = "expired"
                    elif owner_trust_char == '-':
                        trust_level = "unknown"

            if line.startswith("uid:"):
                if len(parts) > 9:
                    uid_string = parts[9].strip()
                    # Also try to get trust level from UID line if pub line wasn't parsed
                    # or if UID specific trust is available.
                    if len(parts) > 1 and parts[1]:
                        uid_trust_char = parts[1]
                        if trust_level == "unknown" and uid_trust_char in 'ufmqne':
                             if uid_trust_char == 'u':
                                trust_level = "ultimate"
                             elif uid_trust_char == 'f':
                                trust_level = "fully"
                             elif uid_trust_char == 'm':
                                trust_level = "marginal"

        return uid_string, trust_level
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"GPG key lookup for '{key_id}' failed: {e}")

    return key_id, "unknown"

def get_default_signer_id() -> str:
    """
    Determines the GPG signer ID. Prioritizes SRCS_SIGNING_KEY env var,
    then attempts to find the default secret key from GPG.
    Returns the UID string (e.g., "Name <email>") or key_id if lookup fails.
    """
    signer_key_env = os.environ.get("SRCS_SIGNING_KEY")
    if signer_key_env:
        # If env var provides key_id, try to get its UID
        uid, _ = get_gpg_uid(signer_key_env)
        if uid != signer_key_env: # If lookup was successful
            return uid
        return signer_key_env # Fallback to key_id

    # Try to get default key ID from gpg
    gpg_cmd = _get_gpg_command()
    try:
        res = subprocess.run(
            [gpg_cmd, "--list-secret-keys", "--with-colons"],
            capture_output=True, text=True, check=True,
        )
        for line in res.stdout.splitlines():
            if line.startswith("uid:"):
                parts = line.split(":")
                if len(parts) > 9:
                    # Extract UID string, e.g., "Won-Kyu Park <wkpark@gmail.com>"
                    return parts[9].strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"GPG --list-secret-keys failed: {e}. Using 'unknown_gpg_user'.")
    return "unknown_gpg_user"

def gpg_sign_callback(message: str, signer_id: str | None = None) -> tuple[str, str]:
    """
    Signs the message using GPG. The GPG key can be specified via the signer_id argument
    or SRCS_SIGNING_KEY env var, then falls back to default system key.
    Invokes pinentry (tty-based) for passphrase if needed.

    Returns:
        (signer_id, signature_value)
    """
    gpg_cmd = _get_gpg_command()

    actual_signer_id = signer_id if signer_id else get_default_signer_id()
    if not actual_signer_id or actual_signer_id == "unknown_gpg_user":
        # Fallback to letting GPG decide default key if we couldn't find one explicitly
        pass

    sign_mode = os.environ.get("SRCS_SIGN_MODE", "detach-sign")

    cmd = [gpg_cmd, "--armor"]
    # Only append default-key if actual_signer_id looks valid
    if actual_signer_id != "unknown_gpg_user":
        # Check if actual_signer_id is a full UID string "Name <email>".
        # GPG --default-key accepts UIDs.
        cmd.append(f"--default-key={actual_signer_id}")

    if sign_mode == "clearsign":
        cmd.append("--clearsign")
    else: # Default to detach-sign
        cmd.append("--detach-sign")

    # Environment for pinentry-tty.
    env = os.environ.copy()
    if sys.stdin.isatty():
        env["GPG_TTY"] = os.ttyname(0)

    try:
        process = subprocess.run(
            cmd,
            input=message.encode('utf-8'),
            capture_output=True, # Signature is on stdout
            check=True,
            env=env,
        )
        signature = process.stdout.decode('utf-8').strip()

        if sign_mode == "clearsign":
            logger.warning("SRCS_SIGN_MODE=clearsign used, but SimpleRCS expects detached signature.")
            raise ValueError("SRCS_SIGN_MODE=clearsign is not supported for detached signatures.")

        return actual_signer_id, signature
    except subprocess.CalledProcessError as e:
        logger.error(f"GPG Signing Failed for {actual_signer_id}: {e.stderr.decode().strip()}")
        raise ValueError(f"GPG signing failed: {e.stderr.decode().strip()}") from e
    except FileNotFoundError as e:
        logger.error(f"GPG cmd not found at '{gpg_cmd}'.")
        raise ValueError(f"GPG cmd not found at '{gpg_cmd}'. Ensure GPG is installed or SRCS_GPGSIGN_PATH.") from e

def gpg_verify_callback(signer_id: str, message: str, signature: str) -> bool:
    """
    Verifies the signature against the message using GPG.

    Args:
        signer_id: The ID of the signer (as provided during signing).
        message: The original message that was signed (timestamp|hash).
        signature: The detached GPG ASCII Armor signature block.

    Returns:
        True if the signature is valid, False otherwise.
    """
    gpg_cmd = _get_gpg_command()

    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as sig_file:
        sig_file.write(signature)
        sig_path = sig_file.name

    try:
        # gpg --verify <signature_file> <signed_file_or_stdin>
        cmd = [gpg_cmd, "--verify", sig_path, "-"]

        subprocess.run(
            cmd,
            input=message.encode('utf-8'),
            capture_output=True, # Capture both stdout/stderr
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        logger.error(f"GPG command not found at '{gpg_cmd}'.")
        return False
    finally:
        if os.path.exists(sig_path):
            os.remove(sig_path)
