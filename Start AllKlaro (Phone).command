#!/bin/zsh
# Serves the translator on your LAN over HTTPS so a phone can use its mic
# (browsers require a secure context for microphone access off-localhost).
# On the phone: open https://<this-mac-ip>:8710 and accept the cert warning.
cd "$(dirname "$0")"

IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1)
CERT_DIR=certs
if [ ! -f "$CERT_DIR/cert.pem" ]; then
  echo "Generating self-signed certificate for $IP ..."
  mkdir -p "$CERT_DIR"
  openssl req -x509 -newkey rsa:2048 -sha256 -days 3650 -nodes \
    -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
    -subj "/CN=ai-transcriber" \
    -addext "subjectAltName=IP:$IP,IP:127.0.0.1,DNS:localhost"
fi

if ! curl -s -o /dev/null http://127.0.0.1:11434/api/tags; then
  echo "Starting Ollama..."
  open -a Ollama 2>/dev/null || (ollama serve &> /dev/null &)
  sleep 3
fi

echo ""
echo "  On your phone, open:  https://$IP:8710"
echo ""
echo "  First time on a new phone? Install the certificate once so iOS"
echo "  fully trusts the site (removes the warning and fixes the"
echo "  home-screen icon): open https://$IP:8710/cert and allow the"
echo "  download, then Settings > General > VPN & Device Management >"
echo "  install the profile, and finally enable it under Settings >"
echo "  General > About > Certificate Trust Settings."
echo ""
# --reload: pick up server.py changes without a manual restart.
exec uv run uvicorn server:app --host 0.0.0.0 --port 8710 --reload \
  --ssl-keyfile "$CERT_DIR/key.pem" --ssl-certfile "$CERT_DIR/cert.pem"
