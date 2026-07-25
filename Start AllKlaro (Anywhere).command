#!/bin/zsh
# Serves AllKlaro over your private Tailscale network (a WireGuard mesh
# VPN) with a real HTTPS certificate — reachable from your signed-in
# devices anywhere in the world, and never exposed to the public internet.
# Needs the Tailscale app installed and signed in on this Mac and on the
# phone; see the README's "Anywhere mode" section.
cd "$(dirname "$0")"

TS="tailscale"
command -v tailscale >/dev/null 2>&1 || TS="/Applications/Tailscale.app/Contents/MacOS/Tailscale"
if ! "$TS" status >/dev/null 2>&1; then
  echo "Tailscale is not running or not signed in."
  echo "Open the Tailscale app, sign in, then run this again."
  exit 1
fi

HOST=$("$TS" status --json 2>/dev/null | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')

if ! curl -s -o /dev/null http://127.0.0.1:11434/api/tags; then
  echo "Starting Ollama..."
  open -a Ollama 2>/dev/null || (ollama serve &> /dev/null &)
  sleep 3
fi

# Front the local server with Tailscale's HTTPS proxy: real certificate
# (no /cert install dance), WebSockets included. --bg keeps the proxy
# active until `tailscale serve reset`. If this errors about HTTPS not
# being enabled, follow the printed link once in the admin console.
"$TS" serve --bg http://127.0.0.1:8710

echo ""
echo "  On any of your Tailscale devices, open:  https://$HOST"
echo ""
echo "  iOS Shortcut URL (works at home and away):"
echo "      https://$HOST/api/translate"
echo "      ^ if your translate shortcut still holds a 192.168.x address,"
echo "        paste this over it — this mode serves 127.0.0.1 only, so the"
echo "        LAN URL now refuses to connect (Back Tap included)."
echo "  Browser smoke test:"
echo "      https://$HOST/api/translate?text=Hallo"
echo ""
echo "  The MacBook must stay awake to serve while you're away"
echo "  (Settings > Battery > prevent sleeping, or run: caffeinate -s)."
echo ""
# --reload: pick up server.py changes without a manual restart.
exec uv run uvicorn server:app --host 127.0.0.1 --port 8710 --reload
