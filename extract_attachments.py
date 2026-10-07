#!/usr/bin/env python3
"""
Extraction des pieces jointes d'un email brut (MIME) - partie 2 du
projet CARTE Assurances.

A utiliser dans NiFi via un processeur ExecuteStreamCommand juste apres
ConsumePOP3 (sur une branche PARALLELE a celle d'Ines qui traite
sujet/expediteur/cc/date/corps).

Fonctionnement :
- Lit l'email brut MIME sur stdin (c'est le contenu du flowfile NiFi).
- Parcourt les parties du message a la recherche de pieces jointes
  (images et PDF - filtre sur le content-type).
- Sauvegarde chaque piece jointe sur disque dans OUTPUT_DIR.
- Ecrit sur stdout un JSON avec les metadonnees de chaque piece jointe,
  que NiFi peut ensuite parser (SplitJson + EvaluateJsonPath) pour
  alimenter la table ATTACHMENT puis appeler le microservice OCR.

Sortie JSON (exemple) :
{
  "message_id": "<abc123@mail.gmail.com>",
  "attachments": [
    {
      "name": "carte_grise.jpg",
      "type": "image/jpeg",
      "size": 245678,
      "path": "/opt/nifi/attachments/1699999999_carte_grise.jpg"
    }
  ]
}

Si aucune piece jointe pertinente n'est trouvee, "attachments" est une
liste vide - a filtrer plus loin dans NiFi (RouteOnAttribute) pour ne
pas inserer de lignes inutiles.
"""

import sys
import os
import json
import time
from email import message_from_bytes
from email.policy import default as default_policy

# Repertoire de sauvegarde des pieces jointes - a adapter a l'environnement
OUTPUT_DIR = os.environ.get("ATTACHMENTS_DIR", "/opt/nifi/attachments")

# Types MIME consideres comme pertinents pour ce projet (carte grise)
ALLOWED_TYPES_PREFIX = ("image/", "application/pdf")

# Taille minimale (en octets) pour filtrer les petites images
# (logos/signatures embarques dans le corps du mail, souvent < 15 Ko)
MIN_SIZE_BYTES = 15_000


def sanitize_filename(name: str) -> str:
    """Evite les caracteres dangereux dans un nom de fichier venant d'un mail."""
    name = os.path.basename(name or "piece_jointe")
    return "".join(c for c in name if c.isalnum() or c in (" ", ".", "_", "-")).strip()


def extract_attachments(raw_bytes: bytes) -> dict:
    msg = message_from_bytes(raw_bytes, policy=default_policy)
    message_id = msg.get("Message-ID", "")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    attachments = []
    for part in msg.walk():
        content_disposition = part.get_content_disposition()
        content_type = part.get_content_type()

        is_attachment_disposition = content_disposition in ("attachment", "inline")
        is_relevant_type = content_type.startswith(ALLOWED_TYPES_PREFIX)

        if not (is_attachment_disposition and is_relevant_type):
            continue

        payload = part.get_payload(decode=True)
        if not payload:
            continue

        if len(payload) < MIN_SIZE_BYTES:
            continue  # probablement un logo/signature, pas une carte grise

        filename = sanitize_filename(part.get_filename())
        unique_name = f"{int(time.time() * 1000)}_{filename}"
        filepath = os.path.join(OUTPUT_DIR, unique_name)

        with open(filepath, "wb") as f:
            f.write(payload)

        attachments.append({
            "name": filename,
            "type": content_type,
            "size": len(payload),
            "path": filepath,
        })

    return {
        "message_id": message_id,
        "attachments": attachments,
    }


def main():
    raw_bytes = sys.stdin.buffer.read()
    result = extract_attachments(raw_bytes)
    sys.stdout.write(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
