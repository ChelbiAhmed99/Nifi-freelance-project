"""
Microservice de pretraitement d'image + OCR pour la partie 2 du projet
CARTE Assurances (extraction des donnees de la carte grise).

Appele par NiFi via un processeur InvokeHTTP (POST multipart/form-data,
champ "image").

Dependances :
    pip install flask opencv-python-headless pytesseract pillow

Tesseract doit etre installe sur la machine avec les packs de langue
fra + ara :
    Ubuntu/Debian : sudo apt install tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara
    Windows       : installer le binaire Tesseract (build UB Mannheim) et
                    definir pytesseract.pytesseract.tesseract_cmd ci-dessous.

Lancement :
    python ocr_service.py
    -> ecoute sur http://0.0.0.0:5001

Endpoint principal :
    POST /process
        form-data: image=<fichier>
        reponse JSON:
        {
          "raw_text": "...texte brut OCR...",
          "fields": [
             {"nom_fr": "Numero d'immatriculation", "nom_ar": "...", "valeur_fr": "123 TUN 456", "valeur_ar": ""},
             ...
          ]
        }
"""

import io
import re
import logging

import cv2
import numpy as np
import pytesseract
from PIL import Image
from flask import Flask, request, jsonify

# Si Tesseract n'est pas dans le PATH (typiquement sous Windows), decommenter et adapter :
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ocr_service")

TESSERACT_LANG = "fra+ara"  # carte grise tunisienne = bilingue FR/AR


# ---------------------------------------------------------------------
# Detection et recadrage automatique de la carte (algorithme "document
# scanner" classique) - essentiel quand la photo inclut le plan de travail
# / fond autour de la carte : sans ca, l'OCR perd beaucoup de resolution
# utile sur le texte et se perturbe avec le bruit du fond.
# ---------------------------------------------------------------------
def detect_and_crop_card(img_bgr: np.ndarray) -> np.ndarray:
    """Tente de detecter le rectangle de la carte dans la photo et de le
    recadrer/redresser (perspective transform). Si aucun contour a 4 cotes
    net n'est trouve, renvoie l'image originale telle quelle (fallback)."""
    orig = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return orig

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    card_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)
        # La carte doit couvrir une portion significative de la photo,
        # sinon on a probablement detecte un objet secondaire.
        if len(approx) == 4 and area > 0.15 * (h * w):
            card_contour = approx
            break

    if card_contour is None:
        return orig  # pas de rectangle net trouve -> on garde la photo entiere

    pts = card_contour.reshape(4, 2).astype("float32")

    # Ordonner les 4 points : haut-gauche, haut-droite, bas-droite, bas-gauche
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = ordered
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 100 or max_height < 100:
        return orig  # dimensions aberrantes -> fallback securite

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(orig, matrix, (max_width, max_height))
    return warped


# ---------------------------------------------------------------------
# Pretraitement d'image (OpenCV) - ameliore nettement la precision OCR
# sur des photos prises au telephone (eclairage, angle, bruit).
# ---------------------------------------------------------------------
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image illisible ou format non supporte")

    img = detect_and_crop_card(img)  # recadre sur la carte si detectee

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Agrandir l'image aide beaucoup Tesseract sur du texte petit/dense
    # (carte grise = beaucoup de texte fin) - sans ca, la binarisation
    # ecrase souvent les petits caracteres.
    h, w = gray.shape
    if max(h, w) < 2000:
        scale = 2000 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)

    # Nettete legere (sans lisser/flouter comme le faisait le debruitage)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

    # Seuillage Otsu (global, plus stable que l'adaptatif sur un fond propre)
    _, binary = cv2.threshold(sharpened, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def preprocess_image_raw_grayscale(image_bytes: bytes) -> np.ndarray:
    """Fallback minimal: carte recadree + niveaux de gris agrandis, sans
    binarisation. Utile en comparaison si preprocess_image degrade le
    resultat sur une image donnee (photo deja nette, bien eclairee)."""
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image illisible ou format non supporte")
    img = detect_and_crop_card(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max(h, w) < 2000:
        scale = 2000 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)
    return gray


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Corrige une legere inclinaison de l'image (scan/photo de travers)."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=100, maxLineGap=10)
    if lines is None:
        return gray

    angles = []
    # lines peut avoir la forme (N,1,4) ou (N,4) selon la version d'OpenCV -
    # reshape(-1, 4) uniformise les deux cas.
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = np.degrees(np.arctan2(int(y2) - int(y1), int(x2) - int(x1)))
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return gray

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return gray  # inclinaison negligeable

    (h, w) = gray.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h),
                           flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------
def run_ocr(binary_image: np.ndarray) -> str:
    pil_img = Image.fromarray(binary_image)
    text = pytesseract.image_to_string(pil_img, lang=TESSERACT_LANG)
    return text


# ---------------------------------------------------------------------
# Extraction des champs "carte grise" a partir du texte brut OCR.
#
# Libelles calibres sur un vrai certificat d'immatriculation tunisien
# (RTE 5022 / vehicule PEUGEOT PARTNER). Les regex tolerent les fautes
# d'OCR courantes (ex: "Constucteur", "lype", "S6rie") en acceptant des
# variations proches du libelle attendu plutot que le mot exact.
# A reajuster si Tesseract donne d'autres variantes sur d'autres cartes.
# ---------------------------------------------------------------------
FIELD_PATTERNS = [
    # (nom_fr, nom_ar, regex)
    ("Constructeur", "الصانع",
     r"Cons\w{0,10}ur[\s\S]{0,20}?([A-Z]{3,20})"),
    ("Type commercial", "الطراز التجاري",
     r"(?:commercial|lype commercial)[\s\S]{0,20}?([A-Z]{3,20})"),
    ("Date de mise en circulation (DPMC)", "تاريخ أول تسجيل",
     r"(\d{4}[/-]\d{2}[/-]\d{2})"),
    ("N° d'immatriculation", "رقم التسجيل",
     r"(?:d.?immatriculation|Immatriculation)\D{0,15}(\d{3,6})"),
]


def extract_serial_number(raw_text: str):
    """Le n° de serie du type (VIN-like, 15-20 caracteres alphanumeriques)
    ne peut pas etre repere par un simple mot-cle (le libelle 'N° Serie du
    type' est souvent mal reconnu par l'OCR). On cherche donc tous les
    blocs alphanumeriques de la bonne longueur et on garde celui qui
    contient le plus de chiffres (un vrai n° de serie en a beaucoup,
    contrairement a des mots comme 'IMMATRICULATION')."""
    candidates = re.findall(r"[A-Z0-9]{13,20}", raw_text)
    best = None
    best_digit_count = 0
    for c in candidates:
        digit_count = sum(ch.isdigit() for ch in c)
        if digit_count >= 4 and digit_count > best_digit_count:
            best = c
            best_digit_count = digit_count
    return best



def extract_fields(raw_text: str):
    fields = []
    for nom_fr, nom_ar, pattern in FIELD_PATTERNS:
        match = re.search(pattern, raw_text, flags=re.IGNORECASE)
        if match:
            fields.append({
                "nom_fr": nom_fr,
                "nom_ar": nom_ar,
                "valeur_fr": match.group(1).strip(),
                "valeur_ar": "",  # a completer si le champ arabe est capture separement
            })

    serial = extract_serial_number(raw_text)
    if serial:
        fields.append({
            "nom_fr": "N° Série du type (châssis)",
            "nom_ar": "رقم الهيكل",
            "valeur_fr": serial,
            "valeur_ar": "",
        })

    return fields


@app.route("/process", methods=["POST"])
def process():
    # Deux modes acceptes :
    # 1. multipart/form-data avec un champ "image" (pratique pour tester avec curl)
    # 2. corps brut de la requete = octets de l'image directement
    #    (c'est ce qu'envoie NiFi InvokeHTTP avec "Send Message Body" = true,
    #    qui poste simplement le contenu du flowfile tel quel)
    if "image" in request.files:
        image_bytes = request.files["image"].read()
    else:
        image_bytes = request.get_data()

    if not image_bytes:
        return jsonify({"error": "aucune image recue (ni champ 'image', ni corps de requete)"}), 400

    try:
        # Essai 1: version pretraitee (agrandie, nette, binarisee)
        binary = preprocess_image(image_bytes)
        raw_text = run_ocr(binary)

        # Essai 2: version brute (juste niveaux de gris + agrandissement) -
        # utile si l'image de depart est deja nette : la binarisation peut
        # parfois degrader plutot qu'aider.
        gray_only = preprocess_image_raw_grayscale(image_bytes)
        raw_text_fallback = run_ocr(gray_only)

        # On garde celle qui produit le plus de texte alphabetique/arabe
        # exploitable (heuristique simple: longueur de texte utile).
        def usable_length(text: str) -> int:
            return len(re.sub(r"[^\w]", "", text, flags=re.UNICODE))

        if usable_length(raw_text_fallback) > usable_length(raw_text):
            raw_text = raw_text_fallback

        fields = extract_fields(raw_text)
    except Exception as exc:  # noqa: BLE001
        log.exception("Erreur de traitement OCR")
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "raw_text": raw_text,
        "fields": fields,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/debug_crop", methods=["POST"])
def debug_crop():
    """Renvoie l'image PNG telle que recadree/redressee par
    detect_and_crop_card - pratique pour verifier visuellement que la
    detection de la carte fonctionne bien sur une photo donnee."""
    if "image" not in request.files:
        return jsonify({"error": "champ 'image' manquant dans la requete"}), 400

    file = request.files["image"]
    image_bytes = file.read()
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Image illisible"}), 400

    cropped = detect_and_crop_card(img)
    success, encoded = cv2.imencode(".png", cropped)
    if not success:
        return jsonify({"error": "Echec encodage image"}), 500

    from flask import Response
    return Response(encoded.tobytes(), mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
