-- =====================================================================
-- Partie 2 - CARTE Assurances - Gestion des pieces jointes (carte grise)
-- Version FINALE - a executer sur la base "email_processing"
--
-- Contexte confirme aupres d'Ines (execution reelle de sa partie 1) :
--   1) NiFi demarre tous les processeurs (ConsumePOP3 -> ... -> PutSQL)
--   2) Un email est envoye a stageassurance5@gmail.com
--   3) Verification via SQL Workbench :
--        USE email_processing;
--        SELECT * FROM extracte_emails;
--
-- => La VRAIE table de production s'appelle "extracte_emails"
--    (avec le "e" final), PAS "extract_emails".
--    Cette table existe DEJA, est alimentee par le flow NiFi d'Ines,
--    et contient de vrais emails. On ne la cree JAMAIS depuis ce
--    script, on ne la modifie JAMAIS, on ne fait AUCUN INSERT dedans :
--    on la reference uniquement en cle etrangere.
-- =====================================================================

USE email_processing;

-- ---------------------------------------------------------------------
-- Etape 0 (a faire une seule fois, manuellement, avant d'executer le
-- reste) : verifier la structure exacte de extracte_emails, en
-- particulier le nom/type de sa cle primaire, car c'est elle qui est
-- referencee par ATTACHMENT.id_email juste en dessous.
--
--   DESCRIBE extracte_emails;
--
-- D'apres fix_extracte_emails.sql (script qui a cree cette table en
-- s'alignant sur le PutSQL reel d'Ines), la cle primaire est bien
-- "id INT AUTO_INCREMENT". Si DESCRIBE montre autre chose chez Ines,
-- adapter le type de ATTACHMENT.id_email et la ligne REFERENCES
-- ci-dessous en consequence avant d'executer.
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ATTACHMENT (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    id_email        INT NOT NULL,                 -- lien vers extracte_emails.id
    nom_piece       VARCHAR(255) NOT NULL,         -- nom du fichier de la piece jointe
    type            VARCHAR(100),                  -- ex: image/jpeg, application/pdf
    taille           INT,                            -- taille en octets
    date_extraction DATETIME DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_attachment_email
        FOREIGN KEY (id_email) REFERENCES extracte_emails(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS ATTACHMENT_DETAILS (
    id                INT AUTO_INCREMENT PRIMARY KEY,
    id_attachment     INT NOT NULL,
    nom_detail_fr     VARCHAR(150),   -- ex: "Numero d'immatriculation"
    nom_detail_ar     VARCHAR(150),   -- ex: equivalent arabe
    valeur_detail_fr  VARCHAR(255),   -- valeur detectee (texte francais/chiffres)
    valeur_detail_ar  VARCHAR(255),   -- valeur detectee (texte arabe)
    confiance_ocr     DECIMAL(5,2),   -- score de confiance Tesseract (0-100)
    date_creation     DATETIME DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_details_attachment
        FOREIGN KEY (id_attachment) REFERENCES ATTACHMENT(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Index utiles pour les jointures / recherches frequentes
CREATE INDEX idx_attachment_email ON ATTACHMENT(id_email);
CREATE INDEX idx_details_attachment ON ATTACHMENT_DETAILS(id_attachment);

-- ---------------------------------------------------------------------
-- Requete de verification a utiliser dans SQL Workbench, sur le meme
-- modele que celle d'Ines, pour confirmer le lien email <-> piece jointe :
--
--   SELECT e.id, e.subject_email, e.sender_email, e.message_id,
--          a.nom_piece, a.type, a.taille
--   FROM extracte_emails e
--   JOIN ATTACHMENT a ON a.id_email = e.id
--   ORDER BY e.id DESC;
-- ---------------------------------------------------------------------
