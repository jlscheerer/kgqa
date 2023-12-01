-- CREATE TYPES --

CREATE TYPE entity_type AS enum ('item', 'property');

CREATE TYPE claims_type AS enum ('statement');

CREATE TYPE claims_rank AS enum (
  'normal', 'preferred', 'deprecated'
);

CREATE TYPE claims_snaktype AS enum ('value', 'novalue', 'somevalue');

CREATE TYPE claims_datatype AS enum (
  'external-id', 'wikibase-item', 'monolingualtext', 
  'time', 'string', 'commonsMedia', 
  'quantity', 'globe-coordinate', 
  'url', 'wikibase-form', 'math', 'tabular-data', 
  'geo-shape', 'wikibase-property', 
  'wikibase-lexeme', 'musical-notation', 
  'wikibase-sense'
);

CREATE TYPE claims_datavalue_type AS enum (
  'string', 'wikibase-entityid', 'time', 
  'globecoordinate', 'quantity', 'monolingualtext'
);

-- CREATE TABLES --
CREATE TABLE entities (
  id VARCHAR(16) PRIMARY KEY, 
  type entity_type, 
  value VARCHAR(32)
);

CREATE TABLE claims (
 entity_id              VARCHAR(16) REFERENCES entities(id),
 id                     VARCHAR(64) PRIMARY KEY,
 property               VARCHAR(16) REFERENCES entities(id),
 datavalue_type         claims_datavalue_type,
 datavalue_string       TEXT,
 datavalue_entity       VARCHAR(16),
 datavalue_date         TIMESTAMP,
 datavalue_quantity     DOUBLE PRECISION -- NUMERIC(256, 128)
);

CREATE TABLE qualifiers (
 claim_id               VARCHAR(64) REFERENCES claims(id),
 qualifier_property     VARCHAR(16) REFERENCES entities(id),
 datavalue_type         claims_datavalue_type,
 datavalue_string       TEXT,
 datavalue_entity       VARCHAR(16),
 datavalue_date         TIMESTAMP,
 datavalue_quantity     DOUBLE PRECISION -- NUMERIC(256, 128)
);

CREATE TABLE labels (
  id VARCHAR(64) REFERENCES entities(id),
  language VARCHAR(16),
  value TEXT
);

CREATE TABLE aliases (
  id VARCHAR(16) REFERENCES entities(id),
  language VARCHAR(16),
  value TEXT
);

CREATE TABLE descriptions (
  id VARCHAR(16) REFERENCES entities(id),
  language VARCHAR(16),
  value TEXT
);

-- IMPORT DATA --

\copy entities from 'csv/entities.txt' DELIMITER E'\t'

-- `claims` requires additional preprocessing through wd-migrate.
\copy claims FROM 'csv/claims.csv' DELIMITER E'\t' NULL '';

CREATE TABLE claims_inv AS (
  SELECT entity_id, id, property, datavalue_type, datavalue_string, datavalue_entity,
         datavalue_date, datavalue_quantity, False AS inverse
  FROM claims
);
DROP TABLE claims;

CREATE TABLE entity_popularity AS (
  SELECT entity_id, COUNT(*)
  FROM claims_inv
  GROUP BY entity_id
);

-- `qualifiers` requires additional preprocessing through wd-migrate.
\copy qualifiers FROM 'csv/qualifiers.csv' DELIMITER E'\t' NULL '';

\copy labels from 'csv/labels.txt' DELIMITER E'\t'

\copy aliases from 'csv/aliases.txt' DELIMITER E'\t'

\copy descriptions from 'csv/descriptions.txt' DELIMITER E'\t'

CREATE TABLE descriptions_en AS (
  SELECT * FROM descriptions WHERE language = 'en'
);
DROP TABLE descriptions;

-- CREATE MATERIALIZED VIEWS --

CREATE MATERIALIZED VIEW labels_en AS (
    SELECT id, language, value FROM labels WHERE language = 'en'
);

CREATE MATERIALIZED VIEW properties AS (
    SELECT DISTINCT property AS id, datatype AS value FROM claims
);

CREATE MATERIALIZED VIEW qualifier_properties AS (
    SELECT DISTINCT q.qualifier_property AS id, c.property, q.datavalue_type AS value
    FROM qualifiers q, claims_inv c
    WHERE q.claim_id = c.id AND c.inverse = False
);

-- CREATE INDEXES --
CREATE INDEX idx_qualifiers_claim_id ON qualifiers (claim_id);

CREATE INDEX idx_labels_en_id ON labels_en (id);

-- CREATE INDEX idx_descriptions_id ON descriptions (id); --
CREATE INDEX idx_descriptions_en_id ON descriptions_en (id);

CREATE INDEX idx_entity_popularity_id ON entity_popularity (entity_id);