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
 entity_id        VARCHAR(16) REFERENCES entities(id),
 id               VARCHAR(64) PRIMARY KEY,
 type             claims_type,
 rank             claims_rank,
 snaktype         claims_snaktype,
 property         VARCHAR(16) REFERENCES entities(id),
 datavalue_string TEXT,
 datavalue_entity VARCHAR(64),
 datavalue_date   VARCHAR(16),
 datavalue_type   claims_datavalue_type,
 datatype         claims_datatype
);

CREATE TABLE qualifiers (
 claim_id           VARCHAR(64) REFERENCES claims(id),
 property           VARCHAR(16) REFERENCES entities(id),
 hash               VARCHAR(64),
 snaktype           claims_snaktype,
 qualifier_property VARCHAR(16) REFERENCES entities(id),
 datavalue_string   TEXT,
 datavalue_entity   VARCHAR(64),
 datavalue_date     VARCHAR(16),
 remove_me_later    TEXT,
 datavalue_type     claims_datavalue_type,
 datatype           claims_datatype,
 counter            VARCHAR(4),
 order_hash         VARCHAR(2)
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

\copy claims from 'csv/labels.txt' DELIMITER E'\t'

\copy qualifiers from 'csv/qualifiers.txt' DELIMITER E'\t'

\copy labels from 'csv/labels.txt' DELIMITER E'\t'

\copy aliases from 'csv/aliases.txt' DELIMITER E'\t'

\copy descriptions from 'csv/descriptions.txt' DELIMITER E'\t'

-- CREATE MATERIALIZED VIEWS --

CREATE MATERIALIZED VIEW labels_en AS (
    SELECT id, language, value FROM labels WHERE language = 'en'
);

CREATE MATERIALIZED VIEW properties AS (
    SELECT DISTINCT property AS id, datatype AS value FROM claims
);

CREATE MATERIALIZED VIEW claims_5m AS (
  SELECT * FROM claims
  WHERE (((substring(entity_id from '[0-9]+$'))::BIGINT) <= 5000000)
    AND (datavalue_entity IS NULL OR (((substring(entity_id from '[0-9]+$'))::BIGINT) <= 5000000))
);

-- CREATE INDEXES --

CREATE INDEX idx_claims_datavalue_entity ON claims (datavalue_entity);
CREATE INDEX idx_claims_entity_id ON claims (entity_id);
CREATE INDEX idx_claims_property ON claims (property);

CREATE INDEX idx_qualifiers_claim_id ON qualifiers (claim_id);
CREATE INDEX idx_qualifiers_datavalue_entity ON qualifiers (datavalue_entity);
CREATE INDEX idx_qualifiers_property ON qualifiers (property);