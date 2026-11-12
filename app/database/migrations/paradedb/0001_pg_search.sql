-- paradedb 0001: the pg_search extension, in the paradedb schema, for LEXICAL_BACKEND=paradedb
-- only (applied and required only then). VectorStoreManager created both inline at start-up
-- before. Not run by any test or results file: no image used here has pg_search (OD07).

CREATE SCHEMA IF NOT EXISTS paradedb;
CREATE EXTENSION IF NOT EXISTS pg_search;
