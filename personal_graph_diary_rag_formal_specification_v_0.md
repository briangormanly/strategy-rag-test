# Personal Graph Diary RAG — Formal Specification v0.1

**Purpose:** Define a minimal-yet-extensible specification for a prototype application whose core is a personal knowledge graph of diary entries. The system supports on‑the‑fly ontology growth, LLM‑assisted extraction/grounding, vector search, and a GraphRAG QA interface that answers strictly from a single person’s data.

**Status:** Draft (implementation-ready).\
**Target DB:** Neo4j 5.x (system of record) with native vector index.\
**Optional Sidecar:** External vector DB (Qdrant/pgvector) — not required for MVP.

---

## 1. High-Level Architecture

**Components**

- **API Gateway / App Server**: REST endpoints, authentication, query planning.
- **Graph DB (Neo4j)**: canonical storage of nodes/relationships & vector properties.
- **Worker(s)**: asynchronous extraction (NER/RE), entity resolution, embeddings, schema proposals, rollup summaries.
- **Embedding Provider**: model to compute entry and query embeddings (dimension D).
- **Object Storage**: media (photos/audio/docs) referenced from graph via URI.
- **Queue**: job dispatch (e.g., BullMQ/Redis or Celery/RabbitMQ).
- **Observability**: logs, metrics, audit events.

**Data Flow**

1. Client → `POST /entries` → create raw `DiaryEntry` → enqueue `extract`.
2. Worker runs NER/RE, geocoding, linking, embedding, vector indexing, summaries.
3. Client → `GET /qa?personId=&q=` → GraphRAG: structured filter → vector recall → neighborhood expansion → grounded answer with citations.

---

## 2. Ontology & Meta‑Schema

### 2.1 Core Labels (Closed Set)

- `Person`
- `DiaryEntry`
- `Event`
- `Place`
- `Organization`
- `Topic`
- `Media`
- `TimeInterval`

### 2.2 Core Relationships

- `(Person)-[:WROTE]->(DiaryEntry)`
- `(DiaryEntry)-[:DESCRIBES]->(Event)`
- `(DiaryEntry)-[:MENTIONS_PERSON]->(Person)`
- `(DiaryEntry)-[:AT_PLACE]->(Place)`
- `(DiaryEntry)-[:ABOUT_TOPIC]->(Topic)`
- `(DiaryEntry)-[:IN_ORG_CONTEXT]->(Organization)`
- `(DiaryEntry)-[:HAS_MEDIA]->(Media)`
- `(Event)-[:INVOLVES_PERSON]->(Person)`
- `(Event)-[:AT_PLACE]->(Place)`
- `(Event)-[:ABOUT_TOPIC]->(Topic)`
- `(DiaryEntry|Event)-[:WITHIN]->(TimeInterval)`

### 2.3 Node Properties (MVP)

- **Person**: `id: string (uuid)`, `fullName: string`, `createdAt: datetime`, `updatedAt: datetime`
- **DiaryEntry**: `id: string (uuid/deterministic)`, `personId: string`, `date: date`, `datetime: datetime?`, `weekday: int (1–7)`, `text: string`, `sentiment: float? (-1..1)`, `mood: string?`, `tokens: int?`, `privacy: enum('private','connections','public')`, `embedding: float[D]`, `confidence: float?`, `createdAt: datetime`, `updatedAt: datetime`
- **Event**: `id`, `title`, `startDate: date`, `endDate: date?`, `kind: string`, `description?: string`, `embedding?: float[D]`
- **Place**: `id`, `name`, `lat: float?`, `lon: float?`, `address?: map`, `externalIds?: map`
- **Organization**: `id`, `name`, `aliases?: string[]`, `externalIds?: map`
- **Topic**: `id`, `name`, `aliases?: string[]`
- **Media**: `id`, `uri: string`, `type: enum('image','audio','video','doc','other')`, `hash: string`, `capturedAt: datetime?`, `mime?: string`
- **TimeInterval**: `id`, `start: date`, `end: date`, `granularity: enum('day','week','month','year')`, `summary?: string`, `embedding?: float[D]`

### 2.4 On‑the‑Fly Schema Evolution (Meta‑Graph)

- **EntityType**: `{ name: string, canonical: boolean, parent?: string }`
- **RelType**: `{ name: string, domain: string, range: string, symmetric?: boolean, transitive?: boolean, cardinality?: string }`
- **SchemaProposal**: `{ id, kind: enum('EntityType','RelType'), name, suggestedParent?: string, rationale?: string, confidence: float, status: enum('pending','auto-accepted','rejected'), createdAt }`

Relationships:

- `(EntityType)-[:SUBTYPE_OF]->(EntityType)`
- `(SchemaProposal)-[:SUGGESTS]->(EntityType|RelType)`
- `(:Instance)-[:IS_A]->(EntityType)` for provisional tagging

**Auto‑Accept Rule (MVP):** if `confidence ≥ 0.85` and no name collision (case‑insensitive) within same parent → create canonical `EntityType` and set proposal to `auto-accepted`.

---

## 3. Neo4j DDL (Constraints & Indexes)

```cypher
// Uniqueness
CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT entry_id  IF NOT EXISTS FOR (e:DiaryEntry) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT event_id  IF NOT EXISTS FOR (n:Event) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT place_id  IF NOT EXISTS FOR (n:Place) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT org_id    IF NOT EXISTS FOR (n:Organization) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT topic_id  IF NOT EXISTS FOR (n:Topic) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT media_id  IF NOT EXISTS FOR (n:Media) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT interval_id IF NOT EXISTS FOR (n:TimeInterval) REQUIRE n.id IS UNIQUE;

// Filtering
CREATE INDEX entry_date IF NOT EXISTS FOR (e:DiaryEntry) ON (e.date);
CREATE INDEX entry_person IF NOT EXISTS FOR (e:DiaryEntry) ON (e.personId);
CREATE INDEX topic_name IF NOT EXISTS FOR (t:Topic) ON (t.name);
CREATE INDEX place_name IF NOT EXISTS FOR (p:Place) ON (p.name);

// Vector index (adjust D to your model)
CALL db.index.vector.createNodeIndex(
  'entry_embedding_idx', 'DiaryEntry', 'embedding', 1536, 'cosine'
);
```

---

## 4. IDs & Idempotency

- `DiaryEntry.id = uuidv5(namespace="urn:entry", value = personId + ":" + ISO8601(datetime or date) + ":" + sha256(text))`
- Upserts MUST match on `id`; duplicate POSTs are no‑ops.

---

## 5. Extraction & Resolution Pipeline (Worker)

**Stages (in order):**

1. **Preprocess**: basic cleanup, token count, weekday derivation (`weekday = dayOfWeek(date)`, ISO 1–7).
2. **NER/RE**: people, places, orgs, time expressions (normalize to absolute dates), event phrases, topics, sentiment/mood.
3. **Geocode**: places via dev geocoder (stage env), store lat/lon when available.
4. **Entity Resolution** (scoped to `personId` first, then global):
   - exact match → normalized string sim → embedding sim.
   - if ambiguous (`Δscore < τ_merge`), create `MatchCandidate` (ephemeral) and skip link until confirmed.
5. **Schema Mapping**: map extracted types/relations into known `EntityType`/`RelType`.
6. **Schema Proposal**: when missing, emit `SchemaProposal`; auto‑accept if rule satisfied.
7. **Linking**: create relationships from `DiaryEntry` to resolved nodes; create/attach `Event` when extraction elevates a specific occurrence.
8. **Embeddings**: compute and persist `embedding` for `DiaryEntry` (and optionally derived `Event` / `TimeInterval`).
9. **Rollups**: upsert `TimeInterval` (month/week/day) nodes and summaries.

**Worker Input Message (JSON):**

```json
{
  "jobType": "extractEntry",
  "entryId": "uuid",
  "personId": "uuid",
  "priority": 5,
  "traceId": "uuid"
}
```

**Worker Result:** updates graph; emits audit event.

---

## 6. GraphRAG Retrieval (QA)

**Inputs:** `personId: string`, `q: string`, `mode: 'factual'|'persona' (default 'factual')`, `k: int (default 10)`

**Planner:**

1. **Query Parse** (LLM/lightweight regex): extract time constraints, places, co‑mentioned people, topics.
2. **Structured Filter**: constrain by `personId` and parsed time window (e.g., June 2020) and weekday if present.
3. **Vector Recall**: if results < `k`, run vector search on `entry_embedding_idx` with query embedding.
4. **Neighborhood Expansion**: within 1–2 hops: `Place`, `Event`, `Topic`, `Media`.
5. **Answer Synthesis**: LLM composes strictly from retrieved snippets; include citations.

**Cypher Snippets**

*Saturday in June of a target year:*

```cypher
MATCH (p:Person {id: $personId})-[:WROTE]->(e:DiaryEntry)
WHERE e.date >= date({year: $year, month: 6, day: 1})
  AND e.date <  date({year: $year, month: 7, day: 1})
  AND e.weekday = 6
RETURN e ORDER BY e.date ASC LIMIT 25;
```

*Vector recall (query embedding supplied as \$qvec):*

```cypher
CALL db.index.vector.queryNodes('entry_embedding_idx', $k, $qvec)
YIELD node AS e, score
MATCH (p:Person {id: $personId})-[:WROTE]->(e)
RETURN e, score ORDER BY score DESC LIMIT $k;
```

---

## 7. REST API (OpenAPI 3.1)

```yaml
openapi: 3.1.0
info:
  title: Personal Graph Diary API
  version: 0.1.0
servers:
  - url: https://api.example.com
paths:
  /healthz:
    get:
      summary: Liveness probe
      responses:
        '200': { description: OK }

  /entries:
    post:
      summary: Create a diary entry (idempotent)
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateEntryRequest'
      responses:
        '201': { description: Created, content: { application/json: { schema: { $ref: '#/components/schemas/Entry' }}}}
        '409': { description: Already exists (returns current resource) }

  /entries/{entryId}:
    get:
      summary: Get a diary entry
      parameters:
        - in: path
          name: entryId
          required: true
          schema: { type: string }
      responses:
        '200': { description: OK, content: { application/json: { schema: { $ref: '#/components/schemas/Entry' }}}}
        '404': { description: Not found }

  /entries/{entryId}/extract:
    post:
      summary: Trigger (re)extraction pipeline for an entry
      responses:
        '202': { description: Accepted }

  /qa:
    get:
      summary: Answer a question grounded in a single person's data
      parameters:
        - in: query
          name: personId
          required: true
          schema: { type: string }
        - in: query
          name: q
          required: true
          schema: { type: string }
        - in: query
          name: mode
          schema: { type: string, enum: [factual, persona], default: factual }
        - in: query
          name: k
          schema: { type: integer, default: 10, minimum: 1, maximum: 100 }
      responses:
        '200': { description: OK, content: { application/json: { schema: { $ref: '#/components/schemas/QAResponse' }}}}

  /timeline:
    get:
      summary: List entries/events in a time window
      parameters:
        - in: query
          name: personId
          required: true
          schema: { type: string }
        - in: query
          name: from
          schema: { type: string, format: date }
        - in: query
          name: to
          schema: { type: string, format: date }
      responses:
        '200': { description: OK, content: { application/json: { schema: { $ref: '#/components/schemas/TimelineResponse' }}}}

  /schema/proposals:
    get:
      summary: List schema proposals
      responses:
        '200': { description: OK, content: { application/json: { schema: { type: array, 
```
