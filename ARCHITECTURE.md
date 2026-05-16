# Engram Architecture

**Version:** 1.0  
**Date:** February 16, 2026  
**Status:** Design Document

---

## Executive Summary

Engram is a conversation-native temporal knowledge graph engine that creates structured, persistent memory for AI systems. It extracts entities and relationships from conversations, tracks how they evolve over time, and enables temporal reasoning ("as of date X, what did we know about Y?").

**Core Innovation:** Bitemporal relationship versioning that tracks both when facts were true and when we learned them.

**Market Position:** Only OSS temporal knowledge graph purpose-built for conversations (not documents).

**License:** MIT core + commercial layer

---

## 1. Graph Architecture

### 1.1 Schema Design

#### Node Structure: Entity

```cypher
(:Entity {
  // Identity (deterministic, group-scoped)
  id: string,                  // "{tenant_id}:{group_id}:{type}:{canonical_name}"
  
  // SCOPING (MANDATORY - prevents cross-tenant contamination)
  tenant_id: string,           // Which customer/user owns this entity
  conversation_id: string,     // Which conversation created this entity
  group_id: string,            // Cross-conversation linking scope (defaults to conversation_id)
  
  // Entity properties
  type: enum,                  // Person | Concept | Goal | Preference | Topic | Event
  canonical_name: string,      // Normalized name ("kendra" - scoped, not global)
  aliases: [string],           // Variations seen ["Kendra", "K", "Kendra M"]
  
  // Semantic
  embedding: vector[1536],     // For entity resolution & similarity
  
  // Temporal
  created_at: datetime,        // First mention timestamp
  last_mentioned: datetime,    // Most recent mention (for decay calculations)
  
  // Provenance
  source_messages: [string],   // message_ids that created/updated this entity
  
  // Metadata
  metadata: map                // Domain-specific attributes
})
```

**Entity Types (Conversation-Native)**:

| Type | Description | Decay Rate | Examples |
|------|-------------|------------|----------|
| `Person` | Individuals mentioned in conversations | Low (0.005/day) | "Kendra Martinez", "Coach Sarah" |
| `Preference` | Likes, dislikes, favorites | High (0.05/day) | "Nike running shoes", "morning workouts" |
| `Goal` | Objectives, intentions | Medium (0.02/day) | "Run a marathon", "Lose 10 lbs" |
| `Concept` | Abstract ideas, topics | Low (0.01/day) | "mindfulness", "work-life balance" |
| `Event` | Time-bound occurrences | Medium (0.02/day) | "Boston Marathon 2026", "Q1 review meeting" |
| `Topic` | Conversation subjects | Medium (0.03/day) | "career planning", "fitness routine" |

**Why these types?**
- Derived from proven production use (coaching platform patterns)
- Conversation-native (vs document-centric types like "chunk" or "paragraph")
- Different decay rates reflect real-world information aging

#### Edge Structure: Versioned Relationships

```cypher
(:Entity)-[r:RELATIONSHIP {
  // SCOPING (inherit from endpoints, enforce isolation)
  tenant_id: string,           // Must match both endpoint tenant_ids
  conversation_id: string,     // Which conversation created this relationship
  group_id: string,            // Cross-conversation linking scope (defaults to conversation_id)
  message_id: string,          // Which specific message created this (for idempotency)
  
  // BITEMPORAL FIELDS (COMPLETE - 4 time columns)
  // VALID TIME: When was this true in the real world?
  valid_from: datetime,        // When fact became true
  valid_to: datetime,          // When fact stopped being true (NULL = still true)
  
  // RECORD TIME: When did we believe/know this?
  recorded_from: datetime,     // When we learned/recorded it
  recorded_to: datetime,       // When belief ended/retracted (NULL = still believed)
  
  // Relationship semantics
  type: string,                // "prefers", "knows", "manages", "discussed", "mentioned_with"
  confidence: float,           // 0.0-1.0 (initial extraction confidence)
  
  // Provenance
  evidence: string,            // Direct quote or paraphrased summary
  
  // Evolution tracking
  version: int,                // Relationship version (1, 2, 3...)
  supersedes: string,          // Previous relationship ID (if this replaces another)
  
  // Metadata
  metadata: map                // Domain-specific attributes
}]->(:Entity)
```

**Complete Bitemporal Model** (4 Time Columns):

```
Timeline Example: Kendra's shoe preference with correction

TRUTH Timeline (valid_from/valid_to): When fact was true in real world
  Adidas actually preferred since W0
  [==============Adidas==================>
  W0                                   now

KNOWLEDGE Timeline (recorded_from/recorded_to): When we believed facts
  Week 1-3: Believed Nike    Week 3+: Believed Adidas (correction)
  [----Nike belief----)  [----Adidas belief---->
  W1                 W3   W3                  now

Database Records:
  r1: valid_from=W1, valid_to=W3, recorded_from=W1, recorded_to=W3  (Nike - RETRACTED)
  r2: valid_from=W0, valid_to=NULL, recorded_from=W3, recorded_to=NULL (Adidas - ACTIVE)

Queries:
  "What was TRUE in Week 2?" → Adidas (valid_from=W0 <= W2 < valid_to=NULL)
  "What did we KNOW in Week 2?" → Nike (recorded_from=W1 <= W2 < recorded_to=W3)
  "What do we know NOW?" → Adidas (recorded_to=NULL)
```

**Why complete bitemporal (4 columns)?**
- **Valid time window** (`valid_from`/`valid_to`): When the relationship was ACTUALLY true in the real world
- **Record time window** (`recorded_from`/`recorded_to`): When we BELIEVED/KNEW this fact
- **Enables "knowledge vs truth" queries**: Separate what we knew from what was true
- **Enables corrections & audit trails**: Track when beliefs were retracted (`recorded_to` set)
- **Example use case**: "Coach corrects notes on Week 3: client actually started preferring Adidas 2 weeks ago" → `valid_from` backdated to W0, `recorded_from` is W3, previous Nike belief has `recorded_to=W3`

**Key insight**: Unlike competitors (Mem0, GraphRAG, LightRAG) which only track "when recorded" (`created_at`), Engram tracks BOTH "when true" AND "when known" with explicit end times for retractions.

### 1.2 Relationship Types

| Type | Source → Target | Semantics | Example |
|------|----------------|-----------|---------|
| `prefers` | Person → Preference | Positive sentiment toward entity | (Kendra)→(Nike shoes) |
| `avoids` | Person → Preference | Negative sentiment | (Kendra)→(early mornings) |
| `knows` | Person → Person | Social relationship | (Kendra)→(Coach Sarah) |
| `discussed` | Person → Topic | Conversation participation | (Kendra)→(marathon training) |
| `mentioned_with` | Entity → Entity | Co-occurrence | (Nike)→(marathon) |
| `has_goal` | Person → Goal | Objective ownership | (Kendra)→(run Boston Marathon) |
| `relates_to` | Entity → Entity | Generic relationship | (mindfulness)→(stress management) |

**Extensibility**: Users can define custom relationship types via configuration.

### 1.3 Indexing Strategy

```cypher
// Neo4j indexes for performance

// ENTITY INDEXES
CREATE INDEX entity_id FOR (e:Entity) ON (e.id);
CREATE INDEX entity_type FOR (e:Entity) ON (e.type);

// Scoping indexes (MANDATORY for tenant isolation)
CREATE INDEX entity_tenant FOR (e:Entity) ON (e.tenant_id);
CREATE INDEX entity_conversation FOR (e:Entity) ON (e.conversation_id);
CREATE INDEX entity_group FOR (e:Entity) ON (e.group_id);
CREATE COMPOSITE INDEX entity_tenant_conv FOR (e:Entity) ON (e.tenant_id, e.conversation_id);
CREATE COMPOSITE INDEX entity_tenant_canonical FOR (e:Entity) ON (e.tenant_id, e.canonical_name);

// Vector index (verify Neo4j Community Edition support, or use pgvector)
CREATE VECTOR INDEX entity_embedding FOR (e:Entity) ON (e.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};

// RELATIONSHIP INDEXES

// Scoping indexes (ALL queries MUST filter by tenant_id)
CREATE INDEX rel_tenant FOR ()-[r:RELATIONSHIP]-() ON (r.tenant_id);
CREATE INDEX rel_conversation FOR ()-[r:RELATIONSHIP]-() ON (r.conversation_id);
CREATE INDEX rel_message FOR ()-[r:RELATIONSHIP]-() ON (r.message_id);

// Temporal indexes (CRITICAL for bitemporal queries)
// Valid time (truth timeline)
CREATE INDEX rel_valid_from FOR ()-[r:RELATIONSHIP]-() ON (r.valid_from);
CREATE INDEX rel_valid_to FOR ()-[r:RELATIONSHIP]-() ON (r.valid_to);
CREATE COMPOSITE INDEX rel_valid_window FOR ()-[r:RELATIONSHIP]-() ON (r.valid_from, r.valid_to);

// Record time (knowledge timeline)
CREATE INDEX rel_recorded_from FOR ()-[r:RELATIONSHIP]-() ON (r.recorded_from);
CREATE INDEX rel_recorded_to FOR ()-[r:RELATIONSHIP]-() ON (r.recorded_to);
CREATE COMPOSITE INDEX rel_recorded_window FOR ()-[r:RELATIONSHIP]-() ON (r.recorded_from, r.recorded_to);

// Composite tenant+temporal (most common query pattern)
CREATE COMPOSITE INDEX rel_tenant_valid FOR ()-[r:RELATIONSHIP]-() 
  ON (r.tenant_id, r.valid_from, r.valid_to);
CREATE COMPOSITE INDEX rel_tenant_recorded FOR ()-[r:RELATIONSHIP]-() 
  ON (r.tenant_id, r.recorded_from, r.recorded_to);
```

**Why these indexes?**
- **Tenant scoping**: Prevents cross-tenant leakage, ALL queries filter by `tenant_id`
- **Conversation scoping**: Isolates entities/relationships per conversation
- **Valid time composite**: "World-state-as-of" queries filter on BOTH `valid_from` and `valid_to`
- **Record time composite**: "Knowledge-as-of" queries filter on BOTH `recorded_from` and `recorded_to`
- **Tenant+temporal composite**: Most queries filter by tenant AND time window (10-100x speedup)

**Note on NULL handling**: Neo4j indexes include NULL values (unlike some databases), so `valid_to IS NULL` (active facts) and `recorded_to IS NULL` (currently believed) are efficiently indexed.

---

## 2. Extraction Pipeline

### 2.1 Architecture Overview

```
┌─────────────────┐
│ Conversation    │
│ Message Input   │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Stage 1:            │
│ Entity Extraction   │ ◄── LLM (GPT-4o-mini / Claude Haiku)
└────────┬────────────┘
         │ entities: [{name, type, canonical}]
         ▼
┌─────────────────────┐
│ Stage 2:            │
│ Relationship        │ ◄── LLM + context (recent entities/rels)
│ Inference           │
└────────┬────────────┘
         │ relationships: [{source, target, type, confidence}]
         ▼
┌─────────────────────┐
│ Stage 3:            │
│ Conflict Resolution │ ◄── Temporal versioning logic
│ & Versioning        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Graph Update        │
│ (Append-Only)       │
└─────────────────────┘
```

**Design Principles**:
- **Incremental**: Process one message at a time (no batch re-indexing)
- **Idempotent**: Re-processing same message produces same result (via message_id tracking)
- **Append-only**: Never delete, only version (preserves history with bitemporal fields)
- **Deterministic**: Service assigns IDs, not LLM (replay-safe, consistent)
- **Scoped**: All entities/relationships isolated by tenant_id + conversation_id
- **LLM-agnostic**: Uses LiteLLM for provider abstraction

**Separation of Concerns**:
- **LLM**: Semantic analysis only (entity mentions, relationship types, confidence)
- **Service**: Deterministic operations (ID assignment, deduplication, graph writes)

### 2.2 Stage 1: Entity Extraction

**Goal**: Identify entities (people, preferences, goals, topics) and resolve to canonical forms.

**LLM Prompt**:

```
You are extracting entities from a conversation message for a knowledge graph.

Message: "{message_text}"
Timestamp: {message_timestamp}
Speaker: {speaker_name}

Context (entities mentioned recently in this conversation):
{context_entities}

Extract entities following these rules:

1. Entity Types:
   - Person: Named individuals (participants or mentioned people)
   - Preference: Things someone likes/dislikes (brands, foods, activities)
   - Goal: Stated objectives or intentions
   - Concept: Abstract ideas or topics being discussed
   - Event: Specific time-bound occurrences (meetings, deadlines, races)
   - Topic: Conversation subjects or themes

2. Canonicalization:
   - Normalize names: "Kendra M" → "Kendra Martinez" (use context)
   - Resolve pronouns: "she" → match to recent Person entity
   - Handle variations: "running shoes" = "running shoe" (singular form)
   - Merge synonyms: "marathon" = "marathon race"

3. Confidence:
   - Explicit mention: confidence = 1.0
   - Pronoun resolution: confidence = 0.8
   - Inferred from context: confidence = 0.6

Return JSON:
{
  "entities": [
    {
      "name": "exact text from message",
      "canonical": "normalized canonical name",
      "type": "Person|Preference|Goal|Concept|Event|Topic",
      "confidence": 0.0-1.0
    }
  ]
}

Only extract entities that are clearly present. Do not infer entities not mentioned.
```

**Processing** (Revised - Idempotent & Deterministic):

```python
def process_message(message: Message, tenant_id: str, conversation_id: str):
    """
    Process a single message with idempotency and deterministic ID assignment.
    
    Key changes from initial design:
    - FIRST: Check if message already processed (idempotency)
    - LLM returns semantic judgments only (NO UUIDs)
    - Service assigns deterministic, conversation-scoped IDs
    - All entities scoped by tenant_id + conversation_id
    """
    
    # 0. IDEMPOTENCY CHECK (FIRST STEP - prevent duplicate processing)
    if redis.exists(f"processed:{message.id}"):
        logger.info(f"Skipping duplicate message: {message.id}")
        return  # Already processed, skip
    
    # Mark as processed IMMEDIATELY (atomic operation, 24h TTL)
    redis.set(f"processed:{message.id}", "1", ex=86400, nx=True)
    
    # 1. Build conversation-scoped context (FILTER BY TENANT + CONVERSATION)
    recent_entities = graph.query("""
        MATCH (e:Entity)
        WHERE e.tenant_id = $tenant_id
          AND e.conversation_id = $conversation_id
          AND e.last_mentioned >= $recent_threshold
        RETURN e.canonical_name, e.type, e.aliases
        ORDER BY e.last_mentioned DESC
        LIMIT 20
    """, tenant_id=tenant_id, 
        conversation_id=conversation_id,
        recent_threshold=message.timestamp - timedelta(hours=1))
    
    # 2. LLM extracts SEMANTIC JUDGMENTS (not IDs)
    prompt = build_extraction_prompt(message, recent_entities)
    response = llm.complete(prompt, model="gpt-4o-mini", temperature=0.1)
    extracted = parse_json(response)  # {"entities": [{"mention": "Kendra", "type": "PERSON"}]}
    
    # 3. SERVICE assigns deterministic, conversation-scoped IDs
    entities = []
    for item in extracted["entities"]:
        # Normalize to canonical form (deterministic)
        canonical = normalize_entity_name(item["mention"])
        
        # Construct deterministic ID: {tenant}:{conversation}:{type}:{canonical}
        entity_id = f"{tenant_id}:{conversation_id}:{item['type']}:{canonical}"
        
        # Idempotent upsert (MERGE handles duplicates)
        entity = upsert_entity(
            id=entity_id,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            canonical_name=canonical,
            type=item["type"],
            aliases=[item["mention"]],
            message_id=message.id,
            timestamp=message.timestamp
        )
        
        entities.append(entity)
    
    return entities

def normalize_entity_name(text: str) -> str:
    """
    Deterministic normalization to canonical form.
    Same input → same output (critical for replay safety).
    """
    # Lowercase, strip punctuation, normalize whitespace
    normalized = re.sub(r'[^\w\s]', '', text.lower())
    normalized = re.sub(r'\s+', '-', normalized.strip())
    return normalized

def upsert_entity(id, tenant_id, conversation_id, canonical_name, type, aliases, message_id, timestamp):
    """
    Idempotent entity creation using Neo4j MERGE.
    
    ON CREATE: Set all fields
    ON MATCH: Update last_mentioned, append to source_messages
    """
    result = graph.query("""
        MERGE (e:Entity {id: $id})
        ON CREATE SET
            e.tenant_id = $tenant_id,
            e.conversation_id = $conversation_id,
            e.canonical_name = $canonical_name,
            e.type = $type,
            e.aliases = $aliases,
            e.created_at = $timestamp,
            e.last_mentioned = $timestamp,
            e.source_messages = [$message_id]
        ON MATCH SET
            e.last_mentioned = $timestamp,
            e.aliases = e.aliases + [alias IN $aliases WHERE NOT alias IN e.aliases],
            e.source_messages = e.source_messages + $message_id
        RETURN e
    """, id=id, tenant_id=tenant_id, conversation_id=conversation_id,
        canonical_name=canonical_name, type=type, aliases=aliases,
        message_id=message_id, timestamp=timestamp)
    
    return result[0]["e"]
```

**Key Improvements**:
- ✅ **Idempotent**: Redis SET NX prevents processing same message twice
- ✅ **Deterministic**: Same message → same entity IDs (via canonical normalization)
- ✅ **Conversation-scoped**: IDs include tenant_id + conversation_id (no collisions)
- ✅ **LLM does semantics only**: No UUIDs in LLM output (service assigns IDs)
- ✅ **Replay-safe**: MERGE handles entity already existing
- ✅ **Scoped context**: Only fetches entities from THIS conversation, not global

### 2.3 Stage 2: Relationship Inference

**Goal**: Extract relationships between entities mentioned in the message.

**LLM Prompt**:

```
You are inferring relationships between entities from a conversation message.

Message: "{message_text}"
Timestamp: {message_timestamp}
Speaker: {speaker_name}

Extracted Entities:
{entities}

Context (existing relationships involving these entities):
{existing_relationships}

Infer relationships following these rules:

1. Relationship Types:
   - prefers: Person → Preference (positive sentiment)
   - avoids: Person → Preference (negative sentiment)
   - knows: Person → Person (social connection)
   - discussed: Person → Topic (conversation participation)
   - mentioned_with: Entity → Entity (co-occurrence, no strong semantic link)
   - has_goal: Person → Goal (ownership of objective)
   - relates_to: Entity → Entity (generic semantic relationship)

2. Evidence:
   - Only infer relationships explicitly stated or strongly implied
   - Provide direct quote or paraphrased evidence
   - Mark uncertain inferences with confidence < 1.0

3. Temporal Markers:
   - "now prefers" → new relationship, invalidate old
   - "used to like" → relationship with valid_to = now
   - "always loved" → relationship with valid_from = distant past
   - "recently switched to" → relationship transition

4. Confidence Scoring:
   - Direct statement: 1.0 ("I love Nike shoes")
   - Strong implication: 0.8 ("Nike is the best")
   - Weak inference: 0.6 ("mentioned Nike positively")
   - Co-occurrence only: 0.4 ("talked about Nike and running")

Return JSON (SEMANTIC JUDGMENTS ONLY - NO IDs):
{
  "relationships": [
    {
      "source_mention": "exact entity mention from message",
      "target_mention": "exact entity mention from message",
      "type": "relationship type",
      "confidence": 0.0-1.0,
      "evidence": "quote or summary",
      "temporal_marker": "new|update|past|ongoing"
    }
  ]
}

CRITICAL: Return entity MENTIONS (text from message), NOT IDs or UUIDs.
Example: {"source_mention": "Kendra", "target_mention": "Nike shoes"}
```

**Processing** (Revised - Service Maps Mentions to IDs):

```python
def infer_relationships(
    message: Message,
    tenant_id: str,
    conversation_id: str,
    entities: list[Entity]
) -> list[Relationship]:
    """
    Infer relationships with SERVICE performing ID mapping.
    
    LLM returns mentions, service maps to deterministic IDs.
    """
    # Build entity mention → ID lookup (from Stage 1 results)
    entity_lookup = {
        normalize_entity_name(e.canonical_name): e.id
        for e in entities
    }
    
    # Get existing relationships for context (SCOPED to conversation)
    existing_rels = graph.query("""
        MATCH (a:Entity)-[r:RELATIONSHIP]->(b:Entity)
        WHERE r.tenant_id = $tenant_id
          AND r.conversation_id = $conversation_id
          AND (a.id IN $entity_ids OR b.id IN $entity_ids)
          AND r.valid_to IS NULL
          AND r.recorded_to IS NULL
        RETURN a.canonical_name AS source, 
               r.type AS type, 
               b.canonical_name AS target,
               r.confidence AS confidence
        ORDER BY r.recorded_from DESC
        LIMIT 10
    """, tenant_id=tenant_id, conversation_id=conversation_id,
        entity_ids=[e.id for e in entities])
    
    # Call LLM (returns MENTIONS, not IDs)
    prompt = build_inference_prompt(message, entities, existing_rels)
    response = llm.complete(prompt, model="gpt-4o-mini", temperature=0.2)
    inferred = parse_json(response)
    
    # SERVICE maps mentions to IDs deterministically
    relationships = []
    for item in inferred["relationships"]:
        # Normalize mentions and lookup IDs
        source_canonical = normalize_entity_name(item["source_mention"])
        target_canonical = normalize_entity_name(item["target_mention"])
        
        source_id = entity_lookup.get(source_canonical)
        target_id = entity_lookup.get(target_canonical)
        
        if not source_id or not target_id:
            logger.warning(f"Skipping relationship: entities not found ({item})")
            continue  # Skip if entities don't exist
        
        # Parse temporal marker
        valid_from = parse_temporal_marker(
            item.get("temporal_marker", "now"),
            message.timestamp
        )
        
        # Create relationship with COMPLETE bitemporal fields
        rel = Relationship(
            # Scoping
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            message_id=message.id,
            
            # Endpoints
            source_id=source_id,
            target_id=target_id,
            
            # Semantics
            type=item["type"],
            confidence=item["confidence"],
            evidence=item["evidence"],
            
            # Bitemporal (4 columns)
            valid_from=valid_from,
            valid_to=None,                      // Initially open-ended (truth)
            recorded_from=message.timestamp,    // When we learned it
            recorded_to=None,                   // Currently believed
            
            # Versioning
            version=1,
            supersedes=None
        )
        relationships.append(rel)
    
    return relationships
```

**Temporal Marker Parsing**:

```python
def parse_temporal_marker(marker: str, message_timestamp: datetime) -> datetime:
    """
    Convert temporal language to timestamps.
    
    Examples:
    - "infer" → message_timestamp (default)
    - "now" → message_timestamp
    - "yesterday" → message_timestamp - 1 day
    - "last week" → message_timestamp - 7 days
    - "always" → message_timestamp - 1 year (arbitrary past)
    """
    if marker == "infer" or marker == "now":
        return message_timestamp
    elif marker == "yesterday":
        return message_timestamp - timedelta(days=1)
    elif marker == "last week":
        return message_timestamp - timedelta(weeks=1)
    elif marker == "always":
        return message_timestamp - timedelta(days=365)
    else:
        return parse_iso_datetime(marker)  # Explicit ISO string
```

### 2.4 Stage 3: Conflict Resolution & Versioning

**Goal**: Handle contradictions, corrections, and relationship evolution.

**Conflict Types**:

1. **Contradiction**: New relationship contradicts existing
   - Example: "I prefer Adidas" when previous = "I prefer Nike"
   - Action: Terminate old (set valid_to), create new version

2. **Correction**: User explicitly corrects previous statement
   - Example: "Actually, I said Adidas, not Nike"
   - Action: Invalidate incorrect relationship, create corrected version (backdate if appropriate)

3. **Reinforcement**: Re-statement of existing relationship
   - Example: "I still love Nike" (when Nike relationship exists)
   - Action: Increase confidence, update last_mentioned

4. **Refinement**: Adding detail to existing relationship
   - Example: "I prefer Nike Air Zoom specifically" (when "Nike" relationship exists)
   - Action: Create more specific relationship, keep general one

**Exclusivity Policies** (NEW - defines which relationships are mutually exclusive):

```python
# Define exclusivity rules per relationship type
EXCLUSIVITY_POLICIES = {
    # Preference relationships: Only one active per category
    "prefers": {
        "exclusivity_scope": ("source",),  # Same source = exclusive
        "max_active": 1,                    # Only one active preference
        "close_on_new": True                # Auto-close old when new created
    },
    
    # Employment: Only one active job at a time
    "works_for": {
        "exclusivity_scope": ("source",),
        "max_active": 1,
        "close_on_new": True
    },
    
    # Goals: Multiple allowed
    "has_goal": {
        "exclusivity_scope": None,
        "max_active": None,
        "close_on_new": False
    },
    
    # Social relationships: Multiple allowed
    "knows": {
        "exclusivity_scope": None,
        "max_active": None,
        "close_on_new": False
    },
    
    # Opposites are mutually exclusive
    "avoids": {
        "exclusive_with": ["prefers"],  # Can't prefer and avoid same thing
        "close_on_new": True
    }
}
```

**Algorithm** (Revised - Enforces Exclusivity):

```python
def resolve_conflicts(new_rel: Relationship, tenant_id: str, group_id: str):
    """
    Apply exclusivity policies and temporal versioning.
    
    Key change: Checks for SAME SOURCE + SAME TYPE (not same target).
    This allows "Kendra prefers Nike" → "Kendra prefers Adidas" to close Nike.
    """
    policy = EXCLUSIVITY_POLICIES.get(new_rel.type, {"close_on_new": False})
    
    if not policy.get("close_on_new"):
        # No exclusivity, just create
        return create_relationship(new_rel)
    
    # Find existing active relationships of SAME TYPE from SAME SOURCE
    existing = graph.query("""
        MATCH (a:Entity {id: $source_id})-[r:RELATIONSHIP {type: $type}]->(b:Entity)
        WHERE r.tenant_id = $tenant_id
          AND r.group_id = $group_id
          AND r.valid_to IS NULL           // Currently valid (truth)
          AND r.recorded_to IS NULL        // Currently believed (knowledge)
        RETURN r, b
        ORDER BY r.recorded_from DESC
    """, source_id=new_rel.source_id, type=new_rel.type,
        tenant_id=tenant_id, group_id=group_id)
    
    # Check exclusivity
    for old_rel, old_target in existing:
        if violates_exclusivity(old_rel, old_target, new_rel, policy):
            # TERMINATE old relationship (BOTH timelines)
            graph.update(old_rel.id,
                valid_to=new_rel.valid_from,         // Close truth window
                recorded_to=new_rel.recorded_from    // Close knowledge window
            )
            
            # Link versions
            new_rel.supersedes = old_rel.id
            new_rel.version = old_rel.version + 1
            
            logger.info(f"Closed relationship {old_rel.id} due to exclusivity with {new_rel}")
    
    # Create new relationship
    return create_relationship(new_rel)

def violates_exclusivity(old_rel, old_target, new_rel, policy):
    """
    Check if new relationship violates exclusivity with old.
    
    Key logic: For "prefers" with scope=("source",), ANY preference 
    from same source is exclusive (even different targets).
    """
    if policy.get("exclusivity_scope") is None:
        return False  # No exclusivity
    
    # Check scope
    if "source" in policy["exclusivity_scope"]:
        # Same source = exclusive
        if old_rel.source_id != new_rel.source_id:
            return False
        
        # Check if targets are different (new preference)
        if old_target.id != new_rel.target_id:
            return True  # EXCLUSIVE: different targets, same source+type
    
    # Check explicit exclusivity (e.g., "prefers" vs "avoids")
    if "exclusive_with" in policy:
        if new_rel.type in policy["exclusive_with"]:
            return True
    
    return False

def create_relationship(rel: Relationship):
    """Create relationship in graph with idempotent MERGE."""
    graph.query("""
        MERGE (a:Entity {id: $source_id})
        MERGE (b:Entity {id: $target_id})
        CREATE (a)-[r:RELATIONSHIP {
            tenant_id: $tenant_id,
            conversation_id: $conversation_id,
            message_id: $message_id,
            type: $type,
            valid_from: $valid_from,
            valid_to: $valid_to,
            recorded_from: $recorded_from,
            recorded_to: $recorded_to,
            confidence: $confidence,
            evidence: $evidence,
            version: $version,
            supersedes: $supersedes
        }]->(b)
        RETURN r
    """, **rel.to_dict())
```

**Example** (Enforcing Single Active Preference):

```cypher
// Message 1: "Kendra prefers Nike"
CREATE (k)-[r1:PREFERS {
  valid_from: datetime('2024-01-15'),
  valid_to: NULL,
  recorded_from: datetime('2024-01-15'),
  recorded_to: NULL,
  version: 1
}]->(nike)

// Message 2: "Kendra now prefers Adidas"
// → Conflict detected (same source "Kendra", same type "prefers")
// → Close r1:

SET r1.valid_to = datetime('2024-03-20'),
    r1.recorded_to = datetime('2024-03-20')

// → Create r2:
CREATE (k)-[r2:PREFERS {
  valid_from: datetime('2024-03-20'),
  valid_to: NULL,
  recorded_from: datetime('2024-03-20'),
  recorded_to: NULL,
  version: 2,
  supersedes: id(r1)
}]->(adidas)

// Query: "What does Kendra prefer NOW?"
MATCH (k)-[r:PREFERS]->(brand)
WHERE r.valid_to IS NULL AND r.recorded_to IS NULL
RETURN brand
// → Adidas (only one active)
```

---

**Concrete Cypher Enforcement** (The Actual Queries That Run):

The Python code above translates to these specific Cypher queries that enforce the "single active preference" guarantee:

**Step 1: Find Conflicting Relationships**

```cypher
// Find existing active "prefers" relationships from same source
MATCH (source:Entity {id: $source_id})-[r:RELATIONSHIP {type: "prefers"}]->(target:Entity)
WHERE r.tenant_id = $tenant_id
  AND r.conversation_id = $conversation_id
  AND r.valid_to IS NULL           // Currently valid (truth timeline)
  AND r.recorded_to IS NULL        // Currently believed (knowledge timeline)
RETURN r, target
ORDER BY r.recorded_from DESC

// Example result: [(r1:PREFERS → Nike)]
```

**Step 2: Terminate Conflicting Relationships (Atomic)**

```cypher
// Close ALL active "prefers" relationships from this source
MATCH (source:Entity {id: $source_id})-[r:RELATIONSHIP {type: "prefers"}]->(old_target:Entity)
WHERE r.tenant_id = $tenant_id
  AND r.conversation_id = $conversation_id
  AND r.valid_to IS NULL
  AND r.recorded_to IS NULL
  AND old_target.id <> $new_target_id  // Don't close if same target (reinforcement)
SET r.valid_to = $new_valid_from,
    r.recorded_to = $new_recorded_from
RETURN count(r) AS closed_count

// Example: Closes r1 (Nike), returns closed_count=1
```

**Step 3: Create New Relationship**

```cypher
// Create new "prefers" relationship
MERGE (source:Entity {id: $source_id})
MERGE (target:Entity {id: $new_target_id})
CREATE (source)-[r:RELATIONSHIP {
    tenant_id: $tenant_id,
    conversation_id: $conversation_id,
    message_id: $message_id,
    type: "prefers",
    
    // Bitemporal fields
    valid_from: $new_valid_from,
    valid_to: NULL,
    recorded_from: $new_recorded_from,
    recorded_to: NULL,
    
    // Versioning
    version: $version,         // Incremented from old relationship
    supersedes: $old_rel_id,   // Link to terminated relationship
    
    confidence: $confidence,
    evidence: $evidence
}]->(target)
RETURN r
```

**Invariant Guarantee**:

After these three queries execute, the following invariant MUST hold:

```cypher
// INVARIANT: At most ONE active "prefers" relationship per source
MATCH (source:Entity {id: $source_id})-[r:RELATIONSHIP {type: "prefers"}]->(target:Entity)
WHERE r.tenant_id = $tenant_id
  AND r.conversation_id = $conversation_id
  AND r.valid_to IS NULL
  AND r.recorded_to IS NULL
RETURN count(r) AS active_count

// MUST return: active_count = 1 (or 0 if all terminated)
```

**Transaction Wrapping** (Ensures Atomicity):

```python
def enforce_exclusivity_and_create(new_rel, tenant_id, conversation_id):
    """
    Atomically enforce exclusivity and create new relationship.
    
    Uses Neo4j transaction to ensure Steps 2+3 are atomic (no race conditions).
    """
    with graph.session() as session:
        with session.begin_transaction() as tx:
            # Step 2: Close conflicting relationships
            result = tx.run("""
                MATCH (source:Entity {id: $source_id})-[r:RELATIONSHIP {type: $type}]->(old_target)
                WHERE r.tenant_id = $tenant_id
                  AND r.conversation_id = $conversation_id
                  AND r.valid_to IS NULL
                  AND r.recorded_to IS NULL
                  AND old_target.id <> $new_target_id
                SET r.valid_to = $new_valid_from,
                    r.recorded_to = $new_recorded_from
                RETURN count(r) AS closed_count
            """, source_id=new_rel.source_id, type=new_rel.type,
                tenant_id=tenant_id, conversation_id=conversation_id,
                new_target_id=new_rel.target_id,
                new_valid_from=new_rel.valid_from,
                new_recorded_from=new_rel.recorded_from)
            
            closed_count = result.single()["closed_count"]
            logger.info(f"Closed {closed_count} conflicting relationships")
            
            # Step 3: Create new relationship
            tx.run("""
                MERGE (source:Entity {id: $source_id})
                MERGE (target:Entity {id: $target_id})
                CREATE (source)-[r:RELATIONSHIP {
                    tenant_id: $tenant_id,
                    conversation_id: $conversation_id,
                    message_id: $message_id,
                    type: $type,
                    valid_from: $valid_from,
                    valid_to: $valid_to,
                    recorded_from: $recorded_from,
                    recorded_to: $recorded_to,
                    confidence: $confidence,
                    evidence: $evidence,
                    version: $version,
                    supersedes: $supersedes
                }]->(target)
            """, **new_rel.to_dict(), tenant_id=tenant_id, 
                conversation_id=conversation_id)
            
            # Commit transaction (atomic)
            tx.commit()
```

**Why This Enforces "Single Active Preference"**:

1. **Step 2 is scope-limited**: Only closes relationships with `type="prefers"` from `source_id` (not global)
2. **Both timelines checked**: `valid_to IS NULL AND recorded_to IS NULL` ensures we only close currently-active relationships
3. **Transaction atomicity**: Steps 2+3 wrapped in transaction prevents race conditions
4. **Different target check**: `old_target.id <> $new_target_id` prevents closing relationship when re-asserting same preference (reinforcement case)

**Test Case** (Verify Invariant):

```cypher
// Setup: Create two "prefers" relationships (violates invariant)
CREATE (k:Entity {id: "t1:c1:PERSON:kendra"})-[r1:PREFERS {valid_to: NULL, recorded_to: NULL}]->(nike:Entity {id: "t1:c1:PREF:nike"})
CREATE (k)-[r2:PREFERS {valid_to: NULL, recorded_to: NULL}]->(adidas:Entity {id: "t1:c1:PREF:adidas"})

// Check invariant (SHOULD FAIL - this is the bug we're preventing)
MATCH (k)-[r:PREFERS]->(target)
WHERE r.valid_to IS NULL AND r.recorded_to IS NULL
RETURN count(r)
// Returns: 2 (VIOLATION!)

// Fix: Run enforcement
[Run Step 2 Cypher to close r1, keeping only r2]

// Re-check invariant (SHOULD PASS)
MATCH (k)-[r:PREFERS]->(target)
WHERE r.valid_to IS NULL AND r.recorded_to IS NULL
RETURN count(r)
// Returns: 1 (CORRECT!)
```

**Correction Detection**:

```python
def detect_correction(message: Message) -> bool:
    """
    Detect correction signals in message text.
    
    Signals:
    - "Actually, ..."
    - "I meant ..."
    - "Correction: ..."
    - "No, I said ..."
    - "Sorry, I misspoke ..."
    """
    correction_patterns = [
        r"^actually,?\s",
        r"\bi meant\b",
        r"^correction:",
        r"^no,?\s",
        r"sorry,?\s.*misspoke",
        r"let me clarify"
    ]
    
    text = message.text.lower()
    return any(re.search(pattern, text) for pattern in correction_patterns)

def handle_correction(message: Message, tenant_id: str, conversation_id: str, new_rels: list[Relationship]):
    """
    When correction detected:
    1. Find recently created relationships (last 5 messages)
    2. RETRACT them (set recorded_to = now, marking belief ended)
    3. Create corrected versions with higher confidence
    
    Key: Use RECORDED_TO (knowledge window), not valid_to (truth window).
    The fact was never true, so we're retracting the BELIEF, not the truth.
    """
    if not detect_correction(message):
        return new_rels
    
    # Find recent relationships (potentially incorrect, SCOPED to conversation)
    recent_window = message.timestamp - timedelta(minutes=10)
    recent_rels = graph.query("""
        MATCH ()-[r:RELATIONSHIP]->()
        WHERE r.tenant_id = $tenant_id
          AND r.conversation_id = $conversation_id
          AND r.recorded_from >= $recent_window
          AND r.recorded_to IS NULL
        RETURN r
        ORDER BY r.recorded_from DESC
        LIMIT 5
    """, tenant_id=tenant_id, conversation_id=conversation_id,
        recent_window=recent_window)
    
    # RETRACT (not delete) incorrect beliefs
    for new_rel in new_rels:
        for old_rel in recent_rels:
            if (new_rel.source_id == old_rel.source_id and 
                new_rel.target_id != old_rel.target_id and
                new_rel.type == old_rel.type):
                # Same source, different target, same type → correction
                
                # RETRACT the incorrect belief
                graph.update_relationship(old_rel.id,
                    recorded_to=message.timestamp,  // Belief ended (KNOWLEDGE)
                    valid_to=old_rel.valid_from,    // Was never true (TRUTH)
                    confidence=0                     // Mark as incorrect
                )
                
                # Corrected facts get high confidence
                new_rel.confidence = 0.95
                
                logger.info(f"Retracted incorrect belief {old_rel.id} via correction")
    
    return new_rels
```

### 2.5 Performance & Latency

**Target**: 50-200ms per message processing

**Breakdown**:
- LLM Stage 1 (entity extraction): 30-80ms (cached embeddings, small prompt)
- LLM Stage 2 (relationship inference): 40-100ms (larger prompt, more context)
- Conflict resolution: 5-10ms (Cypher queries with indexes)
- Graph writes: 5-10ms (append-only, no complex transactions)

**Optimizations**:
- Use fast models (GPT-4o-mini, Claude Haiku) for extraction
- Cache embeddings for known entities
- Batch Neo4j writes (use UNWIND for multiple relationships)
- Parallel LLM calls when possible (Stage 1 and Stage 2 can run concurrently if Stage 2 doesn't need Stage 1 output)

---

## 3. Temporal Reasoning

### 3.1 Decay Function

**Goal**: Model how relationship confidence decreases over time when not reinforced.

**Formula**: Exponential decay with type-specific rates

```python
import math
from datetime import datetime, timedelta

# Decay rates (per day) by relationship type
DECAY_RATES = {
    # Entity types
    "Preference": 0.05,      # Preferences change quickly
    "Goal": 0.02,            # Goals persist longer
    "Person": 0.005,         # People relationships stable
    "Concept": 0.01,         # Conceptual understanding moderately stable
    "Topic": 0.03,           # Topics fade moderately fast
    "Event": 0.02,           # Events decay as they pass
    
    # Relationship types
    "prefers": 0.05,         # Preferences volatile
    "avoids": 0.04,          # Avoidances moderately volatile
    "knows": 0.005,          # Social bonds stable
    "discussed": 0.03,       # Discussion topics fade
    "mentioned_with": 0.1,   # Co-mentions fade quickly
    "has_goal": 0.02,        # Goals persist
    "relates_to": 0.01,      # Conceptual links stable
    
    # Default
    "default": 0.01
}

def calculate_current_confidence(
    relationship: Relationship,
    current_time: datetime
) -> float:
    """
    Calculate decayed confidence based on time since last mention.
    
    Formula: confidence(t) = base_confidence * exp(-decay_rate * days_elapsed)
    
    Confidence floor: 0.1 (never fully forget)
    """
    base_confidence = relationship.confidence
    last_mention = relationship.last_mentioned or relationship.created_at
    
    days_elapsed = (current_time - last_mention).total_seconds() / 86400
    
    # Get decay rate (check relationship type, then entity type, then default)
    decay_rate = DECAY_RATES.get(
        relationship.type,
        DECAY_RATES.get(relationship.target.type, DECAY_RATES["default"])
    )
    
    # Exponential decay
    decayed = base_confidence * math.exp(-decay_rate * days_elapsed)
    
    # Floor at 0.1 (weak but not forgotten)
    return max(decayed, 0.1)

def apply_decay_to_query(query_results: list[Relationship], current_time: datetime):
    """
    Apply decay function to query results before returning to user.
    """
    for rel in query_results:
        rel.current_confidence = calculate_current_confidence(rel, current_time)
    
    # Sort by decayed confidence
    return sorted(query_results, key=lambda r: r.current_confidence, reverse=True)
```

**Decay Visualization**:

```
Confidence over time (Preference, decay_rate=0.05/day):

1.0 |●
    |  ●
0.8 |    ●●
    |       ●●
0.6 |          ●●●
    |              ●●●●
0.4 |                   ●●●●●
    |                         ●●●●●●●
0.2 |                                ●●●●●●●
0.1 |_____________________________________●●●●●●●●
    0   10   20   30   40   50   60   70   80 (days)

After 30 days: confidence ≈ 0.22 (needs reinforcement)
After 60 days: confidence = 0.1 (floor, weak signal)
```

**Reinforcement** (boost confidence when re-mentioned):

```python
def reinforce_relationship(rel: Relationship, new_mention_confidence: float):
    """
    When relationship is re-mentioned, boost confidence.
    
    Strategy: Weighted average favoring new mention.
    """
    current_decayed = calculate_current_confidence(rel, datetime.now())
    
    # 70% new, 30% old (favors recent information)
    boosted = (new_mention_confidence * 0.7) + (current_decayed * 0.3)
    
    # Update relationship
    graph.update_relationship(
        rel.id,
        confidence=min(boosted, 1.0),  # Cap at 1.0
        last_mentioned=datetime.now()
    )
```

### 3.2 Temporal Query Modes (Knowledge vs Truth)

**Core Innovation**: Separate "what we knew" from "what was true" using bitemporal model.

**Three query modes**:
1. **Knowledge-As-Of** (Transaction Time): What did we BELIEVE on date X?
2. **World-State-As-Of** (Valid Time): What was ACTUALLY TRUE on date X (regardless of when we learned it)?
3. **Bitemporal** (Both): What did we believe on date X about the state on date Y?

---

#### Mode 1: Knowledge-As-Of (Transaction Time)

**Goal**: "What did we KNOW/BELIEVE on date X?"

**Use case**: Audit trail, "What did the system believe when this decision was made?"

**Query Pattern**:

```cypher
// Query KNOWLEDGE timeline (recorded_from/recorded_to)
MATCH (subject:Entity)-[r:RELATIONSHIP]->(related:Entity)
WHERE subject.tenant_id = $tenant_id
  AND subject.canonical_name = $entity_name
  
  // KNOWLEDGE WINDOW (what we believed)
  AND $query_date >= r.recorded_from
  AND ($query_date < r.recorded_to OR r.recorded_to IS NULL)
  
RETURN 
  related.canonical_name AS entity,
  related.type AS type,
  r.type AS relationship,
  r.confidence AS confidence,
  r.evidence AS evidence,
  r.recorded_from AS known_since,
  r.recorded_to AS known_until
ORDER BY r.confidence DESC
LIMIT 20
```

**Example**:
```
Query: "What did we know about Kendra's shoe preference on March 1?"
Result: Nike (because we believed Nike from Jan 15 to Mar 20)
```

**API Endpoint**:

```python
@app.get("/query/knowledge-as-of")
def knowledge_as_of_query(
    tenant_id: str,
    entity: str,
    as_of: datetime,
    relationship_type: str | None = None
):
    """
    Query what we KNEW/BELIEVED at a specific point in time.
    
    Example:
    GET /query/knowledge-as-of?tenant=t1&entity=Kendra&as_of=2024-03-01T00:00:00Z
    """
    query = """
        MATCH (subject:Entity)-[r:RELATIONSHIP]->(related:Entity)
        WHERE subject.tenant_id = $tenant_id
          AND subject.canonical_name = $entity
          AND $as_of_date >= r.recorded_from
          AND ($as_of_date < r.recorded_to OR r.recorded_to IS NULL)
    """
    
    if relationship_type:
        query += " AND r.type = $rel_type"
    
    query += """
        RETURN related, r
        ORDER BY r.confidence DESC
        LIMIT 20
    """
    
    return graph.query(query, tenant_id=tenant_id, entity=entity, 
                      as_of_date=as_of, rel_type=relationship_type)
```

---

#### Mode 2: World-State-As-Of (Valid Time)

**Goal**: "What was ACTUALLY TRUE on date X (regardless of when we learned it)?"

**Use case**: Historical reconstruction, "What was the actual state of the world?"

**Query Pattern**:

```cypher
// Query TRUTH timeline (valid_from/valid_to)
MATCH (subject:Entity)-[r:RELATIONSHIP]->(related:Entity)
WHERE subject.tenant_id = $tenant_id
  AND subject.canonical_name = $entity_name
  
  // TRUTH WINDOW (what was actually true)
  AND $query_date >= r.valid_from
  AND ($query_date < r.valid_to OR r.valid_to IS NULL)
  
RETURN 
  related.canonical_name AS entity,
  related.type AS type,
  r.type AS relationship,
  r.confidence AS confidence,
  r.evidence AS evidence,
  r.valid_from AS true_since,
  r.valid_to AS true_until
ORDER BY r.confidence DESC
LIMIT 20
```

**Example**:
```
Setup:
  Week 0: Kendra starts preferring Adidas (truth)
  Week 1: We record "Kendra prefers Nike" (incorrect belief)
  Week 3: We learn the truth, backdate to Week 0

Query: "What was ACTUALLY TRUE about Kendra's preference in Week 2?"
Result: Adidas (because valid_from=W0, even though we didn't know until W3)
```

**API Endpoint**:

```python
@app.get("/query/world-state-as-of")
def world_state_as_of_query(
    tenant_id: str,
    entity: str,
    as_of: datetime,
    relationship_type: str | None = None
):
    """
    Query what was ACTUALLY TRUE at a specific point in time.
    
    Example:
    GET /query/world-state-as-of?tenant=t1&entity=Kendra&as_of=2024-02-15T00:00:00Z
    """
    query = """
        MATCH (subject:Entity)-[r:RELATIONSHIP]->(related:Entity)
        WHERE subject.tenant_id = $tenant_id
          AND subject.canonical_name = $entity
          AND $as_of_date >= r.valid_from
          AND ($as_of_date < r.valid_to OR r.valid_to IS NULL)
    """
    
    if relationship_type:
        query += " AND r.type = $rel_type"
    
    query += """
        RETURN related, r
        ORDER BY r.confidence DESC
        LIMIT 20
    """
    
    return graph.query(query, tenant_id=tenant_id, entity=entity,
                      as_of_date=as_of, rel_type=relationship_type)
```

---

#### Mode 3: Bitemporal (Both Timelines)

**Goal**: "What did we BELIEVE on date X about the state on date Y?"

**Use case**: Complex temporal reasoning, "What did we know on March 25 about what was true on February 25?"

**Query Pattern**:

```cypher
// Query BOTH timelines simultaneously
MATCH (subject:Entity)-[r:RELATIONSHIP]->(related:Entity)
WHERE subject.tenant_id = $tenant_id
  AND subject.canonical_name = $entity_name
  
  // KNOWLEDGE at knowledge_date
  AND $knowledge_date >= r.recorded_from
  AND ($knowledge_date < r.recorded_to OR r.recorded_to IS NULL)
  
  // TRUTH at valid_date
  AND $valid_date >= r.valid_from
  AND ($valid_date < r.valid_to OR r.valid_to IS NULL)
  
RETURN 
  related.canonical_name AS entity,
  r.type AS relationship,
  r.valid_from AS true_from,
  r.valid_to AS true_to,
  r.recorded_from AS known_from,
  r.recorded_to AS known_to
```

**API Endpoint**:

```python
@app.get("/query/bitemporal")
def bitemporal_query(
    tenant_id: str,
    entity: str,
    knowledge_date: datetime,  // When we knew
    valid_date: datetime,       // What was true
    relationship_type: str | None = None
):
    """
    Query bitemporal intersection.
    
    Example:
    GET /query/bitemporal?tenant=t1&entity=Kendra
        &knowledge_date=2024-03-25&valid_date=2024-02-25
    
    Answer: "What did we believe on Mar 25 about what was true on Feb 25?"
    """
    # Implementation as above
```

---

**Comparison Table**:

| Query Mode | Timeline | Use Case | Example |
|------------|----------|----------|---------|
| **Knowledge-As-Of** | `recorded_from`/`recorded_to` | Audit trail | "What did we believe when decision made?" |
| **World-State-As-Of** | `valid_from`/`valid_to` | Historical truth | "What was actually true then?" |
| **Bitemporal** | Both | Complex reasoning | "What did we know on X about state on Y?" |

**Key Differentiator**: Competitors (Mem0, GraphRAG, LightRAG) only support knowledge-as-of (created_at). Engram supports ALL THREE modes.

### 3.3 Relationship Evolution Tracking

**Goal**: Show how a relationship changed over time.

**Query Pattern**:

```cypher
// Get all versions of a relationship between two entities
MATCH (a:Entity {canonical_name: $entity_a})-[r:RELATIONSHIP]->(b:Entity {canonical_name: $entity_b})
WHERE r.type = $relationship_type
RETURN 
  r.version AS version,
  r.valid_from AS started,
  r.valid_to AS ended,
  r.confidence AS confidence,
  r.evidence AS evidence,
  r.supersedes AS previous_version
ORDER BY r.valid_from ASC
```

**Visualization** (returned as timeline):

```json
{
  "entity_a": "Kendra Martinez",
  "entity_b": "Nike",
  "relationship_type": "prefers",
  "timeline": [
    {
      "version": 1,
      "period": "2024-01-15 to 2024-03-20",
      "confidence": 0.95,
      "evidence": "I love Nike running shoes, they're the best",
      "status": "superseded"
    },
    {
      "version": 2,
      "period": "2024-03-20 to present",
      "confidence": 0.85,
      "evidence": "Actually switched to Adidas, better arch support",
      "status": "active"
    }
  ]
}
```

**API Endpoint**:

```python
@app.get("/query/evolution")
def relationship_evolution(
    entity_a: str,
    entity_b: str | None = None,
    relationship_type: str | None = None
):
    """
    Track how relationships evolved over time.
    
    Examples:
    - GET /query/evolution?entity_a=Kendra&relationship_type=prefers
      → All preference changes for Kendra
    
    - GET /query/evolution?entity_a=Kendra&entity_b=Nike
      → All relationship changes between Kendra and Nike
    """
    # Build dynamic query based on parameters
    # Return timeline with versions, dates, confidence, evidence
```

### 3.4 Current State Queries

**Goal**: Get active relationships (most common use case).

**Query Pattern** (Revised - Tenant Scoped + Bitemporal):

```cypher
// Current active relationships (BOTH timelines must be active)
MATCH (subject:Entity)-[r:RELATIONSHIP]->(related:Entity)
WHERE subject.tenant_id = $tenant_id
  AND subject.canonical_name = $entity
  AND r.valid_to IS NULL          // Currently true (TRUTH timeline)
  AND r.recorded_to IS NULL       // Currently believed (KNOWLEDGE timeline)
RETURN related, r
ORDER BY r.confidence DESC
LIMIT 20
```

**API Endpoint** (Revised - Tenant Scoped + Bitemporal):

```python
@app.get("/entities/{entity_name}/relationships")
def get_current_relationships(
    tenant_id: str,
    entity_name: str,
    conversation_id: str | None = None,
    relationship_type: str | None = None
):
    """
    Get current active relationships for an entity, with decay applied.
    
    IMPORTANT: Filters by BOTH valid_to=NULL and recorded_to=NULL.
    """
    query = """
        MATCH (e:Entity)-[r:RELATIONSHIP]->(related:Entity)
        WHERE e.tenant_id = $tenant_id
          AND e.canonical_name = $entity
          AND r.valid_to IS NULL          // Currently true (truth timeline)
          AND r.recorded_to IS NULL       // Currently believed (knowledge timeline)
    """
    
    if conversation_id:
        query += " AND e.conversation_id = $conversation_id"
    
    if relationship_type:
        query += " AND r.type = $rel_type"
    
    query += " RETURN related, r ORDER BY r.recorded_from DESC"
    
    results = graph.query(query, 
                         tenant_id=tenant_id,
                         entity=entity_name,
                         conversation_id=conversation_id,
                         rel_type=relationship_type)
    
    # Apply decay function
    return apply_decay_to_query(results, datetime.now())
```

---

## 4. Core vs. Commercial Split

### 4.1 Philosophy

**MIT Core**: Fully functional engine. Self-hostable. No feature gimping.

**Commercial Layer**: Hosted service + enterprise features. Pay for convenience, compliance, and scale.

**Inspiration**: Plausible Analytics, PostHog, Mastodon (OSS core, commercial hosting)

### 4.2 MIT Licensed Core (`engram-core`)

**Repository**: `github.com/yri-ai/engram`

**Components**:

#### Graph Engine
- Entity & relationship CRUD operations
- Temporal query engine (point-in-time, evolution, current state)
- Bitemporal model implementation
- Decay function calculations
- Cypher query builders

#### Extraction Pipeline
- 3-stage LLM extraction (entity, relationship, conflict resolution)
- Prompt templates (customizable)
- LLM provider abstraction (LiteLLM integration)
- Entity resolution (embedding similarity)
- Conflict detection & versioning logic

#### Storage Adapters
- **Neo4j** (primary, production-grade)
  - Community Edition support
  - Aion extension compatibility (optional)
- **In-Memory** (for testing, demos)
  - Fast, ephemeral, no setup required
- **PostgreSQL + AGE** (community contribution target)
  - For users who want SQL + graphs

#### API Server
- REST API (FastAPI)
- Endpoints:
  - `POST /messages` - Ingest conversation message
  - `GET /entities` - List entities
  - `GET /entities/{id}` - Entity detail + relationships
  - `GET /entities/{id}/relationships` - Current relationships
  - `GET /query/point-in-time` - Historical query
  - `GET /query/evolution` - Relationship timeline
  - `GET /search` - Search entities (text + vector)
- OpenAPI documentation (auto-generated)

#### SDKs
- **Python** (first-class)
  ```python
  from engram import Engram
  
  client = Engram(url="http://localhost:8000")
  client.ingest("Kendra loves Nike shoes")
  rels = client.query(entity="Kendra", relationship_type="prefers")
  ```
- **JavaScript/TypeScript** (community contribution target)

#### CLI Tools
```bash
engram init                    # Setup database, create schema
engram ingest file.json        # Batch import conversations
engram query "Kendra prefers"  # Natural language query
engram export --format json    # Export graph
engram serve                   # Start API server
```

#### Web UI (Basic)
- Graph visualization (react-force-graph or similar)
- Entity browser (list, search, detail view)
- Query interface (forms for point-in-time, evolution queries)
- Conversation log viewer (show messages → extracted entities/rels)

**No Authentication** in core (single-user, localhost deployment assumed)

#### Documentation
- Quickstart guide (Docker Compose → running in 5 minutes)
- Architecture docs (this document)
- API reference (OpenAPI)
- Example use cases:
  - Coaching scenario (Kendra's shoe preferences)
  - Personal knowledge graph (your own conversations)
  - Customer support memory (track user issues over time)
- Deployment guides (Docker, Kubernetes, fly.io)

**License**: MIT

**Size Estimate**: 15-20K lines of Python + TypeScript

---

### 4.3 Commercial Layer (`engram-cloud`)

**Repository**: Private (closed-source)

**Business Model**: Hosted SaaS + Enterprise features

**Pricing** (indicative):
- **Free Tier**: 1,000 messages/month, 1 graph, community support
- **Pro**: $49/month - 50K messages, 5 graphs, email support
- **Team**: $199/month - 500K messages, unlimited graphs, Slack support, SSO
- **Enterprise**: Custom - unlimited, on-prem option, SLA, custom schemas, audit trails

#### Hosted Multi-Tenant Platform

**Features**:
- Managed Neo4j clusters (automatic scaling, backups, updates)
- Multi-tenancy isolation (separate graphs per organization)
- Global CDN (low-latency API access)
- Monitoring & alerting (Prometheus, Grafana dashboards)
- Usage analytics (messages processed, storage used, query patterns)

**Tech Stack**:
- Kubernetes (EKS / GKE)
- Neo4j AuraDB (managed graph database)
- CloudFlare (CDN, DDoS protection)
- Stripe (billing)

#### Enterprise Authentication

**Features**:
- SSO (SAML, OAuth2, OIDC)
- RBAC (role-based access control)
  - Roles: Admin, Editor, Viewer
  - Permissions: ingest messages, query graphs, export data, manage users
- Audit trails (who accessed what, when)
  - Queryable via API: `GET /audit-logs?user=alice&action=export`
- API key management (create, rotate, revoke keys per user/service)

#### Vertical-Specific Schemas

**Goal**: Pre-built entity types & extraction templates for industries.

**Examples**:

1. **Healthcare** (HIPAA-compliant deployment)
   - Entity types: Patient, Diagnosis, Medication, Treatment, Provider
   - Relationships: prescribed, diagnosed_with, treated_by
   - Compliance: PHI handling, audit logs, encryption at rest

2. **Coaching** (proven use case)
   - Entity types: Client, Goal, Progress, Exercise, Metric
   - Relationships: has_goal, achieved, struggled_with
   - Analytics: Goal completion rates, topic trends

3. **Sales/CRM**
   - Entity types: Lead, Account, Deal, Contact, Product
   - Relationships: interested_in, owns, interacted_with
   - Integrations: Salesforce sync, HubSpot webhooks

4. **Customer Support**
   - Entity types: User, Issue, Feature, Bug, Request
   - Relationships: reported, affected_by, wants
   - Analytics: Recurring issues, user pain points

**Schema Builder** (visual editor):
- Define custom entity types
- Configure decay rates per type
- Template extraction prompts
- No-code for non-technical users

#### Advanced Analytics

**Features**:

1. **Relationship Strength Trending**
   - Track confidence over time (with decay factored in)
   - Alert when relationships weaken (e.g., customer interest declining)

2. **Entity Importance Scoring** (PageRank-style)
   - Identify central entities in graph (most connected, highest confidence)
   - Use for prioritization (which topics/people matter most?)

3. **Conversation Topic Clustering**
   - Group related entities via embeddings
   - Detect emerging topics (new clusters forming)

4. **Anomaly Detection**
   - Flag unusual relationship changes (sudden preference shift = churn risk?)
   - Detect contradictions (conflicting preferences = confused user?)

5. **Dashboards**
   - Pre-built visualizations (entity growth, relationship density, confidence distribution)
   - Custom dashboard builder (drag-drop widgets)

#### Integrations

**Communication Platforms**:
- Slack (bot ingests channel conversations)
- Microsoft Teams (same)
- Discord (community server memory)
- Telegram (chatbot context)

**CRMs**:
- Salesforce (sync contacts, deals → entities)
- HubSpot (webhook on new interactions)
- Pipedrive, Close.io (via Zapier)

**LLM Platforms**:
- LangChain (EngRAM retriever plugin)
- LlamaIndex (data connector)
- Haystack (document store adapter)

**Data Sources**:
- CSV/JSON import
- API webhooks (ingest from custom apps)
- Email parsing (conversations from Gmail, Outlook)

#### Premium Support

**Tiers**:
- **Community** (Free): GitHub issues, community Discord
- **Pro**: Email support (24-48h response)
- **Team**: Slack Connect, 12h response
- **Enterprise**: Dedicated Slack channel, 4h SLA, video calls

**Services**:
- Schema consulting (help design entity types for your domain)
- Custom extraction pipelines (fine-tune prompts for your data)
- On-prem deployment assistance (Kubernetes setup, monitoring)
- Training sessions (for your team to use the platform)

---

### 4.4 Revenue Model

**Year 1 Goals**:
- 100 free users (GitHub stars, community traction)
- 20 paying Pro customers ($49/mo × 20 = $980/mo)
- 3 Team customers ($199/mo × 3 = $597/mo)
- Total MRR: ~$1,600 (seed funding runway)

**Year 2 Goals**:
- 1,000 free users
- 100 Pro ($4,900/mo)
- 20 Team ($3,980/mo)
- 2 Enterprise ($2,000/mo each = $4,000/mo)
- Total MRR: ~$13,000 (sustainable, small team)

**Why This Split Works**:
- **OSS builds trust**: Developers vet the tech before buying
- **Self-hosting option**: Privacy-sensitive users can run their own
- **Hosting convenience**: Most users prefer "just works" SaaS
- **Enterprise upsell**: Compliance + integrations + support = high willingness to pay

---

## 5. Tech Stack

### 5.1 Database: Neo4j Community Edition

**Choice**: Neo4j Community Edition (with Aion extension support)

**Reasoning**:

| Criterion | Neo4j | PostgreSQL + AGE | ArangoDB |
|-----------|-------|------------------|----------|
| **Native graph performance** | ✅ Best (10-100x faster traversals) | ⚠️ Acceptable (SQL joins slow) | ✅ Good |
| **Temporal features** | ✅ Transaction-time functions, Aion extension (10x speedup) | ❌ Manual bitemporal modeling | ⚠️ Time-travel docs sparse |
| **Query language** | ✅ Cypher (well-documented, temporal patterns available) | ⚠️ SQL + Cypher hybrid (confusing) | ⚠️ AQL (less familiar) |
| **Ecosystem** | ✅ Largest (APOC, GDS, Bloom) | ⚠️ PostgreSQL huge, but AGE immature | ⚠️ Smaller community |
| **Vector support** | ✅ Native since 5.11 | ✅ pgvector | ✅ Native |
| **Conversation-scale** | ✅ 100M+ relationships on single instance | ✅ Good | ✅ Good |
| **Open-source license** | ✅ GPLv3 (Community Edition) | ✅ PostgreSQL license | ✅ Apache 2.0 |
| **Managed hosting** | ✅ AuraDB (easy commercial upsell) | ✅ RDS, Supabase | ⚠️ ArangoGraph (less known) |

**Winner**: Neo4j
- Best temporal graph performance (Aion research: 10x speedup)
- Most production temporal graph examples (Graphiti, OpenMemory use Neo4j)
- Cypher easier than SQL for graph queries
- Clear commercial path (AuraDB for hosted layer)

**Aion Extension** (optional, performance boost):
- Open-source temporal extension from academic research (EDBT 2024)
- Hybrid storage: TimeStore (indexed by time) + LineageStore (indexed by entity)
- Use for commercial layer when query volume high
- Not required for MVP (vanilla Neo4j temporal queries sufficient)

**Installation**:
```bash
# Docker Compose (MVP)
docker-compose up  # Includes Neo4j Community 5.x

# Production
helm install neo4j neo4j/neo4j-cluster
```

---

### 5.2 Backend: Python + FastAPI

**Choice**: Python 3.11+, FastAPI, Pydantic v2

**Reasoning**:
- **Python**: Industry standard for ML/AI (best LLM library support)
- **FastAPI**: Modern, async, auto-generated OpenAPI docs, fast
- **Pydantic**: Type safety for data models (entities, relationships)

**Dependencies**:
```toml
[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.109.0"
uvicorn = "^0.27.0"       # ASGI server
neo4j = "^5.16.0"         # Neo4j driver
redis = "^5.0.0"          # Idempotency & caching (REQUIRED)
pydantic = "^2.5.0"       # Data validation
litellm = "^1.20.0"       # Multi-LLM provider
openai = "^1.10.0"        # Embeddings
numpy = "^1.26.0"         # Vector math
python-dotenv = "^1.0.0"  # Config management
```

**API Structure**:
```
src/
├── main.py                 # FastAPI app, route definitions
├── models/                 # Pydantic models
│   ├── entity.py           # Entity, EntityType
│   ├── relationship.py     # Relationship, RelationshipType
│   └── message.py          # Message, ConversationContext
├── services/               # Business logic
│   ├── extraction.py       # LLM extraction pipeline
│   ├── graph.py            # Graph operations (CRUD)
│   ├── temporal.py         # Decay, point-in-time queries
│   └── resolution.py       # Conflict resolution
├── storage/                # Database adapters
│   ├── base.py             # Abstract interface
│   ├── neo4j.py            # Neo4j implementation
│   └── memory.py           # In-memory (for testing)
├── llm/                    # LLM integration
│   ├── providers.py        # LiteLLM wrapper
│   ├── prompts.py          # Extraction prompt templates
│   └── embeddings.py       # Embedding generation
└── config.py               # Settings (from env vars)
```

---

### 5.3 Redis: Idempotency & Caching

**Choice**: Redis 7.x (REQUIRED for production, optional for development)

**Role in Architecture**:
- **Message deduplication**: Atomic idempotency check via SET NX
- **Processed message tracking**: Prevents duplicate ingestion on replay
- **Future**: Entity lookup cache, rate limiting, session management

**Why Redis (not database-based deduplication)?**
- **Atomic operations**: `SET NX` (set if not exists) is atomic, prevents race conditions
- **TTL support**: Automatic cleanup (24h expiry prevents unbounded growth)
- **Performance**: < 1ms latency for idempotency checks
- **Separate concern**: Graph database should not track ephemeral ingestion state

**Idempotency Pattern** (from § 2.2):

```python
import redis

# Connection
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Atomic idempotency check
def process_message(message: Message):
    # Attempt to set key with 24h TTL
    is_new = r.set(f"processed:{message.id}", "1", nx=True, ex=86400)
    
    if not is_new:
        logger.info(f"Skipping duplicate message: {message.id}")
        return  # Already processed
    
    # Process message (extraction, graph updates)
    # ...
```

**Key Design Decisions**:

| Decision | Rationale |
|----------|-----------|
| **24h TTL** | Messages older than 24h unlikely to be replayed; prevents unbounded memory |
| **SET NX (not SETNX)** | Modern Redis command with combined set+expire (atomic) |
| **Simple string value** | Don't need to store result; existence check is sufficient |
| **Key prefix `processed:`** | Namespace separation for future cache keys |

**Configuration per Environment**:

```python
# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Redis (required for production)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str | None = None
    redis_db: int = 0
    redis_tls: bool = False  # Enable for production (AWS ElastiCache, etc.)
    
    # Development: Allow disabling Redis (uses in-memory set)
    redis_enabled: bool = True
    
    class Config:
        env_file = ".env"
```

**Deployment**:

**Development** (Docker Compose):
```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes  # Persistence

  api:
    # ...
    environment:
      REDIS_HOST: redis
      REDIS_ENABLED: "true"
    depends_on:
      - redis
      - neo4j
```

**Production**:
- **Managed Redis**: AWS ElastiCache, Google Memorystore, Azure Cache
- **Self-hosted**: Redis Sentinel (HA) or Redis Cluster (sharding)
- **Configuration**: Enable TLS, require password, use dedicated instance

**Fallback for Development** (if Redis unavailable):

```python
# Simple in-memory fallback (NOT for production)
class InMemoryDedup:
    def __init__(self):
        self._processed = set()
        self._max_size = 10000
    
    def check_and_set(self, message_id: str) -> bool:
        if message_id in self._processed:
            return False
        
        if len(self._processed) >= self._max_size:
            self._processed.clear()  # Naive eviction
        
        self._processed.add(message_id)
        return True

# Use Redis if available, else fallback
if settings.redis_enabled:
    dedup = RedisDedup(host=settings.redis_host, port=settings.redis_port)
else:
    logger.warning("Redis disabled - using in-memory dedup (NOT production-safe)")
    dedup = InMemoryDedup()
```

**Why Not PostgreSQL for Idempotency?**

Could use `ON CONFLICT DO NOTHING` pattern:
```sql
INSERT INTO processed_messages (message_id, processed_at)
VALUES ($1, NOW())
ON CONFLICT (message_id) DO NOTHING
RETURNING message_id;
```

**Redis wins because**:
- Faster (< 1ms vs 5-10ms for DB round trip)
- Auto-cleanup via TTL (no manual purge job)
- Doesn't bloat application database
- Standard pattern for idempotent message processing

**Redis is MANDATORY for production** to ensure replay safety and prevent duplicate graph updates.

---

### 5.4 LLM Integration: LiteLLM

**Choice**: LiteLLM (multi-provider abstraction)

**Reasoning**:
- Unified API for OpenAI, Anthropic, Cohere, Ollama, local models
- Users bring their own API keys (core)
- Commercial layer can manage API keys
- Easy to swap providers (cost optimization, fallback)

**Configuration**:
```python
from litellm import completion

# User provides API key via env var or config
response = completion(
    model="gpt-4o-mini",  # or "claude-3-haiku", "ollama/llama3"
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1,
    api_key=user_api_key
)
```

**Supported Models** (MVP):
- **OpenAI**: gpt-4o-mini (fast, cheap extraction)
- **Anthropic**: claude-3-haiku (fast, cheap alternative)
- **Ollama**: llama3, mistral (local, privacy-focused)

**Cost Optimization**:
- Use mini/haiku for extraction (< $0.0001/message)
- Batch calls where possible
- Cache entity embeddings

---

### 5.4 Embedding: text-embedding-3-small

**Choice**: OpenAI text-embedding-3-small (or nomic-embed-text for local)

**Reasoning**:
- **1536 dimensions**: Industry standard, good balance
- **Fast**: < 10ms per embedding
- **Cheap**: $0.00002 per 1K tokens
- **Local alternative**: nomic-embed-text (Apache 2.0, 768d)

**Usage**:
```python
from openai import OpenAI

client = OpenAI(api_key=api_key)

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding  # 1536 floats
```

**Entity Resolution**:
```python
def find_similar_entity(canonical: str, entity_type: str) -> Entity | None:
    query_embedding = embed(canonical)
    
    # Neo4j vector similarity search
    results = graph.query("""
        CALL db.index.vector.queryNodes('entity_embedding', 5, $embedding)
        YIELD node, score
        WHERE node.type = $type AND score > 0.85
        RETURN node, score
        ORDER BY score DESC
        LIMIT 1
    """, embedding=query_embedding, type=entity_type)
    
    if results:
        return results[0].node
    return None
```

---

### 5.5 Vector Storage: Neo4j Vector Index

**Choice**: Neo4j native vector index (not separate vector DB)

**Reasoning**:
- Simpler architecture (one database, not two)
- Neo4j supports vector similarity since 5.11
- Sufficient for conversation-scale (10K-100K entities)
- Co-locate graph + vector data (unified queries)

**Why not Pinecone/Qdrant/Weaviate?**
- Over-engineering for conversation-scale
- Adds operational complexity (two databases to manage)
- Neo4j vector performance adequate (< 50ms for k=10 search on 100K vectors)

**Setup**:
```cypher
CREATE VECTOR INDEX entity_embedding
FOR (e:Entity)
ON e.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
};
```

---

### 5.6 API Layer: FastAPI + Uvicorn

**Why FastAPI?**
- Async support (handle concurrent requests efficiently)
- Auto-generated OpenAPI docs (interactive API explorer)
- Pydantic integration (type-safe request/response)
- WebSocket support (future: real-time graph updates)

**Server Stack**:
```
Client (curl, SDK, web UI)
         ↓
    Nginx (reverse proxy, SSL)
         ↓
    Uvicorn (ASGI server)
         ↓
    FastAPI (routing, validation)
         ↓
   ┌──────────┴──────────┐
   ↓                     ↓
Neo4j (graph + vectors)  LiteLLM (LLM calls)
```

**Deployment**:
- **Development**: `uvicorn main:app --reload`
- **Production**: `gunicorn -k uvicorn.workers.UvicornWorker -w 4 main:app`
- **Docker**: Multi-stage build (small final image)

---

### 5.7 Deployment: Docker Compose (MVP) → Kubernetes (Commercial)

#### MVP: Docker Compose

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.16-community
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc"]'
    ports:
      - "7474:7474"  # Browser
      - "7687:7687"  # Bolt
    volumes:
      - neo4j_data:/data

  api:
    build: .
    environment:
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: password
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    ports:
      - "8000:8000"
    depends_on:
      - neo4j

volumes:
  neo4j_data:
```

**Quickstart**:
```bash
git clone https://github.com/yri-ai/engram
cd engram
cp .env.example .env
# Add your OPENAI_API_KEY to .env
docker-compose up
# Visit http://localhost:8000/docs
```

#### Commercial: Kubernetes + AuraDB

**Stack**:
- **Compute**: EKS (AWS) or GKE (Google Cloud)
- **Database**: Neo4j AuraDB (managed, auto-scaling)
- **CDN**: CloudFlare (global edge, DDoS protection)
- **Monitoring**: Prometheus + Grafana
- **Logging**: Loki or CloudWatch

**Scaling**:
- API pods: Auto-scale based on CPU/memory (2-20 pods)
- Neo4j: AuraDB auto-scales storage, manual node scaling
- Target: 1000 req/sec (conversation-scale, not web-scale)

---

## 6. First Milestone (MVP)

### 6.1 Goal

**Ship v0.1.0 in 4-6 weeks** with:
- Working end-to-end demo (coaching scenario)
- Developers can run locally in < 5 minutes
- **300+ GitHub stars in first month**

### 6.2 Scope

#### What Ships in v0.1.0

**Core Engine**:
- ✅ Neo4j schema (entities, relationships, bitemporal fields)
- ✅ Entity extraction (LLM prompt, OpenAI/Anthropic/Ollama)
- ✅ Relationship inference (LLM prompt with context)
- ✅ Conflict resolution (temporal versioning, terminate old → create new)
- ✅ Decay function (exponential, configurable rates)
- ✅ Point-in-time queries (`WHERE valid_from <= $date AND valid_to >= $date`)
- ✅ Relationship evolution tracking (get all versions)

**API** (8 endpoints):
```
POST   /messages                    # Ingest conversation message
GET    /entities                    # List all entities (paginated)
GET    /entities/{id}               # Entity details + current relationships
GET    /entities/{id}/relationships # Current relationships (with decay)
GET    /query/point-in-time         # Historical query
GET    /query/evolution             # Relationship timeline
GET    /search                      # Search entities (text + vector)
GET    /health                      # Health check
```

**CLI**:
```bash
engram init                         # Create Neo4j schema
engram ingest conversation.json     # Batch import
engram query "Kendra's preferences" # Natural language query (calls API)
engram serve                        # Start API server
engram export graph.json            # Export to JSON
```

**Web UI** (basic):
- Graph visualization (D3.js or react-force-graph)
- Entity list + search
- Entity detail page (show relationships, timeline)
- Conversation log (messages → extracted entities)
- No authentication (localhost only)

**Documentation**:
- README with quickstart (copy-paste Docker Compose commands)
- ARCHITECTURE.md (this document)
- API reference (auto-generated from FastAPI)
- Example: Coaching scenario
  ```
  Week 1: "Kendra loves Nike running shoes"
  Week 3: "Kendra switched to Adidas"
  Query: "Kendra's shoe preference on Week 2" → Nike
  Query: "Kendra's current preference" → Adidas
  ```

**Tests**:
- Unit tests (extraction, conflict resolution logic)
- Integration tests (API endpoints)
- E2E test (coaching scenario, assertions on query results)

#### What Does NOT Ship in v0.1.0

- ❌ Multi-tenancy (single graph only)
- ❌ Authentication (localhost deployment assumed)
- ❌ Advanced analytics (PageRank, clustering)
- ❌ Vertical schemas (generic entities only)
- ❌ Integrations (Slack, CRM)
- ❌ Web UI polish (basic is fine)
- ❌ Hosted version (self-host only)

**Rationale**: Nail the core experience first. Advanced features can wait.

---

### 6.3 Success Metrics

**Technical**:
- ✅ Quickstart works in < 5 minutes (timed test)
- ✅ Coaching scenario demo runs without errors
- ✅ API latency < 200ms (p95) for message ingestion
- ✅ Tests pass (>80% coverage)

**Traction**:
- **300+ GitHub stars** in first month
  - Channels: Product Hunt, Hacker News, LangChain Discord, Reddit (r/MachineLearning, r/LangChain)
- **3-5 production users** (early adopters from coaching, CRM, personal knowledge graph use cases)
- **10+ community contributions** (issues, PRs, discussions)

**Engagement**:
- **20+ Discord members** (community for support, feature requests)
- **5+ blog posts** from users (use case stories)

---

### 6.4 Launch Strategy

#### Pre-Launch (Weeks 1-4: Building)

1. **Week 1**: Core graph engine + extraction pipeline
2. **Week 2**: API endpoints + CLI tools
3. **Week 3**: Web UI (basic) + coaching demo
4. **Week 4**: Documentation + polish

#### Launch Week (Week 5)

**Monday**: 
- Publish to GitHub (MIT license)
- README with demo GIF (coaching scenario in action)
- Submit to Product Hunt (Tuesday launch)

**Tuesday**:
- Product Hunt launch (goal: top 5 of the day)
- Post to Hacker News (Show HN: Engram - OSS temporal knowledge graph for AI)
- Post to Reddit (r/MachineLearning, r/LangChain, r/SideProject)

**Wednesday-Friday**:
- Respond to feedback (GitHub issues, comments)
- Share user stories on Twitter/LinkedIn
- Reach out to LangChain maintainers (potential integration)

#### Post-Launch (Week 6+)

- Weekly feature releases (v0.2, v0.3)
- Community engagement (Discord, office hours)
- Content marketing (blog posts, tutorials, YouTube demos)
- Partnership outreach (coaching platforms, CRM vendors)

---

### 6.5 Differentiation & Messaging

**Tagline**: *"The only OSS temporal knowledge graph built for conversations, not documents."*

**Positioning**:

| Competitor | Limitation | Engram Advantage |
|------------|------------|------------------|
| **Mem0** | Graph is optional; no relationship versioning | Relationships are first-class, fully versioned |
| **GraphRAG** | Batch-only, document-focused | Real-time, conversation-native |
| **LightRAG** | Generic graph, no temporal semantics | Bitemporal model, point-in-time queries |
| **Zep/Graphiti** | Went closed-source | MIT forever, self-hostable |

**Pitch** (30 seconds):

> "Engram is a temporal knowledge graph for AI memory. It automatically extracts entities and relationships from conversations, tracks how they evolve over time, and lets you query 'What did we know about X as of date Y?'
> 
> Unlike Mem0 or GraphRAG, Engram versions relationships — so you can see that Kendra preferred Nike in January but switched to Adidas in March. It's the pattern I built in production for a coaching platform, now open-source."

**Demo** (2 minutes):

```bash
# Start Engram
docker-compose up

# Ingest conversations
engram ingest coaching-log.json

# Query current state
curl "http://localhost:8000/entities/Kendra/relationships?type=prefers"
# → Returns: Adidas (confidence: 0.85)

# Query historical state
curl "http://localhost:8000/query/point-in-time?entity=Kendra&as_of=2024-02-01"
# → Returns: Nike (confidence: 0.95)

# See evolution
curl "http://localhost:8000/query/evolution?entity=Kendra&type=prefers"
# → Timeline: Nike (Jan 15 - Mar 20) → Adidas (Mar 20 - now)
```

**Show in web UI**: Graph visualization with time slider (scrub through relationship changes)

---

## 7. Pushback on Over-Engineering

### What You DON'T Need (at MVP scale)

#### 1. Microservices
**Why not**: Conversation-scale (10K-100K entities) fits comfortably in a monolith.
**Use instead**: FastAPI monolith (extraction + API + graph in one process).
**When to split**: At 10K+ req/sec or 1M+ entities (you're not there yet).

#### 2. Kafka / Streaming
**Why not**: Direct API ingestion is fast enough (50-200ms).
**Use instead**: Synchronous POST /messages endpoint.
**When to add**: When you need exactly-once delivery or event replay (enterprise customers).

#### 3. Distributed Graph Database
**Why not**: Neo4j single instance handles 100M relationships.
**Use instead**: Neo4j Community Edition (or AuraDB for commercial).
**When to shard**: At billions of relationships (conversation-scale rarely reaches this).

#### 4. Custom Graph Storage
**Why not**: Neo4j exists, is battle-tested, and has temporal extensions (Aion).
**Use instead**: Neo4j Community Edition.
**Never**: Don't build your own graph database. This is a PhD-level project.

#### 5. Complex ML Pipelines
**Why not**: LLM extraction + simple decay function beats fancy GNNs for this use case.
**Use instead**: GPT-4o-mini for extraction, exponential decay for confidence.
**When to add ML**: When you have 10K+ users and specific optimization needs (then fine-tune embedding model or train relationship classifier).

#### 6. Real-Time Graph Algorithms
**Why not**: PageRank, community detection, centrality are nice-to-haves, not core to MVP.
**Use instead**: Simple Cypher queries (traversals, pattern matching).
**When to add**: v0.3+ (advanced analytics for commercial layer).

#### 7. Multiple Storage Backends
**Why not**: Maintaining 3+ database adapters (Neo4j, AGE, ArangoDB) is expensive.
**Use instead**: Neo4j (primary), In-Memory (testing only).
**When to add**: When community contributes AGE adapter (don't build it yourself).

---

### Keep It Simple (YAGNI Principle)

**MVP Architecture**:
```
┌─────────────────────────────────┐
│   FastAPI (single process)      │
│  ┌──────────────────────────┐   │
│  │ Extraction Pipeline      │   │
│  │  (LLM calls)             │   │
│  └──────────────────────────┘   │
│  ┌──────────────────────────┐   │
│  │ Graph Engine             │   │
│  │  (Cypher queries)        │   │
│  └──────────────────────────┘   │
│  ┌──────────────────────────┐   │
│  │ API Routes               │   │
│  │  (REST endpoints)        │   │
│  └──────────────────────────┘   │
└─────────────┬───────────────────┘
              │
              ↓
     ┌────────────────┐
     │  Neo4j         │
     │  (Community)   │
     └────────────────┘
```

**One process. One database. Scales to 1000s of users.**

When you have 10K users, *then* optimize. Not before.

---

## 8. Open Questions & Future Work

### 8.1 Open Questions

1. **Entity Merging**: When to auto-merge similar entities vs create duplicates?
   - Current: Threshold = 0.85 embedding similarity
   - Issue: False positives ("Mike Johnson" vs "Michael Johnson" = different people?)
   - Solution: Manual review UI for merges? Active learning?

2. **Relationship Type Explosion**: How to prevent 100+ custom relationship types?
   - Current: Allow custom types
   - Issue: Graph becomes unqueryable if types too specific
   - Solution: Recommend generic types + metadata? Type hierarchy?

3. **Cross-Conversation Entity Resolution**: How to link entities across conversations?
   - Current: Embedding similarity within single conversation context
   - Issue: "Kendra" in conversation A != "Kendra" in conversation B (different people?)
   - Solution: User-level entity namespacing? Explicit entity linking?

4. **Confidence Calibration**: Are LLM confidence scores meaningful?
   - Current: Assume LLM confidence ≈ relationship confidence
   - Issue: LLMs are poorly calibrated (overconfident)
   - Solution: Train calibration model? Use logprobs? Fixed confidence levels?

5. **Decay Rate Tuning**: Should decay rates be learned from data?
   - Current: Hardcoded per entity/relationship type
   - Issue: Different domains have different decay rates
   - Solution: Allow user configuration? Learn from reinforcement patterns?

### 8.2 Future Work (Post-MVP)

#### v0.2 (Week 8)
- Multi-tenancy (isolated graphs per user)
- Basic authentication (API keys)
- Relationship clustering (group related entities via embeddings)

#### v0.3 (Week 12)
- Advanced analytics (PageRank, betweenness centrality)
- Vertical schema templates (coaching, sales, support)
- Slack integration (ingest channel conversations)

#### v1.0 (Month 6)
- Hosted beta (commercial layer)
- SSO (SAML, OAuth)
- Audit trails
- LangChain plugin

#### v2.0 (Year 1)
- Entity merging UI (manual review + active learning)
- Custom decay rate learning (from user feedback)
- Graph diffing (compare graph states between dates)
- Time-travel visualization (animated graph evolution)

---

## 9. Conclusion

Engram solves a real problem (conversation memory for AI systems) with a novel approach (bitemporal relationship versioning). The architecture is proven (your coaching platform) and grounded in research (Graphiti/Zep patterns, Aion temporal graphs).

**What makes it win**:
1. **OSS gap**: Zep went closed-source, leaving a vacuum
2. **Technical moat**: Temporal relationship versioning (not just document chunks)
3. **Proven in production**: Your coaching platform validates the approach
4. **Clear business model**: MIT core + commercial hosting (like Plausible, PostHog)
5. **Developer-first**: Works in 5 minutes (Docker Compose), no magic
6. **Timing**: AI agents need memory; vector DBs aren't enough

**Next steps**:
1. Scaffold project structure (`src/`, `tests/`, `docker-compose.yml`)
2. Implement core graph schema (Neo4j Cypher DDL)
3. Build extraction pipeline (Stage 1-3)
4. Create CLI + API
5. Document + demo
6. Launch 🚀

**This ships in 4-6 weeks and becomes the default OSS temporal knowledge graph for AI memory.**
