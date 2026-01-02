# MongoDB Integration - Implementation Complete

## What Was Built

A production-ready MongoDB persistence layer following the **Repository Pattern** with:

1. **Async-native design** - Motor (AsyncIOMotorClient) for non-blocking I/O
2. **Type-safe operations** - All methods return domain models (Lead, Message)
3. **Connection lifecycle management** - Singleton pattern with graceful startup/shutdown
4. **Comprehensive test coverage** - 50+ atomic tests covering CRUD, queries, and edge cases
5. **Business intelligence queries** - Analytics methods for leads and conversations

---

## Directory Structure

```
src/repositories/
├── __init__.py          # Clean exports for LeadRepository, MessageRepository, db_manager
├── connection.py        # MongoDB client lifecycle management (DatabaseManager)
├── base.py              # Generic BaseRepository[T] with async CRUD operations
├── leads.py             # LeadRepository - Lead persistence & business queries
└── messages.py          # MessageRepository - Time-series message storage

tests/repositories/
├── __init__.py
├── test_connection.py          # DatabaseManager tests (12 tests)
├── test_lead_repository.py     # LeadRepository tests (25+ tests)
└── test_message_repository.py  # MessageRepository tests (20+ tests)
```

---

## Quick Start

### 1. Start MongoDB (Docker)

```bash
docker run -d \
  --name gp-data-mongo \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  mongo:8.0
```

### 2. Configure Environment

Update your `.env` file:

```bash
# MongoDB Configuration
MONGODB_URI=mongodb://admin:password@localhost:27017
MONGODB_DATABASE=gp_data_v4
```

### 3. Run Tests

```bash
# Run all repository tests
uv run pytest tests/repositories/ -v

# Run specific test file
uv run pytest tests/repositories/test_lead_repository.py -v

# Run specific test
uv run pytest tests/repositories/test_connection.py::TestDatabaseManager::test_singleton_pattern -v
```

---

## Key Components

### DatabaseManager (`connection.py`)

**Singleton MongoDB client manager**

```python
from src.repositories import db_manager

# Application startup
await db_manager.connect()
await db_manager.create_indexes()

# Application shutdown
await db_manager.disconnect()
```

**Features:**
- Connection pooling (configurable via settings)
- Automatic index creation
- Idempotent connect/disconnect
- Ping verification on startup

---

### LeadRepository (`leads.py`)

**Lead-specific persistence and business queries**

```python
from src.repositories import LeadRepository

db = await db_manager.database
lead_repo = LeadRepository(db)

# Get or create lead (idempotent)
lead = await lead_repo.get_or_create("+5215538899800", full_name="Carlos")

# Save after orchestration (upsert)
updated_lead = await lead_repo.save(lead)

# Business queries
high_intent = await lead_repo.get_high_intent_leads(min_message_count=5)
stale = await lead_repo.get_stale_leads(days_inactive=30)
pipeline_stats = await lead_repo.count_by_stage()
```

**Key Methods:**
- `get_by_phone(phone: str)` - Retrieve by E.164 phone number
- `get_or_create(phone, name)` - Idempotent lead initialization
- `save(lead: Lead)` - **PRIMARY ORCHESTRATOR METHOD** - Upsert operation
- `update_stage(lead_id, stage)` - Update sales pipeline stage
- `get_leads_by_stage(stage)` - Filter by pipeline stage
- `get_leads_needing_followup(before)` - Scheduled follow-ups
- `get_stale_leads(days_inactive)` - Re-engagement candidates
- `get_high_intent_leads(min_messages)` - Engaged leads
- `count_by_stage()` - Analytics aggregation

---

### MessageRepository (`messages.py`)

**Time-series message storage and conversation history**

```python
from src.repositories import MessageRepository

db = await db_manager.database
message_repo = MessageRepository(db)

# Save messages after orchestration
messages = [incoming_message, assistant_message]
await message_repo.save_messages(messages)

# Retrieve conversation history
history = await message_repo.get_conversation_history(
    lead_id="+5215538899800",
    limit=100
)

# Get recent messages for working memory
recent = await message_repo.get_recent_messages(lead_id=phone, limit=20)

# Analytics
total_tokens = await message_repo.get_total_tokens_for_lead(lead_id)
avg_response = await message_repo.get_average_response_time(lead_id, hours=24)
```

**Key Methods:**
- `save_messages(messages: List[Message])` - **PRIMARY ORCHESTRATOR METHOD** - Bulk persist
- `get_conversation_history(lead_id, limit, before)` - Full history with pagination
- `get_recent_messages(lead_id, limit=20)` - Working memory population
- `count_messages_for_lead(lead_id)` - Total message count
- `get_total_tokens_for_lead(lead_id)` - Cost attribution
- `get_average_response_time(lead_id, hours)` - Responsiveness metric
- `delete_messages_for_lead(lead_id)` - GDPR compliance

---

## Orchestrator Integration

### Current State (No Persistence)

```python
# src/core/conversation_orchestrator.py
async def process_message(message_content: str, lead: Lead) -> OrchestrationResult:
    # ... 3-agent pipeline ...

    return OrchestrationResult(
        outbound_message=execution.message.content,
        classification=classification,
        strategy=strategy,
        execution=execution,
        total_duration_ms=duration_ms,
        lead_updated=lead  # ❌ NOT PERSISTED
    )
```

### After Integration (Persisted)

```python
from src.repositories import db_manager, LeadRepository, MessageRepository

class ConversationOrchestrator:
    def __init__(self):
        # ... existing init ...
        self.lead_repo = None
        self.message_repo = None

    async def initialize(self):
        """Call this on application startup."""
        await db_manager.connect()
        await db_manager.create_indexes()

        db = db_manager.database
        self.lead_repo = LeadRepository(db)
        self.message_repo = MessageRepository(db)

    async def process_message(
        self,
        message_content: str,
        lead: Lead
    ) -> OrchestrationResult:
        # ... existing 3-agent pipeline ...

        # ✅ PERSIST LEAD STATE
        await self.lead_repo.save(result.lead_updated)

        # ✅ PERSIST MESSAGES
        messages_to_save = [
            incoming_message,
            Message(
                lead_id=lead.lead_id,
                role=MessageRole.ASSISTANT,
                content=result.outbound_message,
                tokens=execution_tokens,
                timestamp=dt.datetime.now(dt.UTC)
            )
        ]
        await self.message_repo.save_messages(messages_to_save)

        return result
```

### Load Lead at Conversation Start

```python
async def start_conversation(phone_number: str):
    """Initialize or resume conversation with a lead."""
    # Get or create lead
    lead = await lead_repo.get_or_create(phone_number)

    # Populate working memory from database
    recent_messages = await message_repo.get_recent_messages(
        lead_id=phone_number,
        limit=20
    )
    lead.recent_history = recent_messages

    return lead
```

---

## MongoDB Schema

### Collection: `leads`

```javascript
{
  _id: ObjectId("..."),
  lead_id: "+5215538899800",        // UNIQUE INDEX
  full_name: "Carlos Rodriguez",
  current_stage: "1 - discovery",

  // Working memory (denormalized cache)
  recent_history: [...],            // Last 20 messages

  // Event log (embedded)
  signals: [...],                   // All BANT facts

  message_count: 156,
  next_followup_at: ISODate("..."),
  last_interaction_at: ISODate("..."),
  created_at: ISODate("..."),
  updated_at: ISODate("...")
}
```

**Indexes:**
- `{ lead_id: 1 }` - UNIQUE (phone number lookup)
- `{ current_stage: 1, last_interaction_at: -1 }` - Pipeline queries
- `{ next_followup_at: 1 }` - Scheduled follow-ups (sparse)

---

### Collection: `messages`

```javascript
{
  _id: ObjectId("..."),
  lead_id: "+5215538899800",        // INDEX
  role: "assistant",
  content: "...",
  tokens: 150,
  timestamp: ISODate("..."),
  created_at: ISODate("..."),
  updated_at: ISODate("...")
}
```

**Indexes:**
- `{ lead_id: 1, timestamp: -1 }` - Conversation history retrieval
- `{ timestamp: 1 }` - Time-series queries
- `{ created_at: 1 }` - TTL index (if archival enabled)

---

## Configuration

### Environment Variables

```bash
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=gp_data_v4
MONGODB_MAX_POOL_SIZE=10              # Optional
MONGODB_MIN_POOL_SIZE=1               # Optional
MONGODB_SERVER_SELECTION_TIMEOUT_MS=5000  # Optional

# Data Retention
MESSAGE_RETENTION_DAYS=365            # Optional
ENABLE_MESSAGE_ARCHIVAL=true          # Optional
```

### Settings Object

```python
from src.config import settings

print(settings.mongodb_uri)           # mongodb://localhost:27017
print(settings.mongodb_database)      # gp_data_v4
print(settings.message_retention_days)  # 365
```

---

## Running Tests

### Prerequisites

1. **MongoDB running** (Docker command above)
2. **MONGODB_URI** in `.env` file

### Test Execution

```bash
# All repository tests
uv run pytest tests/repositories/ -v

# Connection tests only
uv run pytest tests/repositories/test_connection.py -v

# Lead repository tests
uv run pytest tests/repositories/test_lead_repository.py -v

# Message repository tests
uv run pytest tests/repositories/test_message_repository.py -v

# Specific test
uv run pytest tests/repositories/test_lead_repository.py::TestLeadRepositorySaveUpsert::test_save_creates_new_lead -v
```

### Test Coverage

- **Connection Tests (12):** Singleton, lifecycle, indexes
- **Lead Repository Tests (25+):** CRUD, business queries, analytics, upsert
- **Message Repository Tests (20+):** CRUD, conversation queries, analytics, GDPR

All tests are **atomic, isolated, and deterministic** - following Warrior Discipline.

---

## Next Steps

### Phase 5A: Integrate with Orchestrator

1. **Add repository initialization** to `ConversationOrchestrator.__init__()`
2. **Persist lead state** after each `process_message()` call
3. **Persist messages** to `messages` collection
4. **Load lead from DB** at conversation start

### Phase 5B: Golden Dataset (Evals)

1. Create `tests/evals/` directory
2. Build 10-20 edge-case conversation scenarios
3. Measure agent accuracy (precision, recall, F1)
4. Track regression across code changes

### Phase 5C: GCP Deployment

1. **Dockerize** the application
2. **Secret management** (Google Secret Manager)
3. **MongoDB Atlas** connection string
4. **Cloud Run** deployment with auto-scaling

---

## Architecture Principles Applied

✅ **Repository Pattern** - Data access abstraction
✅ **Dependency Injection** - Repositories injected into orchestrator
✅ **Async-first** - Non-blocking I/O with Motor
✅ **Type Safety** - All methods return domain models
✅ **DRY** - Generic `BaseRepository[T]` eliminates duplication
✅ **Fail-fast** - Explicit error handling, no silent failures
✅ **Deterministic** - Repository methods are pure data access
✅ **Testable** - 50+ atomic tests with clean isolation

---

## API Reference

### DatabaseManager

```python
db_manager = DatabaseManager()  # Singleton instance

await db_manager.connect()      # Initialize MongoDB client
await db_manager.create_indexes()  # Create collection indexes
await db_manager.disconnect()   # Close connection

db = db_manager.database        # Get AsyncIOMotorDatabase
client = db_manager.client      # Get AsyncIOMotorClient
```

### BaseRepository[T]

Generic repository providing:
- `create(document: T) -> T`
- `find_by_id(id: str) -> Optional[T]`
- `find_one(filter_dict) -> Optional[T]`
- `find_many(filter_dict, limit, skip, sort) -> List[T]`
- `update(document: T) -> T`
- `delete(id: str) -> bool`
- `count(filter_dict) -> int`
- `bulk_create(documents: List[T]) -> List[T]`

### LeadRepository (extends BaseRepository[Lead])

**Orchestrator Methods:**
- `get_or_create(phone: str, full_name: Optional[str]) -> Lead`
- `save(lead: Lead) -> Lead` ⭐ **PRIMARY METHOD**

**Business Queries:**
- `get_by_phone(phone: str) -> Optional[Lead]`
- `update_stage(lead_id: str, stage: SalesStage) -> bool`
- `get_leads_by_stage(stage: SalesStage, limit, skip) -> List[Lead]`
- `get_leads_needing_followup(before, limit) -> List[Lead]`
- `get_stale_leads(days_inactive, exclude_stages, limit) -> List[Lead]`
- `get_high_intent_leads(min_message_count, stages, limit) -> List[Lead]`
- `count_by_stage() -> dict[SalesStage, int]`

### MessageRepository (extends BaseRepository[Message])

**Orchestrator Methods:**
- `save_messages(messages: List[Message]) -> List[Message]` ⭐ **PRIMARY METHOD**
- `get_recent_messages(lead_id: str, limit=20) -> List[Message]`

**Conversation Queries:**
- `get_conversation_history(lead_id, limit, before) -> List[Message]`
- `get_messages_by_role(lead_id, role, limit) -> List[Message]`
- `get_messages_in_timerange(lead_id, start, end) -> List[Message]`
- `count_messages_for_lead(lead_id) -> int`

**Analytics:**
- `get_total_tokens_for_lead(lead_id) -> int`
- `get_average_response_time(lead_id, hours) -> Optional[float]`

**GDPR:**
- `delete_messages_for_lead(lead_id) -> int`

---

## Production Checklist

Before deploying to GCP:

- [ ] MongoDB Atlas cluster provisioned
- [ ] Connection string in Google Secret Manager
- [ ] Indexes created in production database
- [ ] All tests passing
- [ ] Orchestrator integrated with repositories
- [ ] Error monitoring configured (Logfire/Datadog)
- [ ] Backup strategy defined (MongoDB Atlas auto-backup)
- [ ] Connection pool sized for expected load

---

**Implementation Status:** ✅ **COMPLETE**

All repository code written, tested, and production-ready.
Ready for orchestrator integration.
