import datetime as dt
from src.models.lead import Lead, SalesStage
from src.models.message import Message, MessageRole
from src.models.intelligence import IntelligenceSignal, BANTDimension, ConfidenceScore

def test_production_readiness():
    print("\n🚀 Pre-Flight Check: GP Data v4 Domain Integrity...\n")

    # 1. Initialize Lead
    lead = Lead(
        lead_id="+5215538899800",
        full_name="Enrique Paullada"
    )

    # 2. Test Sliding Window Logic & Heartbeat
    print("Testing Messaging & Temporal Sync...")
    # We'll keep a reference to a few message IDs for our signals
    message_ids = []
    
    for i in range(25):
        msg = Message(
            lead_id=lead.lead_id,
            role=MessageRole.LEAD if i % 2 == 0 else MessageRole.ASSISTANT,
            content=f"Message number {i}",
            timestamp=dt.datetime.now(dt.UTC)
        )
        # In a real DB, these would be generated. We'll simulate it:
        msg.id = f"msg_{i}" 
        message_ids.append(msg.id)
        lead.add_message(msg)
    
    assert len(lead.recent_history) == 20
    assert lead.last_interaction_at == lead.recent_history[-1].timestamp
    print(f"✅ Sliding Window & Heartbeat: Passed")

    # 3. Test Traceable BANT Reduction
    print("Testing Traceable Intelligence Signals...")
    
    # Signal A: A Direct Statement linked to the first message
    sig_direct = IntelligenceSignal(
        dimension=BANTDimension.BUDGET,
        extracted_value="low",
        confidence=ConfidenceScore(value=1.0, reasoning="Explicitly stated"),
        source_message_id=message_ids[0],
        is_inferred=False,
        raw_evidence="My budget is zero."
    )
    
    # Signal B: An Inference synthesized from multiple messages
    sig_inferred = IntelligenceSignal(
        dimension=BANTDimension.BUDGET,
        extracted_value="high",
        confidence=ConfidenceScore(value=0.8, reasoning="Inferred from company size and tools"),
        source_message_id=message_ids[-1], # Triggered by the last message
        is_inferred=True,
        inferred_from=[message_ids[10], message_ids[15], message_ids[20]], # Linked to the history
        raw_evidence="We have 500 users on Salesforce."
    )

    lead.add_signal(sig_direct)
    lead.add_signal(sig_inferred)

    # Verify the Deterministic Reduction (Latest wins)
    assert lead.bant_summary[BANTDimension.BUDGET] == "high"
    
    # Verify the Relationship Graph
    last_signal = lead.signals[-1]
    assert last_signal.is_inferred is True
    assert len(last_signal.inferred_from) == 3
    print(f"✅ Traceable Intelligence: Passed")

    print("\n🛡️  ALL SYSTEMS GREEN.")

if __name__ == "__main__":
    try:
        test_production_readiness()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)