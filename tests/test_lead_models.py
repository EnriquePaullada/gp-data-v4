import datetime as dt
from src.models.lead import Lead, SalesStage
from src.models.message import Message, MessageRole
from src.models.intelligence import IntelligenceSignal, BANTDimension, ConfidenceScore

def test_production_readiness():
    print("\n🚀 Pre-Flight Check...\n")

    # 1. Initialize Lead
    lead = Lead(
        lead_id="+5215538899800",
        full_name="Enrique Paullada"
    )

    # 2. Test Sliding Window Logic (The "Memory" Test)
    # We add 25 messages. The model should only keep the LATEST 20.
    print("Testing Sliding Window Context...")
    for i in range(25):
        msg = Message(
            lead_id=lead.lead_id,
            role=MessageRole.LEAD if i % 2 == 0 else MessageRole.ASSISTANT,
            content=f"Message number {i}",
            timestamp=dt.datetime.now(dt.UTC)
        )
        lead.add_message(msg)
    
    assert len(lead.recent_history) == 20
    assert lead.message_count == 25
    print(f"✅ Sliding Window: Passed (History size: {len(lead.recent_history)})")

    # 3. Test Deterministic BANT Reduction (The "Truth" Test)
    print("Testing BANT Intelligence Reduction...")
    
    # Signal A: Early in the conversation
    sig_low = IntelligenceSignal(
        dimension=BANTDimension.BUDGET,
        extracted_value="low",
        confidence=ConfidenceScore(value=0.9, reasoning="User mentioned limited startup funds"),
        raw_evidence="We have a very tight budget."
    )
    
    # Signal B: Later in the conversation (The Pivot)
    sig_high = IntelligenceSignal(
        dimension=BANTDimension.BUDGET,
        extracted_value="high",
        confidence=ConfidenceScore(value=0.7, reasoning="User mentioned Series B funding arrived"),
        raw_evidence="Actually, we just closed our round, budget is no longer an issue."
    )

    lead.add_signal(sig_low)
    lead.add_signal(sig_high)

    # The bant_summary should show 'high' because it was the LATEST signal added
    summary = lead.bant_summary
    assert summary[BANTDimension.BUDGET] == "high"
    print(f"✅ BANT Reduction: Passed (Current Budget: {summary[BANTDimension.BUDGET]})")

    # 4. Test Heartbeat Synchronization
    print("Testing Heartbeat Synchronization...")
    last_msg_time = lead.recent_history[-1].timestamp
    assert lead.last_interaction_at == last_msg_time, f"Heartbeat mismatch! Lead: {lead.last_interaction_at} vs Msg: {last_msg_time}"
    print(f"✅ Heartbeat Sync: Passed ({lead.last_interaction_at})")

    print("\n🛡️  ALL SYSTEMS GREEN.")

if __name__ == "__main__":
    try:
        test_production_readiness()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        # In a production environment, we'd exit with a non-zero code to stop the CI/CD
        exit(1)