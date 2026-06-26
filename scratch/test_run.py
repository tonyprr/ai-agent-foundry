import asyncio
import os
os.environ["MOCK_MODE"] = "True"
os.environ["AZURE_SEARCH_VECTOR_FIELD"] = ""
os.environ["AZURE_SEARCH_SEMANTIC_CONFIG"] = ""

from app.config import Settings
from app.adapters.driven.storage.in_memory_store_adapter import InMemorySessionStoreAdapter
from app.adapters.driven.search.search_adapter import SearchAdapter
from app.adapters.driven.agent.agent_adapter import AgentAdapter

async def main():
    settings = Settings(mock_mode=True)
    session_store = InMemorySessionStoreAdapter()
    search_adapter = SearchAdapter(
        default_endpoint="mock-endpoint",
        default_index="mock-index",
        default_key="mock-key"
    )
    agent_adapter = AgentAdapter(settings, search_adapter, session_store)
    
    session = await session_store.get_or_create_session("test_session_id")
    workflow = agent_adapter._get_or_create_workflow()
    
    print("Running workflow...")
    run_result = await workflow.run(
        message="Develop a Solidity contract using OpenZeppelin",
        checkpoint_storage=agent_adapter._checkpoint_storage
    )
    
    print("\n--- Event List ---")
    for idx, event in enumerate(run_result):
        if event.type == "request_info":
            print(f"Event {idx}: request_info")
            print(f"  Data: {event.data}")
            if hasattr(event.data, "agent_response"):
                for m in event.data.agent_response.messages:
                    print(f"    Message: role={m.role}, author={m.author_name}")
                    for c in m.contents:
                        print(f"      Content: type={c.type}")
                        if c.type == "text":
                            print(f"        text={c.text}")
                        elif c.type == "function_call":
                            print(f"        function_call: name={c.name}, args={c.arguments}")

if __name__ == "__main__":
    asyncio.run(main())
