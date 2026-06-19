import logging
from typing import Optional
from azure.core.exceptions import AzureError
from azure.search.documents.aio import SearchClient
from azure.identity.aio import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential

from agent_framework.azure import AzureAISearchContextProvider
from app.domain.models import SearchConfigOverride

logger = logging.getLogger(__name__)

class SearchAdapter:
    """
    Driven Adapter for Azure AI Search. 
    Handles validation of the connection and factories for MAF context providers.
    """
    def __init__(self, default_endpoint: Optional[str], default_index: Optional[str], default_key: Optional[str]):
        self._default_endpoint = default_endpoint
        self._default_index = default_index
        self._default_key = default_key

    def build_context_provider(self, config: SearchConfigOverride) -> AzureAISearchContextProvider:
        """
        Dynamically builds the AzureAISearchContextProvider from Microsoft Agent Framework
        using default settings merged with per-request overrides.
        """
        endpoint = config.endpoint or self._default_endpoint
        index_name = config.index_name or self._default_index
        api_key = config.api_key or self._default_key
        
        if not endpoint or not index_name:
            raise ValueError("Azure AI Search Endpoint and Index Name must be configured.")

        # Determine credential type (Key vs Managed Identity)
        credential = None
        if not api_key:
            logger.info("No Azure AI Search API key provided. Using DefaultAzureCredential.")
            credential = DefaultAzureCredential()

        mode = config.mode or "semantic"
        top_k = config.top_k or 5

        # Initialize the MAF context provider
        provider = AzureAISearchContextProvider(
            source_id="azure_ai_search",
            endpoint=endpoint,
            index_name=index_name,
            api_key=api_key if api_key else None,
            credential=credential,
            mode="agentic",
            top_k=top_k,
            # vector_field_name=config.vector_field_name,
            # semantic_configuration_name=config.semantic_configuration_name,
            context_prompt=config.context_prompt,
            model=config.model,
            azure_openai_resource_url=config.azure_openai_resource_url,
            azure_openai_api_key=config.azure_openai_api_key
        )
        return provider

    async def validate_connection(self, config: SearchConfigOverride) -> bool:
        """
        Asynchronously validates connection parameters by querying the index statistics.
        Uses advanced asynchronous SearchClient.
        """
        endpoint = config.endpoint or self._default_endpoint
        index_name = config.index_name or self._default_index
        api_key = config.api_key or self._default_key

        if not endpoint or not index_name:
            return False

        credential = AzureKeyCredential(api_key) if api_key else DefaultAzureCredential()

        try:
            # Asynchronous connection check
            async with SearchClient(endpoint=endpoint, index_name=index_name, credential=credential) as client:
                # Retrieve document count or stats as validation
                stats = await client.get_document_count()
                logger.info(f"Successfully validated connection to index '{index_name}'. Document count: {stats}")
                return True
        except AzureError as e:
            logger.error(f"Failed to connect to Azure AI Search index '{index_name}': {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating search index connection: {e}")
            return False
        finally:
            # If we used DefaultAzureCredential, clean up if needed
            if not api_key and hasattr(credential, "close"):
                await credential.close()
