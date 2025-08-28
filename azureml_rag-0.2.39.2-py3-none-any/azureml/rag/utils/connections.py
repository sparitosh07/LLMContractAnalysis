# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""MLIndex auth connection utilities."""

import contextlib
import json
import os
import random
import re
from functools import lru_cache
from typing import Callable, Optional, Union

from azure.core.exceptions import ClientAuthenticationError

from azureml.rag.utils import backoff_retry_on_exceptions
from azureml.rag.utils.logging import get_logger, packages_versions_for_compatibility
from azureml.rag.utils.requests import create_session_with_retry, send_post_request

try:
    from azure.ai.ml.entities import WorkspaceConnection as Connection
except ImportError:
    from azure.ai.ml.entities import Connection

from azure.ai.ml.entities import NoneCredentialConfiguration

with contextlib.suppress(Exception):
    from azure.core.credentials import TokenCredential

logger = get_logger("connections")


def get_connection_credential(
    config, credential: Optional[Union[TokenCredential, object]] = None, data_plane: bool = True
):
    """Get a credential for a connection. by default for data plane operations."""
    try:
        from azure.core.credentials import AzureKeyCredential
    except ImportError as e:
        raise ValueError(
            "Could not import azure-core python package. Please install it with `pip install azure-core`."
        ) from e

    if config.get("connection_type", None) == "workspace_keyvault":
        from azureml.core import Run, Workspace

        run = Run.get_context()
        if hasattr(run, "experiment"):
            ws = run.experiment.workspace
        else:
            try:
                ws = Workspace(
                    subscription_id=config.get("connection", {}).get("subscription"),
                    resource_group=config.get("connection", {}).get("resource_group"),
                    workspace_name=config.get("connection", {}).get("workspace"),
                )
            except Exception as e:
                logger.warning(f"Could not get workspace '{config.get('connection', {}).get('workspace')}': {e}")
                # Fall back to looking for key in environment.
                import os

                key = os.environ.get(config.get("connection", {}).get("key"))
                if key is None:
                    raise ValueError(
                        f"Could not get workspace '{config.get('connection', {}).get('workspace')}' and no key named '{config.get('connection', {}).get('key')}' in environment"
                    ) from e
                return AzureKeyCredential(key)

        keyvault = ws.get_default_keyvault()
        connection_credential = AzureKeyCredential(keyvault.get_secret(config.get("connection", {}).get("key")))
    elif config.get("connection_type", None) == "workspace_connection":
        connection_id = config.get("connection", {}).get("id")
        connection = get_connection_by_id_v2(connection_id, credential=credential)
        connection_credential = connection_to_credential(connection, credential=credential, data_plane=data_plane)
    elif config.get("connection_type", None) == "environment":
        import os

        key = os.environ.get(config.get("connection", {}).get("key", "OPENAI_API_KEY"))
        connection_credential = (
            (credential if credential is not None else get_default_azure_credential())
            if key is None
            else AzureKeyCredential(key)
        )
    else:
        connection_credential = credential if credential is not None else get_default_azure_credential()

    return connection_credential


def workspace_connection_to_credential(connection: Union[dict, Connection]):
    """Get a credential for a workspace connection for control plane operations."""
    return connection_to_credential(connection=connection, data_plane=False)


def connection_to_credential(
    connection: Union[dict, Connection], credential: TokenCredential = None, data_plane: bool = True
):
    """
    Get a credential for a workspace connection.

    Args:
    ----
        connection: The connection object or dictionary.
        credential: If provided, it will be used to get credentials or token provider for AAD auth connection.
        data_plane: Get credential for a data plane or a control plane access. Default is data plane.

    """
    if isinstance(connection, dict):
        props = connection["properties"]
        auth_type = props.get("authType", props.get("AuthType"))
        if auth_type == "ApiKey":
            from azure.core.credentials import AzureKeyCredential

            return AzureKeyCredential(props["credentials"]["key"])
        elif auth_type == "PAT":
            from azure.core.credentials import AccessToken

            return AccessToken(props["credentials"]["pat"], props.get("expiresOn", None))
        elif auth_type == "AAD":
            if connection["properties"]["category"] == "CognitiveSearch":
                logger.info(
                    f"The '{connection.get('name', 'no-name connection')}' is an AAD auth type CognitiveSearch connection."
                )
                return get_obo_or_default_azure_credential() if credential is None else credential
            elif connection["properties"]["category"] == "AzureOpenAI":
                logger.info(
                    f"The '{connection.get('name', 'no-name connection')}' is an AAD auth type AzureOpenAI connection."
                )
                return _token_provider_for_aoai_connection(data_plane=data_plane, credential=credential)

            else:
                raise ValueError(
                    f"Unknown category '{connection['properties']['category']}' for AAD auth type connection"
                )

        elif auth_type == "CustomKeys":
            # OpenAI connections are made with CustomKeys auth, so we can try to access the key using known structure
            from azure.core.credentials import AzureKeyCredential

            if (
                connection.get("metadata", {})
                .get("azureml.flow.connection_type", connection.get("ApiType", connection.get("apiType", "")))
                .lower()
                == "openai"
            ):
                # Try to get the the key with api_key, if fail, default to regular CustomKeys handling
                try:
                    key = props["credentials"]["keys"]["api_key"]
                    return AzureKeyCredential(key)
                except Exception as e:
                    logger.warning(f"Could not get key using api_key, using default handling: {e}")
            key_dict = props["credentials"]["keys"]
            if len(key_dict.keys()) != 1:
                raise ValueError(
                    f"Only connections with a single key can be used. Number of keys present: {len(key_dict.keys())}"
                )
            first_key = next(iter(key_dict.keys()))
            return AzureKeyCredential(props["credentials"]["keys"][first_key])
        else:
            raise ValueError(f"Unknown auth type '{auth_type}'")
    elif isinstance(connection, Connection):
        if not connection.credentials or isinstance(connection.credentials, NoneCredentialConfiguration):
            if connection.properties.get("authType", connection.properties.get("AuthType", "")).lower() == "aad":
                logger.info(f"The connection '{connection.name}' is a {type(connection)} with AAD auth type.")
                try:
                    from azure.ai.ml.entities import AzureOpenAIWorkspaceConnection as AzureOpenAIConnection
                except ImportError:
                    from azure.ai.ml.entities import AzureOpenAIConnection

                if isinstance(connection, AzureOpenAIConnection):
                    return _token_provider_for_aoai_connection(data_plane=data_plane, credential=credential)

                return get_obo_or_default_azure_credential() if credential is None else credential

            raise ValueError(
                f"Unknown auth type '{connection.properties.get('authType', 'None')}' for connection '{connection.name}'"
            )
        elif connection.credentials.type.lower() == "api_key":
            from azure.core.credentials import AzureKeyCredential

            logger.info(f"The connection '{connection.name}' is a {type(connection)} with api_key auth type.")
            return AzureKeyCredential(connection.credentials.key)
        elif connection.credentials.type.lower() == "aad":
            logger.info(f"The connection '{connection.name}' is a {type(connection)} with AAD auth type.")
            try:
                from azure.ai.ml.entities import AzureOpenAIWorkspaceConnection as AzureOpenAIConnection
            except ImportError:
                from azure.ai.ml.entities import AzureOpenAIConnection

            if isinstance(connection, AzureOpenAIConnection):
                return _token_provider_for_aoai_connection(data_plane=data_plane, credential=credential)

            return get_obo_or_default_azure_credential() if credential is None else credential
        elif connection.credentials.type.lower() == "pat":
            from azure.core.credentials import AccessToken

            return AccessToken(connection.credentials.pat, 0)
        elif connection.credentials.type.lower() == "custom_keys":
            if connection._metadata.get("azureml.flow.connection_type", "").lower() == "openai":
                from azure.core.credentials import AzureKeyCredential

                try:
                    key = connection.credentials.keys.api_key
                    return AzureKeyCredential(key)
                except Exception as e:
                    logger.warning(f"Could not get key using api_key, using default handling: {e}")
            key_dict = connection.credentials.keys
            if len(key_dict.keys()) != 1:
                raise ValueError(
                    f"Only connections with a single key can be used. Number of keys present: {len(key_dict.keys())}"
                )
            first_key = next(iter(key_dict.keys()))
            return AzureKeyCredential(connection.credentials.keys[first_key])
        else:
            raise ValueError(f"Unknown auth type '{connection.credentials.type}' for connection '{connection.name}'")
    else:
        if connection.credentials.type.lower() == "api_key":
            from azure.core.credentials import AzureKeyCredential

            return AzureKeyCredential(connection.credentials.key)
        else:
            raise ValueError(f"Unknown auth type '{connection.credentials.type}' for connection '{connection.name}'")


def get_default_azure_credential():
    """
    Get the DefaultAzureCredential.

    managed_identity_client_id: The client ID of a user-assigned managed identity.
                                If DEFAULT_IDENTITY_CLIENT_ID is set, use it first.
                                Then, use the value of AZURE_CLIENT_ID if it is set.
                                If not specified, a system-assigned identity will be used.
    """
    try:
        from azure.identity import DefaultAzureCredential
    except ImportError as e:
        raise ValueError(
            "Could not import azure-identity python package. Please install it with `pip install azure-identity`."
        ) from e

    default_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID", None)
    if default_client_id:
        logger.info(f"Using DefaultAzureCredential with DEFAULT_IDENTITY_CLIENT_ID: {default_client_id}.")
        return DefaultAzureCredential(managed_identity_client_id=default_client_id, process_timeout=60)
    else:
        logger.info("Using DefaultAzureCredential.")
        logger.info(f"AZURE_CLIENT_ID: {os.environ.get('AZURE_CLIENT_ID', None)}")
        return DefaultAzureCredential(process_timeout=60)


def _token_provider_for_aoai_connection(
    data_plane: bool = True, credential: Optional[TokenCredential] = None
) -> Callable[[], str]:
    """Token provider for AOAI connection.data plane operations by default."""
    from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
    from azure.identity import get_bearer_token_provider

    scopes = "https://cognitiveservices.azure.com/.default" if data_plane else "https://management.azure.com/.default"

    if credential is not None:
        logger.info(f"Getting the token provider with provided {type(credential)} credential. scopes: {scopes}.")
        return get_bearer_token_provider(credential, scopes)

    is_obo_enabled = os.environ.get("AZUREML_OBO_ENABLED", "False") == "True"
    if is_obo_enabled:
        logger.info(f"Getting the token provider with AzureMLOnBehalfOfCredential. scopes: {scopes}.")

        def wrapper() -> str:
            credential = AzureMLOnBehalfOfCredential()
            return credential.get_token(scopes).token

        return wrapper
    else:
        logger.info(f"Getting the token provider with DefaultAzureCredential. scopes: {scopes}.")
        logger.info(f"AZURE_CLIENT_ID: {os.environ.get('AZURE_CLIENT_ID', None)}")
        return get_bearer_token_provider(get_default_azure_credential(), scopes)


def get_obo_or_default_azure_credential():
    """Get the On-Behalf-Of credential or the DefaultAzureCredential."""
    from azure.ai.ml.identity import AzureMLOnBehalfOfCredential

    is_obo_enabled = os.environ.get("AZUREML_OBO_ENABLED", "False") == "True"
    if is_obo_enabled:
        logger.info("Using User Identity for authentication.")
        credential = AzureMLOnBehalfOfCredential()
    else:
        logger.info("Using DefaultAzureCredential for authentication.")
        logger.info(f"AZURE_CLIENT_ID: {os.environ.get('AZURE_CLIENT_ID', None)}")
        credential = get_default_azure_credential()
    return credential


def snake_case_to_camel_case(s):
    """Convert snake case to camel case."""
    first = True
    final = ""
    for word in s.split("_"):
        if first:
            first = False
            final += word
        else:
            final += word.title()
    return final


def recursive_dict_keys_snake_to_camel(d: dict, skip_keys=[]) -> dict:
    """Convert snake case to camel case in dict keys."""
    new_dict = {}
    for k, v in d.items():
        if k not in skip_keys:
            if isinstance(v, dict):
                v = recursive_dict_keys_snake_to_camel(v, skip_keys=skip_keys)
            if isinstance(k, str):
                new_key = snake_case_to_camel_case(k)
                new_dict[new_key] = v
        else:
            new_dict[k] = v
    return new_dict


# Concurrent list_secrets calls will increase the chance of hitting ClientAuthenticationError on a deployed endpoint.
# To mitigate this issue, we will retry the list_secrets call with a backoff strategy.
@backoff_retry_on_exceptions(
    max_attempts=5, initial_delay=random.uniform(0, 1), max_delay=10, retry_exceptions={ClientAuthenticationError}
)
@lru_cache(maxsize=64)
def get_connection_by_id_v2(
    connection_id: str, credential: TokenCredential = None, client: str = "sdk"
) -> Union[dict, Connection]:
    """
    Get a connection by id using azure.ai.ml.

    If an AOAI or ACS connection uses AAD auth
        - azure.ai.ml >= v 1.14.0, set connection.properties["authType"] to "AAD"
        - azure.ai.ml <= v 1.13.0, set connection["properties"]["authType"] to "AAD"
    """
    uri_match = re.match(
        r"/subscriptions/(.*)/resourceGroups/(.*)/providers/Microsoft.MachineLearningServices/workspaces/(.*)/connections/(.*)",
        connection_id,
        flags=re.IGNORECASE,
    )

    if uri_match is None:
        logger.error(f"Invalid connection_id {connection_id}, expecting Azure Machine Learning resource ID")
        raise ValueError(f"Invalid connection id {connection_id}")

    subscription_id = uri_match.group(1)
    resource_group_name = uri_match.group(2)
    workspace_name = uri_match.group(3)
    connection_name = uri_match.group(4)

    logger.info(f"Getting workspace connection: {connection_name}, with input credential: {type(credential)}.")
    from azure.ai.ml import MLClient

    if credential is None:
        if os.environ.get("AZUREML_RUN_ID", None) is not None:
            from azureml.dataprep.api._aml_auth._azureml_token_authentication import AzureMLTokenAuthentication

            logger.info("Getting credential from AzureMLTokenAuthentication._initialize_aml_token_auth")
            credential = AzureMLTokenAuthentication._initialize_aml_token_auth()
        else:
            credential = get_default_azure_credential()

    if client == "sdk" and MLClient is not None and packages_versions_for_compatibility["azure-ai-ml"] >= "1.10.0":
        logger.info(
            f"Getting workspace connection via MLClient with auth: {type(credential)}, subscription_id: {subscription_id}, resource_group_name: {resource_group_name}, workspace_name: {workspace_name}."
        )

        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            workspace_name=workspace_name,
        )

        original_base_url = ml_client.connections._operation._client._base_url

        if os.environ.get("AZUREML_RUN_ID", None) is not None:
            # In AzureML Run context, we need to use workspaces internal endpoint that will accept AzureMLToken auth.
            ml_client.connections._operation._client._base_url = (
                f"{os.environ.get('AZUREML_SERVICE_ENDPOINT')}/rp/workspaces"
            )

        logger.info(
            f"Using ml_client base_url: {ml_client.connections._operation._client._base_url}, original_base_url: {original_base_url}."
        )

        list_secrets_response = None

        list_secrets_response = ml_client.connections._operation.list_secrets(
            connection_name=connection_name,
            resource_group_name=resource_group_name,
            workspace_name=workspace_name,
        )

        ml_client.connections._operation._client._base_url = original_base_url

        if list_secrets_response is None:
            raise ValueError(f"Failed to get connection: {connection_id}.")

        try:
            connection = Connection._from_rest_object(list_secrets_response)
            logger.info(f"Parsed Connection: {connection.id}")
            try:
                from azure.ai.ml.entities import AadCredentialConfiguration

                # There is to guarantee that all the aad credentialConfiguration has the authType
                # For example, For azure-ai-ml > 1.15.0, aoai connection does not have this property
                if isinstance(connection.credentials, AadCredentialConfiguration):
                    if connection.properties.get("authType") is None:
                        connection.properties["authType"] = "AAD"
            except ImportError:
                logger.info("Cannot import AadCredentialConfiguration as it is only for azure-ai-ml >= 1.16.0")
            # For azure-ai-ml-1.15.0 and below the ACS and AOAI connections with AAD authType won't have credentials
            if connection.credentials is None or isinstance(connection.credentials, NoneCredentialConfiguration):
                if connection.type == "custom":
                    from azure.core.credentials import AzureKeyCredential

                    connection.credentials = AzureKeyCredential(
                        list_secrets_response.properties.credentials.keys["api_key"]
                    )
                elif connection.type == "azure_open_ai":
                    # When the connection's AuthType is AAD, with azure-ai-ml >= 1.14.0,
                    # Connection._from_rest_object returns connection with credentials as None
                    try:
                        from azure.ai.ml.entities import AzureOpenAIWorkspaceConnection as AzureOpenAIConnection
                    except ImportError:
                        from azure.ai.ml.entities import AzureOpenAIConnection

                    if isinstance(connection, AzureOpenAIConnection):
                        connection.properties["authType"] = "AAD"
                elif connection.type == "cognitive_search" or connection.type == "azure_ai_search":
                    # For azure-ai-ml < 1.16.0, the type is cognitive_search
                    # For azure-ai-ml >= 1.16.0, the type is azure_ai_search
                    try:
                        from azure.ai.ml.entities import AzureAISearchWorkspaceConnection as AzureAISearchConnection
                    except ImportError:
                        from azure.ai.ml.entities import AzureAISearchConnection

                    if isinstance(connection, AzureAISearchConnection):
                        connection.properties["authType"] = "AAD"
                else:
                    raise Exception(f"Could not parse connection credentials for connection: {connection.id}")
        except Exception as e:
            logger.warning(f"Failed to parse connection into azure-ai-ml sdk object, returning response as is: {e}")
            connection = recursive_dict_keys_snake_to_camel(
                list_secrets_response.as_dict(), skip_keys=["credentials", "metadata"]
            )
            # When the connection's AuthType is AAD, with azure-ai-ml <= 1.13.0, Connection._from_rest_object
            # will throw an UnboundLocalError, the authType is None in the list_secrets_response.
            if isinstance(e, UnboundLocalError) and list_secrets_response.properties.auth_type is None:
                connection["properties"] = list_secrets_response.properties.as_dict()
                if connection["properties"]["category"] in ("AzureOpenAI", "CognitiveSearch"):
                    connection["properties"]["authType"] = "AAD"

    else:
        logger.info(f"Getting workspace connection via REST as fallback, with auth: {type(credential)}")
        return get_connection_by_id_v1(connection_id, credential)

    logger.info(f"Got connection: {connection_id} as {type(connection)}.")
    return connection


def get_id_from_connection(connection: Union[dict, Connection]) -> str:
    """Get a connection id from a connection."""
    if isinstance(connection, dict):
        return connection["id"]
    elif isinstance(connection, Connection):
        return connection.id
    else:
        raise ValueError(f"Unknown connection type: {type(connection)}")


def get_target_from_connection(connection: Union[dict, Connection]) -> str:
    """Get a connection target from a connection."""
    if isinstance(connection, dict):
        return connection["properties"]["target"]
    elif isinstance(connection, Connection):
        return connection.target
    else:
        raise ValueError(f"Unknown connection type: {type(connection)}")


def get_metadata_from_connection(connection: Union[dict, Connection]) -> dict:
    """Get a connection metadata from a connection."""
    if isinstance(connection, dict):
        return connection["properties"]["metadata"]
    elif isinstance(connection, Connection):
        return connection.tags
    else:
        raise ValueError(f"Unknown connection type: {type(connection)}")


def get_connection_by_name_v2(
    workspace, name: str, credential: Optional[Union[TokenCredential, object]] = None
) -> dict:
    """Get a connection from a workspace."""
    try:
        bearer_token = None
        if credential:
            bearer_token = credential.get_token("https://management.azure.com/.default").token
        if not bearer_token:
            if hasattr(workspace._auth, "get_token"):
                bearer_token = workspace._auth.get_token("https://management.azure.com/.default").token
            else:
                bearer_token = workspace._auth.token
    except Exception as e:
        raise ValueError("Fail to get bearer token.") from e

    endpoint = workspace.service_context._get_endpoint("api")
    url = f"{endpoint}/rp/workspaces/subscriptions/{workspace.subscription_id}/resourcegroups/{workspace.resource_group}/providers/Microsoft.MachineLearningServices/workspaces/{workspace.name}/connections/{name}/listsecrets?api-version=2023-02-01-preview"
    resp = send_post_request(url, {"Authorization": f"Bearer {bearer_token}", "content-type": "application/json"}, {})

    return resp.json()


def get_connection_by_id_v1(connection_id: str, credential: Optional[TokenCredential] = None) -> dict:
    """Get a connection from a workspace."""
    uri_match = re.match(
        r"/subscriptions/(.*)/resourceGroups/(.*)/providers/Microsoft.MachineLearningServices/workspaces/(.*)/connections/(.*)",
        connection_id,
    )

    if uri_match is None:
        logger.error(f"Invalid connection_id {connection_id}, expecting Azure Machine Learning resource ID")
        raise ValueError(f"Invalid connection id {connection_id}")

    from azureml.core import Run, Workspace

    run = Run.get_context()
    if hasattr(run, "experiment"):
        ws = run.experiment.workspace
    else:
        try:
            ws = Workspace(
                subscription_id=uri_match.group(1), resource_group=uri_match.group(2), workspace_name=uri_match.group(3)
            )
        except Exception as e:
            logger.warning(f"Could not get workspace '{uri_match.group(3)}': {e}")
            raise ValueError(f"Could not get workspace '{uri_match.group(3)}'") from e

    return get_connection_by_name_v2(ws, uri_match.group(4))


def send_put_request(url, headers, payload):
    """Send a PUT request."""
    with create_session_with_retry() as session:
        response = session.put(url, data=json.dumps(payload), headers=headers)
        # Raise an exception if the response contains an HTTP error status code
        response.raise_for_status()

    return response.json()


def create_connection_v2(workspace, name, category: str, target: str, auth_type: str, credentials: dict, metadata: str):
    """Create a connection in a workspace."""
    url = f"https://management.azure.com/subscriptions/{workspace.subscription_id}/resourcegroups/{workspace.resource_group}/providers/Microsoft.MachineLearningServices/workspaces/{workspace.name}/connections/{name}?api-version=2023-04-01-preview"

    resp = send_put_request(
        url,
        {
            "Authorization": f"Bearer {workspace._auth.get_token('https://management.azure.com/.default').token}",
            "content-type": "application/json",
        },
        {
            "properties": {
                "category": category,
                "target": target,
                "authType": auth_type,
                "credentials": credentials,
                "metadata": metadata,
            }
        },
    )

    return resp
