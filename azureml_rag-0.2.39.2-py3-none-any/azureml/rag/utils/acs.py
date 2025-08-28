# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Functions for interacting with AzureSearch."""

import threading
from collections import namedtuple
from contextlib import contextmanager

import langchain_community.vectorstores.azuresearch as azuresearch
from langchain_community.vectorstores.azuresearch import AzureSearch, AzureSearchVectorStoreRetriever

AzureSearchModuleSettings = namedtuple(
    "AzureSearchModuleSettings", ["FIELDS_ID", "FIELDS_CONTENT", "FIELDS_CONTENT_VECTOR", "FIELDS_METADATA"]
)

module_settings_lock = threading.Lock()


@contextmanager
def azuresearch_module_settings(module_settings: AzureSearchModuleSettings):
    """
    Context manager to temporarily set the module-level properties for azuresearch.

    Args:
        module_settings (AzureSearchModuleSettings): New langchain AzureSearch module settings.

    Yields:
        None

    """
    # Save the current settings
    current_settings = AzureSearchModuleSettings(
        FIELDS_ID=azuresearch.FIELDS_ID,
        FIELDS_CONTENT=azuresearch.FIELDS_CONTENT,
        FIELDS_CONTENT_VECTOR=azuresearch.FIELDS_CONTENT_VECTOR,
        FIELDS_METADATA=azuresearch.FIELDS_METADATA,
    )

    module_settings_lock.acquire()
    # Set the new settings
    azuresearch.FIELDS_ID = module_settings.FIELDS_ID
    azuresearch.FIELDS_CONTENT = module_settings.FIELDS_CONTENT
    azuresearch.FIELDS_CONTENT_VECTOR = module_settings.FIELDS_CONTENT_VECTOR
    azuresearch.FIELDS_METADATA = module_settings.FIELDS_METADATA

    try:
        yield
    finally:
        # Restore the original settings
        azuresearch.FIELDS_ID = current_settings.FIELDS_ID
        azuresearch.FIELDS_CONTENT = current_settings.FIELDS_CONTENT
        azuresearch.FIELDS_CONTENT_VECTOR = current_settings.FIELDS_CONTENT_VECTOR
        azuresearch.FIELDS_METADATA = current_settings.FIELDS_METADATA
        module_settings_lock.release()


class AzureSearchProxy:
    """
    Proxy class to intercept method calls and apply temporary settings for AzureSearch.

    Args:
        azuresearch_instance (AzureSearch): The AzureSearch instance to wrap.
        fields_id (str): The ID field setting.
        fields_content (str): The content field setting.
        fields_content_vector (str): The content vector field setting.
        fields_metadata (str): The metadata field setting.

    """

    def __init__(self, azuresearch_instance: AzureSearch, module_settings: AzureSearchModuleSettings):
        """Saving azuresearch module settings."""
        self.azuresearch_instance = azuresearch_instance
        self.module_settings = module_settings

    def __getattr__(self, name):
        """
        Intercept method calls and apply temporary settings.

        Args:
            name (str): The name of the attribute or method being accessed.

        Returns:
            callable: A wrapped method that applies temporary settings.

        """
        original_attr = getattr(self.azuresearch_instance, name)

        if callable(original_attr):

            def hooked(*args, **kwargs):
                with azuresearch_module_settings(self.module_settings):
                    return original_attr(*args, **kwargs)

            return hooked
        else:
            return original_attr

    def get_module_settings(self):
        """
        Return module settings for this instance.

        Returns:
            Temporary settings.

        """
        return self.module_settings

    def get_azuresearch_instance(self):
        """
        Return the instance of AzureSearch served by this proxy.

        Returns:
            AzureSearch object.

        """
        return self.azuresearch_instance


class AzureSearchVectorStoreRetrieverProxy:
    """
    Proxy class to intercept method calls and apply temporary settings for AzureSearchVectorStoreRetriever.

    Args:
        retriever_instance (AzureSearchVectorStoreRetriever): The AzureSearch instance to wrap.
        fields_id (str): The ID field setting.
        fields_content (str): The content field setting.
        fields_content_vector (str): The content vector field setting.
        fields_metadata (str): The metadata field setting.

    """

    def __init__(self, retriever_instance: AzureSearchVectorStoreRetriever, module_settings: AzureSearchModuleSettings):
        """Saving azuresearch module settings."""
        self.retriever_instance = retriever_instance
        self.module_settings = module_settings

    def __getattr__(self, name):
        """
        Intercept method calls and apply temporary settings.

        Args:
            name (str): The name of the attribute or method being accessed.

        Returns:
            callable: A wrapped method that applies temporary settings.

        """
        original_attr = getattr(self.retriever_instance, name)

        if callable(original_attr):

            def hooked(*args, **kwargs):
                with azuresearch_module_settings(self.module_settings):
                    return original_attr(*args, **kwargs)

            return hooked
        else:
            return original_attr
