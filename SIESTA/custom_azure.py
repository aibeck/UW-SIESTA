from storages.backends.azure_storage import AzureStorage
from django.conf import settings

class AzureMediaStorage(AzureStorage):
    account_name = f'{settings.ACCOUNT_NAME}' # Must be replaced by your <storage_account_name>
    account_key = f'{settings.ACCOUNT_KEY}' # Must be replaced by your <storage_account_key>
    azure_container = f'{settings.MEDIA_LOCATION}'
    expiration_secs = None

class AzureStaticStorage(AzureStorage):
    account_name = f'{settings.ACCOUNT_NAME}' # Must be replaced by your storage_account_name
    account_key = f'{settings.ACCOUNT_KEY}' # Must be replaced by your <storage_account_key>
    azure_container = f'{settings.STATIC_LOCATION}'
    expiration_secs = None