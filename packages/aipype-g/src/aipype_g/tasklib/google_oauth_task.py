"""Google OAuth Task - Standalone authentication task for Google APIs."""

from typing import List, Dict, Any, Optional

from typing import override

from aipype.framework.base_task import BaseTask
from aipype.framework.task_dependencies import TaskDependency
from aipype.framework.task_result import TaskResult
from .google_auth_service import (
    GoogleAuthService,
    GoogleAuthError,
    GMAIL_AND_SHEETS_SCOPES,
)


class GoogleOAuthTask(BaseTask):
    """Task that handles Google OAuth2 authentication for specified services."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        dependencies: Optional[List[TaskDependency]] = None,
    ):
        """Initialize Google OAuth task.

        Args:
            name: Task name
            config: Configuration dictionary containing:
                - scopes: List of Google API scopes (optional, defaults to Gmail+Sheets)
                - service_types: List of service types ["gmail", "sheets", "drive"] (alternative to scopes)
                - credentials_file: Path to OAuth2 credentials file (optional)
                - token_file: Path to OAuth2 token file (optional)
            dependencies: List of task dependencies
        """
        super().__init__(name, config)
        self.dependencies = dependencies or []
        self.validation_rules = {
            "defaults": {
                "scopes": GMAIL_AND_SHEETS_SCOPES,
                "service_types": [],
            },
            "types": {
                "scopes": list,
                "service_types": list,
                "credentials_file": str,
                "token_file": str,
            },
        }

    @override
    def get_dependencies(self) -> List[TaskDependency]:
        """Get the list of task dependencies.

        Returns:
            List of TaskDependency objects
        """
        return self.dependencies

    @override
    def run(self) -> TaskResult:
        """Perform Google OAuth2 authentication.

        Returns:
            TaskResult containing:
                - credentials: Authenticated credentials object (serialized)
                - scopes: List of authorized scopes
                - auth_info: Authentication metadata
                - service_access: Dictionary showing which services are accessible
        """
        from datetime import datetime

        start_time = datetime.now()

        # Validate configuration using instance validation rules
        validation_failure = self._validate_or_fail(start_time)
        if validation_failure:
            return validation_failure

        # Get configuration values
        scopes = self.config.get("scopes")
        service_types = self.config.get("service_types", [])
        credentials_file = self.config.get("credentials_file")
        token_file = self.config.get("token_file")

        self.logger.info(
            f"Starting Google OAuth authentication for services: {service_types or 'custom scopes'}"
        )

        try:
            # Create authentication service
            if service_types:
                auth_service = GoogleAuthService.create_service_with_scopes(
                    service_types=service_types,
                    credentials_file=credentials_file,
                    token_file=token_file,
                )
            else:
                auth_service = GoogleAuthService(
                    credentials_file=credentials_file,
                    token_file=token_file,
                    scopes=scopes,
                )

            # Perform authentication
            self.logger.debug("Initiating OAuth2 authentication flow...")
            credentials = auth_service.authenticate()

            # Prepare result data
            result_data = {
                "credentials": credentials,  # Actual credentials object for use by other tasks
                "scopes": auth_service.get_scopes(),
                "auth_info": auth_service.to_dict(),
                "service_access": {
                    "gmail": auth_service.has_gmail_access(),
                    "sheets": auth_service.has_sheets_access(),
                },
                "credentials_file": auth_service.credentials_file,
                "token_file": auth_service.token_file,
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            self.logger.info(
                f"Google OAuth authentication completed successfully: "
                f"Gmail={'[OK]' if auth_service.has_gmail_access() else '[X]'}, "
                f"Sheets={'[OK]' if auth_service.has_sheets_access() else '[X]'}"
            )

            return TaskResult.success(
                data=result_data,
                execution_time=execution_time,
                metadata={
                    "task_type": "google_oauth",
                    "scopes_count": len(auth_service.get_scopes()),
                    "gmail_access": auth_service.has_gmail_access(),
                    "sheets_access": auth_service.has_sheets_access(),
                },
            )

        except GoogleAuthError as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"GoogleOAuthTask authentication failed: {str(e)}"
            self.logger.error(error_msg)

            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_type": "google_oauth",
                    "error_type": "GoogleAuthError",
                    "scopes_requested": scopes or service_types,
                },
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"GoogleOAuthTask operation failed: {str(e)}"
            self.logger.error(error_msg)

            return TaskResult.failure(
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    "task_type": "google_oauth",
                    "error_type": type(e).__name__,
                    "scopes_requested": scopes or service_types,
                },
            )

    @staticmethod
    def create_gmail_auth_config(
        credentials_file: Optional[str] = None,
        token_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Helper to create configuration for Gmail-only authentication.

        Args:
            credentials_file: Path to OAuth2 credentials file
            token_file: Path to OAuth2 token file

        Returns:
            Configuration dictionary for GoogleOAuthTask
        """
        config: Dict[str, Any] = {
            "service_types": ["gmail"],
        }

        if credentials_file:
            config["credentials_file"] = credentials_file
        if token_file:
            config["token_file"] = token_file

        return config

    @staticmethod
    def create_sheets_auth_config(
        credentials_file: Optional[str] = None,
        token_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Helper to create configuration for Sheets-only authentication.

        Args:
            credentials_file: Path to OAuth2 credentials file
            token_file: Path to OAuth2 token file

        Returns:
            Configuration dictionary for GoogleOAuthTask
        """
        config: Dict[str, Any] = {
            "service_types": ["sheets"],
        }

        if credentials_file:
            config["credentials_file"] = credentials_file
        if token_file:
            config["token_file"] = token_file

        return config

    @staticmethod
    def create_combined_auth_config(
        credentials_file: Optional[str] = None,
        token_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Helper to create configuration for Gmail + Sheets authentication.

        Args:
            credentials_file: Path to OAuth2 credentials file
            token_file: Path to OAuth2 token file

        Returns:
            Configuration dictionary for GoogleOAuthTask
        """
        config: Dict[str, Any] = {
            "service_types": ["gmail", "sheets"],
        }

        if credentials_file:
            config["credentials_file"] = credentials_file
        if token_file:
            config["token_file"] = token_file

        return config

    @staticmethod
    def create_custom_scopes_config(
        scopes: List[str],
        credentials_file: Optional[str] = None,
        token_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Helper to create configuration with custom scopes.

        Args:
            scopes: List of Google API scopes
            credentials_file: Path to OAuth2 credentials file
            token_file: Path to OAuth2 token file

        Returns:
            Configuration dictionary for GoogleOAuthTask
        """
        config: Dict[str, Any] = {
            "scopes": scopes,
        }

        if credentials_file:
            config["credentials_file"] = credentials_file
        if token_file:
            config["token_file"] = token_file

        return config
