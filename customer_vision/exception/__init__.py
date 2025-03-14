import sys
import traceback


def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error message including file name, line number, and error message.
    """
    exc_type, exc_value, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return f"Error: {error} (No traceback available)"

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = f"Error occurred in script: [{file_name}] at line [{line_number}] → {str(error)}"
    return error_message


class AppException(Exception):
    def __init__(self, error_message, error_detail: sys):
        """
        Custom exception class that logs detailed error messages.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message


# Example Usage
# if __name__ == "__main__":
#     try:
#         x = 1 / 0  # ZeroDivisionError
#     except Exception as e:
#         raise AppException(e, sys)
