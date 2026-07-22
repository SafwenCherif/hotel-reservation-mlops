from src.logger import get_logger
from src.custom_exception import CustomException
import sys

logger = get_logger(__name__)

def divide_numbers(a, b):
    try:    
        result = a / b
        logger.info(f"Division successful: {a} / {b} = {result}")
        return result
    except ZeroDivisionError as e:
        logger.error(f"Attempted to divide by zero: {a} / {b}")
        raise CustomException("Division by zero is not allowed.", sys) from e
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise CustomException("An unexpected error occurred during division.", sys) from e

if __name__ == "__main__":
    # Test cases
    try:
        print(divide_numbers(10, 0))  # Should log success and print 5.0
      # Should log error and raise CustomException
    except CustomException as ce:
        logger.error(f"CustomException caught: {ce}")