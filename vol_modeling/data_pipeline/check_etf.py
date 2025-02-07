import databento as db

def is_etf(symbol, api_key):
    """
    Check if a given symbol is an ETF using the Databento API.

    Parameters:
        symbol (str): The stock symbol to check.
        api_key (str): Your Databento API key.

    Returns:
        bool: True if the symbol is an ETF, False otherwise.
        str: Additional information (e.g., error message or instrument type).
    """
    try:
        # Initialize the Databento Reference client
        client = db.Reference(key=api_key)

        # Query the symbol's metadata using security_master
        metadata = client.security_master.get_last(symbols=symbol)

        if metadata.empty:
            return False, "Symbol not found"

        # Check the security_type for ETF classification
        security_type = metadata['security_type'].iloc[0]
        
        if security_type == 'ETF':
            return True, "Symbol is classified as an ETF."

        return False, f"Symbol is not an ETF. Detected as {security_type}"

    except Exception as e:
        return False, f"Error occurred: {str(e)}"

# Example usage
if __name__ == "__main__":
    api_key = os.environ["DATABENTO_API_KEY"]
    symbol = "SPY"

    result, info = is_etf(symbol, api_key)
    print(f"Is {symbol} an ETF? {result}. Info: {info}")
